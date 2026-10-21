"""Finite binary grammar and real dev HTTP protocol; no Qwen model in local tests."""
import asyncio,io,itertools,json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from ecospec_kg import binary_decoding_v230 as decoding
from ecospec_kg import partial_label_v226 as policy
from test_experiment_chain_v2 import source_unit
from ecospec_kg.candidate_profile_v228 import candidates_for_unit
from test_unique_selection_v2 import AsciiTokenizer,assert_path
VERSION="binary-candidate-json-v230"
E1,E2,R1,R2=(c*16 for c in "1234")

@pytest.fixture(name="decoding")
def decoding_fixture():return decoding

def wire_prompt():
    u=source_unit();return policy.build_messages(u,candidates_for_unit(u))

@pytest.mark.parametrize("ec,rc",[(0,0),(2,2),(0,2),(3,0)])
def test_all_binary_choices_and_exact_counts(ec,rc):
    es=tuple(str(i+1)*16 for i in range(ec));rs=(R1,R2)[:rc]
    d=decoding.BinaryDecisionConstraint(AsciiTokenizer(),es,rs,[999])
    for bits in itertools.product('01',repeat=ec+rc):
        text=json.dumps(dict(zip(policy.FIELDS,[' '.join(bits[:ec]),' '.join(bits[ec:])])),separators=(',',':'))
        assert_path(d,d.tokenizer.encode(text))
        assert len(text)+1==d.max_tokens
    for bad in ['{}','{"entity_decisions":"2"','{"entity_decisions":"1 1 1 1"']:
        with pytest.raises(ValueError):d.allowed(d.tokenizer.encode(bad))

def test_merged_token_crosses_binary_boundary_and_both_decisions_remain_free():
    class Merged(AsciiTokenizer):
        merged={'0 1':1000,'1 0':1001,'","relation_decisions":"':1002}
        def get_vocab(self):return {**super().get_vocab(),**self.merged}
        def encode(self,text,**kw):
            out=[]
            while text:
                s=next((s for s in self.merged if text.startswith(s)),text[0]);out.append(self.merged.get(s,ord(s[0])));text=text[len(s):]
            return out
        def decode(self,ids,**kw):
            reverse={v:k for k,v in self.merged.items()};return ''.join(reverse[i] if i in reverse else chr(i) for i in ids if i!=999)
    d=decoding.BinaryDecisionConstraint(Merged(),(E1,E2),(R1,R2),[999])
    for e,r in [('0 1','1 0'),('1 0','0 1')]:
        text=json.dumps(dict(zip(policy.FIELDS,[e,r])),separators=(',',':'));assert_path(d,d.tokenizer.encode(text))
    prefix=d.tokenizer.encode('{"entity_decisions":"');allowed=d.allowed(prefix)
    assert ord('0') in allowed and ord('1') in allowed and 999 not in allowed

@pytest.fixture
def binary_http():
    from ecospec_kg import binary_decoding_v230 as b
    state=dict(calls=[],fail=False,bad_proof=False,models=['eco-pair-base','eco-pair-control','eco-pair-repair'])
    class Reply(io.BytesIO):
        def __enter__(self):return self
        def __exit__(self,*a):self.close()
    def respond(request,**kw):
        if request.full_url.endswith('/models'):value={'data':[{'id':m} for m in state['models']]}
        elif request.full_url.endswith('/ecospec-binary-decoding'):
            value=dict(version=b.VERSION,implementation=b.implementation_hashes(),backend='transformers',swift_version='4.4.2')
        else:
            p=json.loads(request.data);system=p['messages'][0]['content'];prompt=p['messages'][1]['content'];q=json.loads(prompt)
            assert p['chat_template_kwargs']=={'enable_thinking':False,b.MARKER:b.VERSION}
            assert p['temperature']==0 and p['seed']==42
            assert 'channel' not in q and 'unknown_relation_ids' not in q
            state['calls'].append((p['model'],q['source_unit']['unit_id'],prompt))
            if state['fail']:raise TimeoutError('synthetic failure')
            chosen=dict(selected_entity_ids=policy.ordered_ids(prompt)[0],selected_relation_ids=policy.ordered_ids(prompt)[1])
            answer=policy.answer_for(prompt,chosen)
            proof={**b.request_identity(system,prompt),'prefix_calls':1,'max_completion_tokens':len(answer)+1}
            if state['bad_proof']:proof['prompt_hash']='bad'
            value={'choices':[{'finish_reason':'stop','message':{'content':answer}}],b.MARKER:proof}
        return Reply(json.dumps(value).encode())
    state['respond']=respond;return state

@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message
    finish_reason: str = "stop"


@dataclass
class Response:
    choices: list


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))
    def tolist(self): return self.values


@pytest.fixture
def hooked(decoding):
    class Engine:
        tokenizer = AsciiTokenizer()
        max_batch_size = 1
        use_prefix = True

        def _infer(self, infer_requests, request_config, *, adapter_request=None, pre_infer_hook=None):
            if not infer_requests[0].chat_template_kwargs.get("ecospec_binary_decoding"):
                return ["legacy result"]
            kwargs = {"inputs": {"input_ids": Tensor([11, 22, 33])},
                "generation_config": SimpleNamespace(eos_token_id=[999], max_new_tokens=request_config.max_tokens)}
            kwargs = pre_infer_hook(kwargs)
            prefix = kwargs["inputs"]["prefix_allowed_tokens_fn"]
            ids = []
            if self.use_prefix:
                for _ in range(request_config.max_tokens):
                    token = min(prefix(0, Tensor([11, 22, 33] + ids)))
                    if token == 999: break
                    ids.append(token)
                text = self.tokenizer.decode(ids)
            else:
                text = '{"entity_decisions":"","relation_decisions":""}'
            return [Response([Choice(Message(text))])]

    class App:
        routes = {}
        def get(self, path):
            return lambda fn: self.routes.setdefault(path, fn)

    class Deploy:
        args = SimpleNamespace(infer_backend="transformers")
        infer_engine = Engine()
        app = App()
        def _register_app(self): pass
        async def create_chat_completion(self, request, raw_request, *, return_cmpl_response=False):
            return self.infer_engine._infer([request], request)[0]

    decoding.install_swift_hooks(Engine, Deploy, swift_version="4.4.2",
        json_response=lambda payload, status_code=200: {"status_code": status_code, "body": payload})
    system, prompt = wire_prompt()
    request = SimpleNamespace(messages=[{"role": "system", "content": system},
        {"role": "user", "content": prompt}], chat_template_kwargs={"enable_thinking": False,
        "ecospec_binary_decoding": VERSION}, stream=False, n=1, num_beams=1, temperature=0, max_tokens=3072)
    return Engine, Deploy, request


def test_swift_hook_passes_constraint_and_returns_verified_proof(decoding, hooked):
    Engine, Deploy, request = hooked
    service = Deploy()
    service._register_app()
    capabilities = asyncio.run(service.app.routes["/v1/ecospec-binary-decoding"]())
    assert capabilities["version"] == VERSION
    assert capabilities["implementation"] == decoding.implementation_hashes()
    original_hook_calls = []
    response = Engine()._infer([request], request,
        pre_infer_hook=lambda kwargs: original_hook_calls.append(kwargs) or kwargs)[0]
    assert original_hook_calls
    reply = asyncio.run(service.create_chat_completion(request, None))
    assert reply["status_code"] == 200
    metadata = reply["body"]["ecospec_binary_decoding"]
    assert metadata["prefix_calls"] > 0
    decoding.verify_response(reply["body"], request.messages[0]["content"], request.messages[1]["content"])
    assert Engine()._infer([SimpleNamespace(chat_template_kwargs={})], request) == ["legacy result"]


@pytest.mark.parametrize("fault", ["budget", "sampling", "stream", "thinking", "unsupported", "missing_callback"])
def test_hook_fails_closed_instead_of_claiming_unconstrained_success(hooked, fault):
    Engine, Deploy, request = hooked
    service = Deploy()
    if fault == "budget": request.max_tokens = 10
    elif fault == "sampling": request.temperature = 0.7
    elif fault == "stream": request.stream = True
    elif fault == "thinking": request.chat_template_kwargs["enable_thinking"] = True
    elif fault == "unsupported": request.chat_template_kwargs["ecospec_binary_decoding"] = "other"
    elif fault == "missing_callback": service.infer_engine.use_prefix = False
    reply = asyncio.run(service.create_chat_completion(request, None))
    assert reply["status_code"] == 400
    # An error must not kill the engine worker or change unmarked requests.
    assert Engine()._infer([SimpleNamespace(chat_template_kwargs={})], request) == ["legacy result"]


@pytest.mark.parametrize("field", ["version", "prompt_hash", "implementation", "prefix_calls"])
def test_mismatched_server_proof_rejected(decoding, hooked, field):
    _, Deploy, request = hooked
    reply = asyncio.run(Deploy().create_chat_completion(request, None))["body"]
    reply["ecospec_binary_decoding"][field] = None
    with pytest.raises(ValueError):
        decoding.verify_response(reply, request.messages[0]["content"], request.messages[1]["content"])
