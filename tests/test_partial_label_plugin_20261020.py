"""Plugin wiring tests with a minimal SDK double; real SDK check runs on server."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from ecospec_kg import partial_label_v226 as p


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode('ascii')), 'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class Base:
    def __init__(self):
        self.is_training = True
        self.tokenizer = Tokenizer()
    def _encode(self, inputs):
        if not self.is_training:
            return {'input_ids': [9, 10]}
        answer = inputs.messages[-1]['content']
        # Replicate SDK mutation after caller captured the raw answer.
        inputs.messages[-1]['content'] = '<think></think>' + answer
        ids = [9, 10] + self.tokenizer(answer)['input_ids'] + [3]
        return {'input_ids': ids, 'labels': [-100, -100] + ids[2:]}
    def encode(self, row):
        inputs = SimpleNamespace(messages=deepcopy(row['messages'][1:]), channel=row.get('channel'))
        result = self._encode(inputs)
        if inputs.channel:
            result['channel'] = inputs.channel
        return result


class Plugin(unittest.TestCase):
    def setUp(self):
        modules = {n: ModuleType(n) for n in ['swift', 'swift.template', 'swift.template.register',
                                            'swift.template.templates', 'swift.template.templates.qwen']}
        self.mapping = {'qwen3_5': SimpleNamespace(template_type='qwen3_5', template_cls=Base)}
        modules['swift.template.register'].TEMPLATE_MAPPING = self.mapping
        modules['swift.template'].register_template = lambda meta: self.mapping.update({meta.template_type: meta})
        modules['swift.template.templates.qwen'].Qwen3_5Template = Base
        path = Path(__file__).resolve().parents[1] / 'tools/swift_partial_label_plugin_20261020.py'
        spec = importlib.util.spec_from_file_location('_fixture_partial_plugin', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch('importlib.metadata.version', return_value='4.4.2'):
            spec.loader.exec_module(module)
        self.template = module.PartialLabelTemplate()
        prompt = json.dumps({'candidate_entities': [{'id': 'a'}], 'candidate_relations': [{'id': 'r1'}, {'id': 'r2'}]})
        answer = p.answer_for(prompt, {'selected_entity_ids': ['a'], 'selected_relation_ids': ['r1']})
        self.row = {'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': prompt},
                                 {'role': 'assistant', 'content': answer}],
                    'channel': p.supervision_channel(prompt, answer, ['r2'])}

    def test_training_mask_survives_base_mutation_and_channel_removed(self):
        before = deepcopy(self.row)
        encoded = self.template.encode(self.row)
        baseline = Base().encode(self.row)
        expected, proof = p.mask_encoded(baseline, self.template.tokenizer, self.row['messages'][1]['content'],
                                         self.row['messages'][2]['content'], self.row['channel'])
        self.assertEqual(encoded['labels'], expected['labels'])
        self.assertEqual(len(proof['unknown_token_indices']), 1)
        self.assertNotIn('channel', encoded)
        self.assertEqual(self.row, before)
        self.assertEqual(self.mapping['qwen3_5'].template_cls, Base)

    def test_metadata_cannot_be_omitted_in_training_or_supplied_in_inference(self):
        row = deepcopy(self.row); row.pop('channel')
        with self.assertRaises(ValueError):
            self.template.encode(row)
        self.template.is_training = False
        with self.assertRaises(ValueError):
            self.template.encode(self.row)
        self.assertEqual(self.template.encode({'messages': row['messages'][:2]}), {'input_ids': [9, 10]})


if __name__ == '__main__':
    unittest.main()
