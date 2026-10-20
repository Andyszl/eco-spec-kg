import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import inspect_partial_label_runtime_20261020 as inspect_runtime


@pytest.fixture
def distribution(tmp_path):
    files={
        'swift/template/base.py':b'class Template:\n    def encode(self): pass\n',
        'swift/template/templates/qwen.py':b'class Qwen3_5Template: pass\n',
        'swift/plugin/loss_scale/base.py':b'raise RuntimeError("must never import")\nclass LossScale: pass\n',
        'swift/plugin/loss.py':b'LOSSES = {}\n',
        'swift/trainers/trainers.py':b'def compute_loss(): pass\n',
        'swift/.env':b'private-do-not-read',
        'swift/model/model.py':b'private weights config must not be read\nloss_scale=1',
    }
    for n,data in files.items():
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return SimpleNamespace(version='4.4.2',files=list(files),locate_file=lambda n:tmp_path/n)


def test_collect_only_package_interfaces_without_import_or_source_mutation(distribution,tmp_path):
    original={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    out=tmp_path/'out';report=inspect_runtime.collect(distribution,out)
    assert report['status']=='runtime_interfaces_collected'
    assert report['training_ready'] is False and report['model_calls']==0
    assert report['partial_label_loss_implemented'] is False
    assert report['file_count']==5
    assert report['template_files'] and report['loss_files']
    for n,meta in report['files'].items():
        assert (out/'source'/n).read_bytes()==distribution.locate_file(n).read_bytes()
        assert hashlib.sha256((out/'source'/n).read_bytes()).hexdigest()==meta['sha256']
    assert not (out/'source/swift/.env').exists()
    assert not (out/'source/swift/model/model.py').exists()
    assert all(Path(n).read_bytes()==data for n,data in original.items())


@pytest.mark.parametrize('kind',['version','no_loss','outside','large'])
def test_invalid_or_unsafe_package_fails_before_output(distribution,tmp_path,kind):
    if kind=='version':distribution.version='different'
    elif kind=='no_loss':
        distribution.files=['swift/template/base.py']
        for name in ('swift/plugin/loss.py','swift/plugin/loss_scale/base.py','swift/trainers/trainers.py'):
            distribution.locate_file(name).unlink()
    elif kind=='outside':distribution.files.append('swift/template/../../secret.py')
    else:(distribution.locate_file('swift/template/base.py')).write_bytes(b'X'*(inspect_runtime.MAX_FILE+1))
    with pytest.raises((AssertionError,ValueError)):
        inspect_runtime.collect(distribution,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_existing_output_preserved(distribution,tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'keep').write_bytes(b'original')
    with pytest.raises(AssertionError,match='Preserve'):
        inspect_runtime.collect(distribution,out)
    assert (out/'keep').read_bytes()==b'original'


def test_visible_gpu_fails_before_package_lookup(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(sys,'argv',['inspect','--out',str(tmp_path/'out')])
    with pytest.raises(AssertionError,match='hide GPUs'):inspect_runtime.main()
    assert not (tmp_path/'out').exists()


def test_cpu_entry_records_code_provenance_without_importing_swift(distribution,tmp_path,monkeypatch):
    out=tmp_path/'out'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(sys,'argv',['inspect','--out',str(out)])
    monkeypatch.setattr(inspect_runtime.importlib.metadata,'distribution',lambda name:distribution)
    monkeypatch.setattr(inspect_runtime,'resolve_package_root',lambda d:tmp_path/'swift')
    before=set(sys.modules)
    inspect_runtime.main()
    assert not any(n=='swift' or n.startswith('swift.') for n in set(sys.modules)-before)
    provenance=json.loads((out/'provenance.json').read_text())
    assert len(provenance['code_commit'])==40 and len(provenance['tool_sha256'])==64


def test_wrapper_keeps_offline_environment_and_hides_gpu():
    text=(ROOT/'tools/run_partial_label_runtime_inspection_20261020.sh').read_text(encoding='utf-8')
    assert 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""' in text


def test_unlisted_physical_sources_are_discovered_read_only(distribution,tmp_path):
    distribution.files=['ms_swift-4.4.2.dist-info/METADATA']
    report=inspect_runtime.collect(distribution,tmp_path/'out')
    assert report['file_count']==5
    assert report['source_discovery']['metadata_source_files']==0
    assert report['source_discovery']['filesystem_source_files']==5


def test_legacy_llm_package_layout_is_classified(distribution,tmp_path):
    sdk=tmp_path/'swift'
    (sdk/'llm').mkdir()
    for name in ('template','plugin'):(sdk/name).rename(sdk/'llm'/name)
    distribution.files=['ms_swift-4.4.2.dist-info/METADATA']
    report=inspect_runtime.collect(distribution,tmp_path/'out')
    assert any(n.startswith('swift/llm/template/') for n in report['template_files'])
    assert any(n.startswith('swift/llm/plugin/') for n in report['loss_files'])


def test_actual_package_root_overrides_incomplete_metadata_location(distribution,tmp_path):
    actual=tmp_path/'swift';distribution.locate_file=lambda n:tmp_path/'wrong-site-packages'/n
    distribution.files=None
    report=inspect_runtime.collect(distribution,tmp_path/'out',package_root=actual)
    assert report['file_count']==5 and report['package_root']==str(actual.resolve())


def test_missing_interfaces_keep_directory_diagnosis_in_failed_job(distribution,tmp_path,monkeypatch):
    for name in ('swift/plugin/loss.py','swift/plugin/loss_scale/base.py','swift/trainers/trainers.py'):
        distribution.locate_file(name).unlink()
    distribution.files=['swift/template/base.py']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(sys,'argv',['inspect','--out',str(tmp_path/'job')])
    monkeypatch.setattr(inspect_runtime.importlib.metadata,'distribution',lambda name:distribution)
    monkeypatch.setattr(inspect_runtime,'resolve_package_root',lambda d:tmp_path/'swift',raising=False)
    with pytest.raises(AssertionError,match='interfaces missing'):inspect_runtime.main()
    report=json.loads((tmp_path/'inspection_failure.json').read_text())
    assert report['diagnostics']['template_files'] and report['diagnostics']['loss_files']==[]
    assert report['diagnostics']['package_subdirectories']
    assert len(report['provenance']['tool_sha256'])==64
    assert not (tmp_path/'job').exists()


def test_find_spec_resolves_package_without_executing_it(tmp_path,monkeypatch):
    sdk=tmp_path/'swift';sdk.mkdir()
    (sdk/'__init__.py').write_text('raise RuntimeError("must not execute Swift")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    before=set(sys.modules)
    assert inspect_runtime.resolve_package_root(None)==sdk.resolve()
    assert 'swift' not in set(sys.modules)-before
