"""Read installed Swift Python interfaces for partial-label loss preparation.

This does not implement loss masking, import Swift, load a model or tokenizer,
read environment/config files, change data, or start training.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import platform
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MAX_FILE=2*1024*1024
MAX_TOTAL=8*1024*1024
PREFIXES=('swift/template/','swift/plugin/','swift/trainers/','swift/arguments/',
          'swift/dataset/','swift/pipelines/train/',
          'swift/llm/template/','swift/llm/plugin/','swift/llm/trainers/',
          'swift/llm/argument/','swift/llm/dataset/','swift/llm/train/')
MODULES=('swift/template.py','swift/plugin.py','swift/loss.py','swift/loss_scale.py',
         'swift/llm/template.py','swift/llm/plugin.py','swift/llm/loss.py','swift/llm/loss_scale.py')
PATTERNS=(r'loss_scale',r'class\s+(?:Qwen\w*Template|Template\b|StdTemplateInputs\b|TemplateInputs\b|LossScale\b)',
          r'def\s+(?:get_template|register_template|compute_loss|_compute_loss|encode|_encode)\s*\(',
          r'class\s+RowPreprocessor\b')


def allowed(name):
    return name.endswith('.py') and (name.startswith(PREFIXES) or name in MODULES)


def resolve_package_root(distribution):
    # find_spec for a top-level name resolves the importer without executing
    # Swift's __init__.py, unlike importing the package to find __file__.
    spec=importlib.util.find_spec('swift')
    assert spec is not None,'Swift package cannot be resolved by this Python'
    locations=list(spec.submodule_search_locations or ())
    assert len(locations)==1,'Swift import directory is missing or ambiguous'
    return Path(locations[0]).resolve()


def collect(distribution,out,package_root=None,diagnostics=None):
    assert not out.exists(),'Preserve existing inspection output'
    assert distribution.version=='4.4.2','Expected the frozen ms-swift 4.4.2 runtime'
    package=Path(package_root or distribution.locate_file('swift')).resolve()
    diagnostic=diagnostics if diagnostics is not None else {}
    metadata=sorted({str(n).replace('\\','/') for n in distribution.files or ()})
    diagnostic.update(package_root=str(package),metadata_package_root=str(distribution.locate_file('swift')),
                      metadata_file_count=len(metadata),metadata_source_files=0)
    for name in metadata:
        if not allowed(name):continue
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name:
            raise ValueError('Unsafe package source path: '+name)
        diagnostic['metadata_source_files']+=1
    assert package.is_dir(),'Resolved Swift source directory does not exist'
    diagnostic['package_subdirectories']=sorted(p.name for p in package.iterdir()
        if p.is_dir() and not p.is_symlink() and p.name!='__pycache__')[:64]
    llm=package/'llm'
    if llm.is_dir() and not llm.is_symlink():
        diagnostic['llm_subdirectories']=sorted(p.name for p in llm.iterdir()
            if p.is_dir() and not p.is_symlink() and p.name!='__pycache__')[:64]
    # Distribution RECORD metadata can omit live files in editable/custom
    # installs. Inspect the resolved package itself, retaining the same source
    # directory allowlist and all path/size/version checks.
    physical=[];enumerated=0
    for file in package.rglob('*.py'):
        enumerated+=1;assert enumerated<=4096,'Too many installed Python source paths'
        name='swift/'+file.relative_to(package).as_posix()
        if allowed(name):physical.append((name,file))
    physical.sort()
    diagnostic.update(filesystem_source_files=len(physical),
                      available_source_paths=[n for n,_ in physical[:256]])
    selected={};payload={};total=0
    for name,file in physical:
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name:
            raise ValueError('Unsafe package source path: '+name)
        file=file.resolve()
        if not file.is_relative_to(package):
            raise ValueError('Package source escapes the installed Swift directory')
        assert file.is_file(),'Missing installed source: '+name
        assert file.stat().st_size<=MAX_FILE,'Unexpectedly large interface source: '+name
        data=file.read_bytes();text=data.decode('utf-8-sig')
        reasons=[p for p in PATTERNS if re.search(p,text)]
        # Loss plugin registration and Qwen template support may be delegated
        # to sibling modules; preserve those small package files as context.
        if 'loss' in path.parts or 'loss_scale' in path.parts or 'loss' in path.stem:
            reasons.append('loss plugin source')
        if '/template/' in name and path.name.startswith('qwen'):reasons.append('Qwen template source')
        if not reasons:continue
        total+=len(data);assert total<=MAX_TOTAL,'Inspection source limit exceeded'
        assert len(selected)<128,'Too many interface sources'
        selected[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
            'reasons':reasons,'matched_lines':[n for n,line in enumerate(text.splitlines(),1)
                if any(re.search(pattern,line) for pattern in PATTERNS)]}
        payload[name]=data
    templates=[n for n in selected if n.startswith(('swift/template/','swift/llm/template/')) or n in ('swift/template.py','swift/llm/template.py')]
    losses=[n for n in selected if ('loss' in n or b'compute_loss' in payload[n])
            and not n.startswith(('swift/template/','swift/llm/template/'))]
    diagnostic.update(template_files=templates,loss_files=losses,selected_source_files=len(selected))
    assert templates and losses,'Template or loss interfaces missing; no training preparation completed'
    # Stage only after all source/path/size/category checks have passed.
    out.mkdir(parents=True)
    for name,data in payload.items():
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    report={'status':'runtime_interfaces_collected','distribution':'ms-swift','version':distribution.version,
        'python':platform.python_version(),'platform':platform.platform(),'package_root':str(package),
        'file_count':len(selected),'total_source_bytes':total,'files':selected,
        'template_files':templates,'loss_files':losses,
        'source_discovery':diagnostic,
        'model_calls':0,'training_started':False,'training_ready':False,
        'partial_label_loss_implemented':False,'tokenizer_loaded':False,
        'scope':'Installed package Python source inspection only; partial-label design and real token/mask preflight remain pending.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU inspection must hide GPUs'
    assert not args.out.exists(),'Preserve existing inspection output'
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    provenance={'code_commit':commit,'tool_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()}
    diagnostics={}
    try:
        distribution=importlib.metadata.distribution('ms-swift')
        diagnostics['distribution_version']=distribution.version
        diagnostics['metadata_package_root']=str(distribution.locate_file('swift'))
        package=resolve_package_root(distribution)
        report=collect(distribution,args.out,package_root=package,diagnostics=diagnostics)
    except Exception as error:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        failure=args.out.parent/'inspection_failure.json'
        if not failure.exists():
            failure.write_text(json.dumps({'status':'failed','type':type(error).__name__,
                'message':str(error),'diagnostics':diagnostics,'provenance':provenance,'training_ready':False},
                ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        raise
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':report['status'],'version':report['version'],'source_files':report['file_count'],
        'source_bytes':report['total_source_bytes'],'training_ready':False}),flush=True)
    print('RUNTIME_INSPECTION_OK=true\nTRAINING_READY=false',flush=True)


if __name__=='__main__':main()
