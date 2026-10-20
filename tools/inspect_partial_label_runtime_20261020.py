"""Read installed Swift Python interfaces for partial-label loss preparation.

This does not implement loss masking, import Swift, load a model or tokenizer,
read environment/config files, change data, or start training.
"""
import argparse
import hashlib
import importlib.metadata
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
          'swift/dataset/','swift/pipelines/train/')
PATTERNS=(r'loss_scale',r'class\s+(?:Qwen\w*Template|Template\b|StdTemplateInputs\b|TemplateInputs\b|LossScale\b)',
          r'def\s+(?:get_template|register_template|compute_loss|_compute_loss|encode|_encode)\s*\(',
          r'class\s+RowPreprocessor\b')


def collect(distribution,out):
    assert not out.exists(),'Preserve existing inspection output'
    assert distribution.version=='4.4.2','Expected the frozen ms-swift 4.4.2 runtime'
    assert distribution.files,'Installed package file metadata unavailable'
    package=Path(distribution.locate_file('swift')).resolve()
    selected={};payload={};total=0
    for name in sorted({str(n).replace('\\','/') for n in distribution.files}):
        if not name.startswith(PREFIXES) or not name.endswith('.py'):
            continue
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name:
            raise ValueError('Unsafe package source path: '+name)
        file=Path(distribution.locate_file(name)).resolve()
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
        if name.startswith('swift/template/') and path.name.startswith('qwen'):reasons.append('Qwen template source')
        if not reasons:continue
        total+=len(data);assert total<=MAX_TOTAL,'Inspection source limit exceeded'
        assert len(selected)<128,'Too many interface sources'
        selected[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
            'reasons':reasons,'matched_lines':[n for n,line in enumerate(text.splitlines(),1)
                if any(re.search(pattern,line) for pattern in PATTERNS)]}
        payload[name]=data
    templates=[n for n in selected if n.startswith('swift/template/')]
    losses=[n for n in selected if 'loss' in n and n.startswith(('swift/plugin/','swift/trainers/'))]
    assert templates and losses,'Template or loss interfaces missing; no training preparation completed'
    # Stage only after all source/path/size/category checks have passed.
    out.mkdir(parents=True)
    for name,data in payload.items():
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    report={'status':'runtime_interfaces_collected','distribution':'ms-swift','version':distribution.version,
        'python':platform.python_version(),'platform':platform.platform(),'package_root':str(package),
        'file_count':len(selected),'total_source_bytes':total,'files':selected,
        'template_files':templates,'loss_files':losses,
        'model_calls':0,'training_started':False,'training_ready':False,
        'partial_label_loss_implemented':False,'tokenizer_loaded':False,
        'scope':'Installed package Python source inspection only; partial-label design and real token/mask preflight remain pending.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU inspection must hide GPUs'
    report=collect(importlib.metadata.distribution('ms-swift'),args.out)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    provenance={'code_commit':commit,'tool_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()}
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':report['status'],'version':report['version'],'source_files':report['file_count'],
        'source_bytes':report['total_source_bytes'],'training_ready':False}),flush=True)
    print('RUNTIME_INSPECTION_OK=true\nTRAINING_READY=false',flush=True)


if __name__=='__main__':main()
