"""Train-derived clause rejection must preserve proper method names and input bytes."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from ecospec_kg.method_clause_guard_v218 import rejection_reason,filter_prediction


@pytest.mark.parametrize('name',['）的计算方法','如下公式计算','如下公式进行计算',
    '地上生物量估算可采用植被指数法','回归系数常用最小二乘法','可根据区域特点选用模型'])
def test_reviewed_training_clause_is_rejected(name):
    assert rejection_reason(name)


@pytest.mark.parametrize('name',['每木检尺','支持向量机','实地测量','归一化处理','异速生长方程计算','侵蚀-生产力评价模型',
    '最小二乘法','机器学习方法','修正土壤流失方程（RUSLE）','基于遥感影像分类的方法'])
def test_no_method_whitelist_or_blanket_calculation_suffix_rule(name):
    assert rejection_reason(name) is None


def test_only_invalid_method_and_incident_edges_removed_without_mutation():
    p={'entities':[{'entity_id':'m','entity_type':'method','name':'如下公式计算'},
        {'entity_id':'d','entity_type':'data_source','name':'如下公式计算'},
        {'entity_id':'x','entity_type':'model_variable','name':'x'}],
       'relations':[{'relation_id':'bad','head_id':'x','tail_id':'m'},{'relation_id':'good','head_id':'x','tail_id':'d'}],
       'no_relation_reason':'','config_hash':'original','experiment_id':'fixture'}
    old=deepcopy(p);filtered,proof=filter_prediction(p)
    assert p==old and [r['relation_id'] for r in filtered['relations']]==['good']
    assert [e['entity_id'] for e in filtered['entities']]==['d','x']
    assert proof['removed_entity_ids']==['m'] and proof['removed_relation_ids']==['bad']
    assert filter_prediction(filtered)[0]==filtered
