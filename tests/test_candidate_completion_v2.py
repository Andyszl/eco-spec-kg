"""Source-derived field and method candidates; labels are never inference inputs."""
import pytest
import sys
from pathlib import Path

from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_candidate_completion_20261002 import compare_train


def source(kind, text="", cells=None, raw_cells=None):
    return {
        "unit_id": "candidate-fixture", "unit_type": kind,
        "table_title": "表 A.8 湿地生态系统样方调查表",
        "cells": cells or {}, "raw_cells": raw_cells if raw_cells is not None else cells or {},
        "clause_text": text,
        "provenance": {
            "standard_code": "HJ 1169-2021", "document_title": "湿地生态系统野外观测",
            "section": "9.8", "heading_chain": ["9.8 叶面积指数"], "pages": [8],
            "evidence_spans": [{"span_id": "row", "page": 8,
                                "bbox": [0, 0, 100, 20], "text": text or str(cells)}],
        },
    }


def predict(unit):
    result = RuleCandidateExtractorV2().predict_unit(unit)
    entities = {(e["name"], e["entity_type"]) for e in result["entities"]}
    edges = {(r["head_name"], r["relation_type"], r["tail_name"]) for r in result["relations"]}
    return entities, edges


@pytest.mark.parametrize("field,name,unit", [
    ("径流量/（m³/s）", "径流量", "m³/s"),
    ("土壤容重/（g/cm3）", "土壤容重", "g/cm3"),
    ("植被覆盖度/%", "植被覆盖度", "%"),
    ("土壤机械组成a/%", "土壤机械组成", "%"),
    ("坡度/（°）", "坡度", "（°）"),
    ("水温/℃", "水温", "℃"),
])
def test_form_field_has_its_own_unit(field, name, unit):
    entities, edges = predict(source("table_record", cells={"样方编号 / 样方定位": field}))
    assert (name, "observation_variable") in entities
    assert (unit, "unit") in entities
    assert (name, "has_unit", unit) in edges


def test_form_multiple_fields_do_not_cross_link_units_or_inherit_headers():
    cells = {"样方编号 / 样方定位": "生物量/g", "地理位置 / 编号": "平均高/m",
             "样方编号 / 样方定位_2": "物种名称： 多度："}
    raw = {**cells, "样方编号 / 样方定位_2": None}
    entities, edges = predict(source("table_record", cells=cells, raw_cells=raw))
    assert {("生物量", "has_unit", "g"), ("平均高", "has_unit", "m")} <= edges
    assert ("生物量", "has_unit", "m") not in edges
    assert ("平均高", "has_unit", "g") not in edges
    assert ("物种名称", "observation_variable") not in entities
    assert ("样方编号", "observation_variable") not in entities


def test_form_plain_label_and_colon_unit_and_combined_fields():
    cells = {"行政区 / 编码": "水分条件", "行政区 / 编码_2": "年平均降雨量： mm",
             "地理位置 / 坐标": "风速/（m/s）、风向", "样方定位": "叶面积指数"}
    entities, edges = predict(source("table_record", cells=cells))
    assert {(n, "observation_variable") for n in ("水分条件", "年平均降雨量", "风速", "风向", "叶面积指数")} <= entities
    assert ("年平均降雨量", "has_unit", "mm") in edges
    assert ("风速", "has_unit", "m/s") in edges
    assert not any(edge[0] == "风向" and edge[1] == "has_unit" for edge in edges)


def test_choice_values_and_non_form_tables_are_not_field_labels():
    cells = {"样地所在行政区 / 行政编码": "土壤质地",
             "样地所在行政区 / 行政编码_2": "砾石质□、沙土□、壤土□、黏土□"}
    entities, _ = predict(source("table_record", cells=cells))
    assert ("土壤质地", "observation_variable") in entities
    assert not any(e[0] in {"砾石质", "沙土", "壤土", "黏土"} for e in entities)
    unit = source("table_record", cells={"说明": "误差/m"})
    unit["table_title"] = "精度分级标准"
    assert ("误差", "observation_variable") not in predict(unit)[0]


@pytest.mark.parametrize("text,method", [
    ("采用吸管法测定，具体采样依据相关要求执行。", "吸管法"),
    ("用烘干法测定区域调查点的土壤含水量。", "烘干法"),
    ("采用重铬酸钾氧化-分光光度法测量。", "重铬酸钾氧化-分光光度法"),
    ("依据异速生长方程计算，获取地面观测数据。", "异速生长方程"),
    ("采用摄影成像技术获取植被冠层间隙率。", "摄影成像技术"),
    ("运用修正土壤流失方程（RUSLE）计算土壤保持量。", "修正土壤流失方程（RUSLE）"),
    ("观测内容：每木检尺；观测指标：胸径；观测频度：一年一次", "每木检尺"),
    ("采用目测法和照相法，记录草本植物的覆盖度。", "照相法"),
    ("使用机器学习模型进行训练，处理观测数据。", "机器学习模型"),
    ("可使用数字高程模型（DEM）提取或通过 GPS 测量。", "GPS测量"),
])
def test_method_boundaries_and_local_aliases(text, method):
    entities, edges = predict(source("procedure_clause", text=text))
    assert (method, "method") in entities
    assert any(e[1:] == ("obtained_by", method) for e in edges)


def test_processing_enumeration_is_kept_in_quality_clause():
    text = "对获取的遥感数据进行辐射校正、几何精校正、大气校正、图像配准、拼接与裁剪等一系列处理。"
    entities, _ = predict(source("quality_clause", text=text))
    assert {(n, "method") for n in ("辐射校正", "几何精校正", "大气校正", "图像配准", "拼接", "裁剪")} <= entities


def test_source_visible_instrument_and_method_edge():
    text = "采用冠层分析仪和叶面积仪测定叶面积指数；使用雨量器测量降水。"
    entities, edges = predict(source("procedure_clause", text=text))
    assert {(n, "instrument") for n in ("冠层分析仪", "叶面积仪", "雨量器")} <= entities
    assert ("叶面积指数", "measured_with", "冠层分析仪") in edges


def test_reference_standard_is_not_a_method_name():
    text = "具体测定步骤依据 HJ 962 的相关要求执行。"
    entities, _ = predict(source("procedure_clause", text=text))
    assert not any("HJ 962" in name for name, typ in entities if typ == "method")


def test_dem_source_is_not_added_as_a_bare_method():
    entities, _ = predict(source("procedure_clause", text="可使用数字高程模型（DEM）提取或通过 GPS 测量。"))
    assert ("数字高程模型", "method") not in entities
    assert ("数字高程模型（DEM）", "method") not in entities
    assert ("数字高程模型（DEM）", "data_source") in entities


def test_standard_scope_quality_sentence_is_not_a_method_list():
    entities, _ = predict(source("quality_clause", text="本标准规定了数据质量评价方法、质量控制与集成要求。"))
    assert not any(typ == "method" for _, typ in entities)


def test_unit_exponent_typography_is_local_to_the_field():
    entities, edges = predict(source("table_record", cells={"样方定位": "径流量/（m3/s）"}))
    assert {("径流量", "has_unit", u) for u in ("m3/s", "m³/s")} <= edges
    assert ("m²/s", "unit") not in entities


def test_percentage_definition_links_only_the_named_observation():
    cells = {"观测指标": "林下覆盖度", "指标定义": "垂直投影面积占统计区总面积的百分比"}
    entities, edges = predict(source("table_record", cells=cells))
    assert ("百分比", "unit") in entities
    assert ("林下覆盖度", "has_unit", "百分比") in edges


def test_candidate_audit_rejects_test_labels():
    with pytest.raises(ValueError, match="train annotations only"):
        compare_train([], [{"unit_id": "u", "split": "test"}], None, None)


def test_candidate_audit_exposes_a_loss_even_if_total_coverage_is_equal():
    class Fixed:
        def __init__(self, name):
            self.name = name
        def predict_unit(self, unit):
            return {"entities": [{"name": self.name, "entity_type": "unit"}], "relations": []}

    labels = [{"unit_id": "u", "split": "train", "relations": [],
               "entities": [{"name": n, "entity_type": "unit"} for n in ("m", "cm")]}]
    summary, changes, gaps = compare_train([{"unit_id": "u"}], labels, Fixed("m"), Fixed("cm"))
    assert summary["baseline"]["covered_entity_count"] == summary["current"]["covered_entity_count"] == 1
    assert summary["lost_coverage_by_type"] == {"unit": 1}
    assert changes[0]["entities"]["lost_coverage"] == [("m", "unit")]
    assert gaps[0]["gold"]["name"] == "m"
