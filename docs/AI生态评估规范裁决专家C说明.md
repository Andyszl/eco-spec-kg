# AI生态评估规范裁决专家C说明

## 1. 角色性质

“生态评估技术规范AI裁决专家C”是用于来源核验的AI模拟角色，不是自然人领域
专家。角色定义位于
`config/adjudication/ai_ecology_standard_expert_c_v1.json`，当前7条裁决记录位于
`config/adjudication/expert_c_ai_adjudication_v22.json`。

该角色可以生成 `ai_expert_adjudicated_gold`，不能生成或声明
`human_expert_gold`。`human_expert_gold` 是对审核者身份、资质和独立审核过程的
来源声明，不是通过提示词或角色扮演可以获得的质量等级。

## 2. 当前裁决结果

当前7条疑似金标准问题已经完成AI裁决，无待定项：

1. 新增森林、灌丛、草地3个 `ecosystem_type` 实体。
2. 删除仅来自文档标题的“全国” `spatial_scope` 实体。
3. 删除由过程短语名词化得到的“观测数据拟合” `method` 实体。
4. 保留叶面积指数到公式B.1、B.2、B.3的现有计算关系。
5. 保留“生态系统质量--has_indicator-->EQI”的规范关系，只登记
   “生态系统质量指数（EQI）”为别名，不重复增加金标准边。

## 3. 裁决写回

先把7条结构化裁决写入新的标注文件，原V2.1标注文件保持不变：

```powershell
python -m ecospec_kg.cli_v2 apply-adjudication-v2 `
  --source-units results/dataset_v2_20260724/gold/source_units_enriched.jsonl `
  --annotations results/dataset_v2_20260724/gold/annotations_all_units.jsonl `
  --decisions config/adjudication/expert_c_ai_adjudication_v22.json `
  --out results/dataset_v2_20260724/gold/annotations_all_units_v22.jsonl
```

命令同时生成：

- `annotations_all_units_v22.adjudication_manifest.json`：输入输出哈希、动作计数和实体关系增量。
- `annotations_all_units_v22.aliases.json`：仅登记别名，不改变严格金标准。

写回器会拒绝待定记录、未知来源单元、无效证据坐标、重复新增、删除不存在或仍被关系引用的实体，以及缺失的保留关系。

## 4. 冻结要求

冻结AI裁决数据包时必须同时传入角色和裁决来源文件：

```powershell
python -m ecospec_kg.cli_v2 prepare-experiment-v2 `
  --source-units <source_units.jsonl> `
  --annotations <adjudicated_annotations.jsonl> `
  --out data/frozen/v2.2 `
  --dataset-version v2.2 `
  --gold-nature ai_expert_adjudicated_gold `
  --review-provenance config/adjudication/expert_c_ai_adjudication_v22.json
```

冻结器会把来源记录复制到数据包并记录SHA-256。若把该AI来源文件与
`--gold-nature human_expert_gold` 组合使用，命令必须失败。

本地V2.2产物的验收结果为：882个来源单元，训练/开发/外部测试划分为
693/22/167；盲集答案字段为0；LoRA训练记录为693条；候选实体和关系召回上限
分别为0.903537和0.823529。

## 5. 论文可用表述

可表述为：

> 本研究构建了来源约束的AI双代理复核与第三代理裁决流程，所有争议项均回溯至
> HJ技术规范原文。所得数据记为AI裁决参考集，用于系统开发、预实验和消融分析。

不得表述为“由两名生态领域专家独立标注”“经真实专家裁决”或
“human expert gold”。在没有自然人领域专家时，论文应把这项限制写入有效性威胁，
正式性能结论不得使用“专家金标准”措辞。
