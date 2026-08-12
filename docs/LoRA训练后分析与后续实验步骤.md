# LoRA 训练后分析与后续实验步骤

本文档从 Qwen3.5-9B LoRA `seed=42` 的正式开发集评测结果继续。当前最优
检查点为 `checkpoint-117`。在完成错误分析前，不重训模型，不读取测试集金标准。

## 1. 修正并完成结果归档

ms-swift 生成的 `logging.jsonl` 位于 `v0-*` 运行目录，但项目的
`run_manifest.json` 位于其父目录。按以下方式补拷训练清单：

```bash
export ECOSPEC_RUNS=/home/hello/szl/eco-spec-runs
export TRAIN_ROOT="$ECOSPEC_RUNS/lora/qwen35-9b/seed_42_full_v21_clean1"
export TRAIN_RUN="$TRAIN_ROOT/v0-20260807-143629"
export RESULT_ROOT=/home/hello/szl/eco-spec-results/lora_seed42

test -f "$TRAIN_ROOT/run_manifest.json" && echo "training manifest OK"
cp "$TRAIN_ROOT/run_manifest.json" "$RESULT_ROOT/training_manifest.json"
```

若第一条测试没有输出，定位文件后再复制：

```bash
find "$TRAIN_ROOT" -maxdepth 2 -name run_manifest.json -print
```

## 2. 更新分析代码

```bash
conda activate ecospec-train
cd /home/hello/szl/eco-spec-kg
git pull --ff-only origin main
python -m pip install -e ".[analysis]"
python -m pip install "pdfplumber==0.11.9" "Pillow>=10,<12"
python -m pip check

python -m ecospec_kg.cli_v2 --help | grep -E \
  'analyze-errors-v2|analyze-training-v2'
```

必须同时显示两个命令。

## 3. 生成 LoRA 开发集错误报告

```bash
export DATA_PACKAGE=/home/hello/szl/eco-spec-data/frozen/v2.1
export DEV_RUN="$ECOSPEC_RUNS/v2.1/qwen35_lora/dev_seed42_best_epoch3"
export ANALYSIS_ROOT="$RESULT_ROOT/analysis"

python -m ecospec_kg.cli_v2 analyze-errors-v2 \
  --units "$DATA_PACKAGE/blind/dev_units.jsonl" \
  --gold "$DATA_PACKAGE/gold/dev_annotations.jsonl" \
  --predictions "$DEV_RUN/validation/validated_predictions.jsonl" \
  --out "$ANALYSIS_ROOT/errors"

python -m json.tool "$ANALYSIS_ROOT/errors/error_summary.json"
ls -lh "$ANALYSIS_ROOT/errors/error_details.xlsx"
```

应生成：

```text
error_summary.json
error_details.jsonl
error_details.xlsx
```

当前 LoRA 结果预期为 63 个错误：实体 FN 16、实体 FP 12、关系 FN 10、关系
FP 25。`constrained_by` 的合计错误应为 12，其中 11 个 FP、1 个 FN。

Excel 的 `review_reason` 和 `correction_action` 故意留空，供人工复核填写。

## 4. 生成训练集分布报告

```bash
python -m ecospec_kg.cli_v2 analyze-training-v2 \
  --units "$DATA_PACKAGE/blind/train_units.jsonl" \
  --annotations "$DATA_PACKAGE/gold/train_annotations.jsonl" \
  --out "$ANALYSIS_ROOT/training"

python -m json.tool "$ANALYSIS_ROOT/training/training_distribution.json"
ls -lh "$ANALYSIS_ROOT/training/training_distribution.xlsx"
```

V2.1 当前基线应接近：

```text
unit_count: 693
gold_entity_count: 1244
covered_entity_count: 1122
entity_recall_upper_bound: 0.901929
gold_relation_count: 799
covered_relation_count: 651
relation_recall_upper_bound: 0.814768
duplicate_source_text_group_count: 39
duplicate_source_text_record_count: 109
```

重复来源单元不应直接删除。先判断它们是合法的表格行/跨标准重复，还是数据构造
过程产生的重复记录。

## 5. 人工复核错误报告

优先处理以下内容：

1. 筛选 `relation_type=constrained_by`，逐条判断 11 个 FP 和 1 个 FN。
2. 筛选 `error_type=entity_false_negative`，检查漏掉的 16 个实体。
3. 筛选 `relation_type=calculated_by`、`has_indicator`、`has_input`，检查公式路径。
4. 按 `standard_code` 和 `unit_type` 统计错误是否集中。
5. 在 `review_reason` 中只填写可由来源文本证明的原因。
6. 在 `correction_action` 中区分 Schema、候选生成、训练标注和模型选择错误。

完成标准：63个错误全部经过人工复核，不留空白裁决栏。

## 6. 决定是否构建 V2.2

只有人工复核证明问题来自训练数据或候选生成器时才构建 V2.2。不得把开发集的
答案或错误实例直接复制到训练集。允许的调整包括：

- 从训练集内部补充 `constrained_by` 困难负例；
- 修正训练集中可以由来源证据确认的错误标注；
- 补齐公式输入、输出、单位和来源关系；
- 修复候选生成器无法覆盖的实体或关系模式；
- 审核重复来源单元并记录保留或删除理由。

调整后重新冻结数据包，并验证 manifest 中所有文件哈希。

## 7. 后续实验顺序

固定 V2.2 后按以下顺序执行：

1. 实现并运行正式 few-shot，示例只从 train 检索。
2. 先重训 LoRA `seed=42`。
3. 仅当实体 F1、关系 F1、Macro 关系 F1和路径指标综合改善时，运行
   `seed=43`、`seed=44`。
4. 三个 seed 顺序运行，固定 `data_seed=42`，只改变模型 seed。
5. 再运行 GraphRAG 及单因素消融。
6. 所有方案冻结后，只运行一次测试集盲测。

当前金标准为 `ai_expert_pre_gold`，开发集结果只能作为预实验。论文正式性能结论
仍需基于专家复核后的 `human_expert_gold`。
