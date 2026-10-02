# 人工确认数据与模型实验准备（2026-10-02）

项目用户于2026-10-02明确确认：现有A/B结果与分歧裁决使用AI辅助，人工已经确认。当前数据口径为“AI辅助复核、人工确认并裁决”。完整私有冻结包为 `v2.2-human-confirmed-20261002`，package_id=`3d80e7fb28e3a308`；882单元，原划分693 train / 22 dev / 167 test保持。272个分歧单元均有裁决。后续以该版本为准。

本Git交付包含693条训练来源及最终标注、22条开发来源及最终标注、Schema、逐文件哈希和复验期望值。审核者身份及审核备注已从公开交付中去除，实体、关系、证据ID保持原结果；完整专家记录与test标注保留在本地私有包中。交付文件不是再次人工审核的起点。

## 已完成的本地验证

- A、B及最终标注均882条，顺序、状态、关系端点、本体组合、证据ID检查通过；272条分歧与裁决日志逐一对应。
- 完整包文件哈希核验通过；公开train/dev片段保持最终实体与关系内容。
- 导出693条LoRA训练记录；规则开发集22条抽取和预测合法性检查全部通过。
- 规则开发集实体F1=0.711111、关系F1=0.571429、关系宏F1=0.654449。这是本地规则基线，未运行Qwen/LoRA模型训练。

## 本轮发现的具体候选问题

固定已确认train标注和v2.6候选生成器：实体1018/1389=0.732901、关系603/840=0.717857，分别缺371个实体和237条关系。主要缺失实体为观测变量102、方法94、单位50、质量规则33；主要缺失关系为obtained_by 67、has_unit 51、constrained_by 33、applies_to_space 27、measured_with 26。

公式has_input（77/77）和has_output（32/32）在train均已覆盖。新数据增加并规范了方法、表格字段和单位标注，当前候选生成器尚未匹配全部口径；不能用此前旧标注的0.860553覆盖值代替本轮结果。闭集选择器无法恢复未生成的候选。

下一轮候选改进应只从train来源及标注分析通用模式，先补表格字段/单位候选和方法、仪器名称归一化，再处理质量约束和空间关系；以新增/丢失覆盖和候选数量同时验收。开发集用于方案选择，现有167条test按内部留出评估记录。当前v2.6可作为固定对照基线，正式改进模型须记录其候选上限。

## 服务器拉取与复验

在已有 `ecospec-train` 环境执行。Git获取后固定为本次取到的提交，不改原工作区；目标目录存在时停止。

```bash
conda activate ecospec-train
(
  set -euo pipefail
  REPO=/home/hello/szl/eco-spec-kg
  WORK=/home/hello/szl/eco-spec-kg-human-confirmed-20261002
  OUT=/home/hello/szl/eco-spec-results/human_confirmed_20261002
  git -C "$REPO" fetch https://github.com/Andyszl/eco-spec-kg.git main
  REV=738f1d66bad904cb270ed389fbad83b0478d3694
  test ! -e "$WORK"
  test ! -e "$OUT"
  git -C "$REPO" worktree add --detach "$WORK" "$REV"
  cd "$WORK"
  export PYTHONPATH="$PWD/src"
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONIOENCODING=utf-8
  python -m pytest tests/test_human_review_gate.py tests/test_confirmed_delivery.py \
    -o addopts='' -p no:cacheprovider -q
  python tools/verify_human_confirmed_20261002.py \
    --delivery deliveries/human_confirmed_20261002 --out "$OUT"
)
```

复验器核对交付字节、train/dev单元与端点/证据，重新生成训练数据、规则预测和开发集指标，最后将实际统计与expected.json逐项比较；结果在 `verification.json`。它不需要服务器原专家提交文件，也不运行GPU模型训练。

## v2.6对照LoRA训练命令

完成上述服务器复验后，使用已生成的training/qwen35_train.jsonl。先确认GPU资源和本地模型路径；命令不自动终止已有服务。20条烟雾训练成功后，可保持同一候选版本和数据运行693条seed42对照训练，之后按完全相同设置顺序运行seed43/44，data_seed始终42。

```bash
conda activate ecospec-train
(
  set -euo pipefail
  cd /home/hello/szl/eco-spec-kg-human-confirmed-20261002
  export PYTHONPATH="$PWD/src"
  export PYTHONDONTWRITEBYTECODE=1
  export QWEN_LLM=/home/hello/szl/models/Qwen3.5-9B
  test -f "$QWEN_LLM/config.json"
  swift sft --help >/dev/null
  nvidia-smi
  TRAIN_OUT=/home/hello/szl/eco-spec-runs/lora/human_confirmed_20261002/seed42_smoke_v26
  test ! -e "$TRAIN_OUT"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m ecospec_kg.cli train \
    --prepared /home/hello/szl/eco-spec-results/human_confirmed_20261002/training/qwen35_train.jsonl \
    --out "$TRAIN_OUT" --model "$QWEN_LLM" --trainer swift \
    --precision bf16 --max-length 8192 --train-batch-size 1 --eval-batch-size 1 \
    --gradient-accumulation-steps 16 --lora-rank 8 --lora-alpha 16 \
    --lora-dropout 0.05 --learning-rate 2e-4 --epochs 1 \
    --seed 42 --data-seed 42 --smoke-limit 20 --run
)
```

完整对照训练使用同一命令，去掉 `--smoke-limit 20`，将epochs设5，输出换为全新seed42_full_v26目录。不得用烟雾训练结果代替全量模型结果，也不得把改配置标签当作有效消融。本轮仍有候选覆盖缺口，建议优先改进候选并重新导出训练数据，再运行改进方案的三seed实验。
