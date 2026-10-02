# 候选补齐 v2.7（2026-10-02）

以人工确认冻结数据 `v2.2-human-confirmed-20261002` 为固定口径，仅从693条训练单元的漏项归纳规则。冻结来源、实体关系标注、划分、私人审核记录均未修改。本轮公开交付复用相邻 `human_confirmed_20261002` 中的匿名化train/dev数据，其manifest哈希写入本目录manifest。test不参与本轮开发或复评。

## 完成内容

- 观测调查表：识别实际行中“字段/单位”和“字段：单位”、多个字段、无单位字段；使用raw_cells，排除跨行继承值、表头和勾选选项。每条has_unit只连接同一字段的单位。
- 方法：增加采用/运用等触发词之后的独立方法名、法/方程/模型/技术/抽样边界、原文括号简称及遥感预处理步骤枚举。新增的纯DEM数据来源不归为方法。保留原候选，供闭集选择器判别。
- 仪器：补充原文可见的观测仪器词汇，包含冠层分析仪、叶面积仪、雨量器等，并生成对应measured_with候选。
- 单位：保持原文单位；仅在明确单位字段内增加m3/m³等指数排版变体；支持定义中的“百分比”。公式符号、局部别名和求和绑定索引规则未修改。
- 候选版本升级为structure-aware-rule-v2.7，规则、零样本、LoRA配置同步。旧v2.6交付及历史结果保留，旧交付复验应使用固定提交738f1d66bad904cb270ed389fbad83b0478d3694。

## 同一训练金标下的本地验证

| 指标 | v2.6 | v2.7 |
|---|---:|---:|
| 实体候选覆盖 | 1018/1389（73.29%） | 1203/1389（86.61%） |
| 关系候选覆盖 | 603/840（71.79%） | 704/840（83.81%） |
| 实体候选总数 | 1921 | 2220 |
| 关系候选总数 | 1092 | 1296 |
| 未匹配金标的实体候选 | 903 | 1017 |
| 未匹配金标的关系候选 | 489 | 592 |

新增覆盖185个实体：观测变量62、单位43、方法56、仪器24；新增覆盖101条关系：has_unit44、obtained_by35、measured_with22。175个训练单元候选变化，原已覆盖实体/关系丢失均为0。模型变量97/97、has_input77/77和has_output32/32保持；仪器54/54均覆盖。

这些是候选召回上限，不是模型F1。规则直接输出全部候选的22条dev复测：实体F1由0.711111降至0.680851，关系F1由0.571429降至0.564706，关系宏F1由0.654449降至0.651671；新增正确项均为0，新增输出为误报。train/dev全部715条预测的端点、本体、证据ID和复现字段校验通过。正式Qwen/LoRA实验须验证候选选择是否能够消化新增候选，本轮未进行模型训练或推理。

剩余缺186个实体、136条关系；逐条清单由复验生成candidate_gaps_train.jsonl。主要剩余实体包括方法38、正文观测变量40、质量规则33、空间范围19、数据来源24；本轮未用金标文本直接填充推理候选。后续优先处理正文获取语义、质量约束与空间关联，按同一冻结口径检验，不能仅提高候选数量。

## 服务器拉取与复验

使用本次交付消息给出的完整Git提交SHA作为REV，固定检出新工作树。原服务器目录及旧报告保留。无需上传压缩包；复验工具只依赖已在Git中的train/dev文件和历史提交。

```bash
conda activate ecospec-train
(
  set -euo pipefail
  REPO=/home/hello/szl/eco-spec-kg
  WORK=/home/hello/szl/eco-spec-kg-candidates-v27-20261002
  OUT=/home/hello/szl/eco-spec-results/candidates_v27_20261002
  git -C "$REPO" fetch https://github.com/Andyszl/eco-spec-kg.git main
  # 将REV设为本次交付消息中的完整提交SHA。
  : "${REV:?请先设置本次交付提交SHA}"
  git -C "$REPO" cat-file -e "$REV^{commit}"
  test ! -e "$WORK"
  test ! -e "$OUT"
  git -C "$REPO" worktree add --detach "$WORK" "$REV"
  cd "$WORK"
  export PYTHONPATH="$PWD/src"
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONIOENCODING=utf-8
  git log -1 --oneline
  python -m pytest tests -o addopts='' -p no:cacheprovider -q -rs
  python tools/audit_candidate_completion_20261002.py --out "$OUT"
)
```

复验的verification.json只有实际统计和expected.json完全相等、没有已覆盖项丢失时才passed=true。它还校验交付和数据字节哈希、重建693条LoRA训练记录并验证693条train及22条dev规则预测。服务器尚未执行时，不能写成服务器复验已通过。

主要输出：comparison.json（总览及各类型覆盖）、train_candidate_changes.jsonl（逐单元增删）、candidate_gaps_train.jsonl（剩余缺项）、training/qwen35_train.jsonl及manifest、train_rule/validation、dev_rule/evaluation/metrics.json、verification.json。

## 下一步模型实验使用位置

服务器复验通过后，新版LoRA准备数据为`$OUT/training/qwen35_train.jsonl`，候选版本应始终为v2.7。先运行原交付中相同的20条烟雾训练参数，将prepared改为此文件、输出改为全新seed42_smoke_v27；通过后运行693条全量训练（seed42/43/44，data_seed42），并在同一候选版本的dev来源上推理、校验、评估。v2.6对照必须保留其旧工作树/准备数据，不能拿旧adapter配新版候选直接作为严格对照。
