# 领域专家审核与数据冻结入口

本交付只包含生成与校验程序。专家工作包、审核结果和测试标注保存在服务器受限目录，不进入公开 Git。当前状态是**待审核**；已有“专家 C”是 AI 模拟角色，不能替代自然人签署。

## 审核范围

从已经在服务器复验的 rc2 训练候选取 693 条，从原冻结 V2.2 取开发集 22 条、测试集 167 条，合计 882 条。训练候选 rc2 的来源及标注哈希分别应为：

```
6cb286dd48a36d478f68eaf2ed7e1ec28cd8bbd31fdef9ae33359e0446a52162
97c942f32658595b804106fd172b96018be7ee388e812f443bde11ee8278a906
```

服务器运行 `bash tools/prepare_human_review_20261001.sh`。脚本核对两份数据包的清单及文件哈希，生成私有工作目录 `/home/hello/szl/eco-spec-results/human_review_20261001/packet`；目标存在时拒绝覆盖。

包内文件：

| 路径 | 用途 |
|---|---|
| `sources/all_units.jsonl` | 882 条只含来源的盲单元，供原文与证据核对 |
| `reference/ai_candidate_annotations.jsonl` | AI 候选，仅供人工核对，不能直接称专家金标 |
| `templates/expert_A.jsonl`、`expert_B.jsonl` | 两份相同候选起点，互不共享审核结果，均标记 `pending_human_review` |
| `manifest.json` | 单元数、父包版本和工作包文件哈希 |

应让两位真实生态领域专家分别审核全部 882 条，检查实体、关系方向、证据 span、公式角色和漏标；可纠正、添加、删除标注。审阅过程彼此独立，但因模板预填 AI 候选，论文必须如实称为“独立的人类专家复核 AI 候选”，不能写成“从零独立标注”。审核完的每条记录设 `annotator_id` 为实际审核者 ID，`review_status` 为 `human_reviewed`，并填写 `human_review.reviewer_id`、`human_review.reviewed_at`。模板中的 `PENDING_EXPERT_A/B` 不能用于冻结。

每行 JSON 对应一个 `unit_id`，对照同 ID 的 `sources/all_units.jsonl` 审核。原 PDF 对于公式版面、上下标和表格仍是必要证据；服务器没有完整原规范文件时，须私下向专家提供相应原件。不能仅依赖预填 AI 标签。修改实体时保留或新建本单元唯一 `entity_id`，关系的 `head_id/tail_id` 必须指向本单元实体且名称和类型一致；所有 `evidence_span_ids` 应来自该来源单元。每位专家只修改自己提交文件，保留原模板作核验对照。填写审核时间应是实际完成时间，不能批量标记未核对的单元。

两个独立提交齐全后先生成分歧清单，再进行裁决：

```bash
python tools/prepare_human_review_v2.py compare \
  --packet "$PACKET" --expert-a "$REVIEW_A" --expert-b "$REVIEW_B" \
  --out "$COMPARISON_OUT"
```

输出 `disagreements.jsonl` 和 `agreement.json`，同时检查审核状态、全量单元、实体/关系类型、证据及关系端点 ID。分歧按单元列出，供裁决者逐项回看原文。若无分歧，`disagreements.jsonl` 和后续 `adjudication_log.jsonl` 均可为空文件，但最终882条仍须由指定裁决者确认。

对两位专家意见不同的单元，裁决者逐条写一条 `adjudication_log.jsonl`，包含 `unit_id`、`adjudicator_id`、具体 `reason` 和 `signed_at`；全量 `adjudicated_annotations.jsonl` 每条设 `review_status=human_adjudicated`，并填写裁决者 ID、时间、最终实体及关系。若两位专家完全一致，最终标注仍需经指定裁决者确认，且不能无理由改变一致的实体或关系。

另需 `review_provenance.json`：`schema_version=ecospec-review-provenance-v2.0`、`gold_nature=human_expert_gold`、`claims_human_expert_review=true`；`reviewers` 中有两位不同的 `human_domain_expert`，各含 `reviewer_id`、`human_expert=true`、`qualification_summary`、`identity_verification_reference`、`signed_at`；`adjudication` 中明确其中一位为 `adjudicator_id` 且 `unresolved_count=0`。这些字段必须由实际审核过程产生，不得从 AI 候选或本模板自动填充。

专家文件齐全后，在独立目录运行：

```bash
python tools/prepare_human_review_v2.py validate \
  --packet "$PACKET" --expert-a "$REVIEW_A" --expert-b "$REVIEW_B" \
  --adjudicated "$ADJUDICATED" --provenance "$PROVENANCE" \
  --decision-log "$DECISION_LOG" --out "$VALIDATION_OUT"

python tools/prepare_human_review_v2.py freeze \
  --packet "$PACKET" --expert-a "$REVIEW_A" --expert-b "$REVIEW_B" \
  --adjudicated "$ADJUDICATED" --provenance "$PROVENANCE" \
  --decision-log "$DECISION_LOG" --out "$NEW_FROZEN_PACKAGE"
```

校验器拒绝未审核模板、同一人占两个审核席位、单元缺失或乱序、证据失效、关系端点缺失、未裁决分歧和缺少真实专家来源记录。冻结器会复跑全部校验，记录各专家文件哈希，并检查原划分未变；输出目录存在则停止。

## 正式实验边界

现有 167 条测试单元及其预标注指标已在早期工程调试中使用。用户确认目前没有全新的独立外部来源，所以它们不能作为首次盲测的正式外部测试结果。两位专家尚未提供审核记录前，也不能把 rc2 或上述模板冻结为 `human_expert_gold`。

完成专家审核后，先在冻结开发集上确定模型、提示词、阈值、最大输出长度及候选生成版本；记录 Qwen 模型/LoRA 权重、Git 提交、训练数据哈希、随机种子和推理请求种子。三个训练随机种子可取 42/43/44，固定数据划分与数据随机种子。当前 V2 `use_schema/use_layout/use_evidence=false` 会被代码拒绝，故不能把改配置标签当作有效消融；消融要先实现真实行为差异并用测试证实。已有开发/测试集可用于标明为预实验的工程验证。最终外部测试须等获得此前未参与开发的独立来源、完成专家裁决，并冻结方法后再执行一次。

本地已用现有冻结 V2.2 开发集执行规则后端流程预检：22 条抽取成功、22 条合法性检查通过，AI 裁决参考标注上的实体 F1 为 0.818182、关系 F1 为 0.714286。此处数字只证明命令链衔接，不能写作 rc2 专家金标或正式模型实验结果；没有运行 LoRA 训练或 Qwen 推理。
