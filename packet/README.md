# 独立验收原文复核包

本包已核对官方 PDF 字节和关键页，尚无人工金标，也没有模型预测。现在先做原文盲标，不需要训练或 GPU。

1. 打开 review/01_source_eligibility.md，确认两份新材料是否曾参与本项目。自动检查只覆盖已记录的来源，不能替代项目历史确认。
2. 打开 sources/ 中两份原文，对照 review_scope.json 的固定主选段、上下文边界和页码，填写 review/02_HJ1339-2023_review.md 与 review/03_HJ1300-2023_review.md。可用记事本编辑，Markdown 表格可自行增加行。page_images/ 是原 PDF 的关键页预览。
3. 在不查看新材料模型预测的条件下，按 ontology_reference.md 完成实体、关系、主/辅助公式、输入/输出、证据与歧义审核。所有字段留空是因为尚未有人审核，不是负例。
4. 填写 review/04_adjudication_and_coverage.md。完成后仅提交 review/ 下的人工记录；原 PDF、source_manifest.json、review_scope.json 和本体参考均保留原样。
5. 取回审核记录，核对争议和完整性，冻结金标与实际计分范围后，才生成最终评估入口。三组复用当前冻结模型/提示/候选规则及结构、证据模块；不按新材料结果修改规则或重训。

这是两份规范的两段局部案例，尚不能称为正式最终验收集，不能支持广泛的来源泛化结论。既有开发集门槛未通过的结论保持，不扩三种子。

BLIND_PACKET_OK 只表示材料完整复制，INFERENCE_READY=false 是当前的正确状态；不需要为了把它改成 true 再跑旧预检。
