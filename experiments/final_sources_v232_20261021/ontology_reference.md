# 冻结本体参考

只标注原文明示且落在本体范围内的信息。下表不是这两份新原文的预测或金标。

实体类型：assessment_indicator、assessment_subject、classification_rule、data_source、ecosystem_type、formula、frequency、instrument、method、model_variable、observation_variable、quality_rule、spatial_scope、temporal_scope、unit。

| 起点类型 | 关系 | 终点类型 | 原证据口径 |
|---|---|---|---|
| assessment_subject | has_indicator | assessment_indicator | 指标体系表或正文明确列出组成关系。 |
| assessment_indicator | calculated_by | formula | 指标/变量与公式编号在公式包中明确关联。 |
| model_variable | calculated_by | formula | 指标/变量与公式编号在公式包中明确关联。 |
| formula | has_input | model_variable | 变量符号真实出现在公式右侧并有坐标证据。 |
| formula | has_output | model_variable | 变量符号位于公式左侧并有坐标证据。 |
| model_variable | obtained_by | method | 同一来源单元存在获取/计算方法触发语义。 |
| observation_variable | obtained_by | method | 同一来源单元存在获取/计算方法触发语义。 |
| observation_variable | measured_with | instrument | 变量与仪器位于同一条款或表格逻辑行。 |
| model_variable | sourced_from | data_source | 来源触发词与变量在同一来源单元明确关联。 |
| observation_variable | sourced_from | data_source | 来源触发词与变量在同一来源单元明确关联。 |
| assessment_indicator | has_unit | unit | 单位与对象位于同一变量说明或表格逻辑行。 |
| model_variable | has_unit | unit | 单位与对象位于同一变量说明或表格逻辑行。 |
| observation_variable | has_unit | unit | 单位与对象位于同一变量说明或表格逻辑行。 |
| observation_variable | observed_during | temporal_scope | 变量与时段位于同一条款或表格逻辑行。 |
| observation_variable | observed_every | frequency | 变量与频率位于同一条款或表格逻辑行。 |
| assessment_indicator | applies_to_ecosystem | ecosystem_type | 适用对象必须由正文、表头或注释明确表达。 |
| formula | applies_to_ecosystem | ecosystem_type | 适用对象必须由正文、表头或注释明确表达。 |
| method | applies_to_ecosystem | ecosystem_type | 适用对象必须由正文、表头或注释明确表达。 |
| assessment_indicator | applies_to_space | spatial_scope | 空间条件必须由正文、表头或注释明确表达。 |
| formula | applies_to_space | spatial_scope | 空间条件必须由正文、表头或注释明确表达。 |
| method | applies_to_space | spatial_scope | 空间条件必须由正文、表头或注释明确表达。 |
| data_source | constrained_by | quality_rule | 约束对象与阈值、条件或动作必须在质量条款中共现。 |
| method | constrained_by | quality_rule | 约束对象与阈值、条件或动作必须在质量条款中共现。 |
| model_variable | constrained_by | quality_rule | 约束对象与阈值、条件或动作必须在质量条款中共现。 |
| observation_variable | constrained_by | quality_rule | 约束对象与阈值、条件或动作必须在质量条款中共现。 |
| assessment_indicator | classified_by | classification_rule | 指标与等级边界必须来自同一分级表或条款。 |
