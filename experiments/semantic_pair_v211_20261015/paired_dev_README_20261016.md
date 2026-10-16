# 新监督配对的完整开发集评估

适用训练目录：`/home/hello/szl/eco-spec-runs/lora/semantic_seed42_pair_20261015-182817-1829977`。
两臂实际200步，均按共同70条内部验证最低loss选checkpoint-80。接收原始清单SHA256：`ae47c5462ff58ec2117bcff8fc27454e916a0389766924ebadc5735c31744358`。

执行入口：`bash tools/run_semantic_pair_dev_20261016.sh`。入口仅推理，不重训；启动前核验清单、训练输入/源码/冻结数据、两份选定adapter及trainer state哈希、分词配置、Swift4.4.2、GPU0及8018端口。不会停止现有GPU任务。

基座、控制LoRA、修复LoRA各评估完整dev22，总66请求，共用v2.11候选、v2.6提示和固定唯一ID约束解码。每组保存原始预测、响应证明、证据/模式校验、结构前指标、同一冻结结构模块的补边证明及结构后指标。严格门槛沿用原先约定，不根据本次结果调整。错误/无效响应不产出通过结论。推理服务退出仅清理本次创建的子进程组。

结果包自动上传`codex/semantic-seed42-pair-dev-*`，回传RESULTS_UPLOAD_OK、EVALUATION_EXIT_CODE、EVALUATION_DIRECTORY。不传权重，训练数据及历史结果不变。

本入口本地测试使用合成adapter和模拟HTTP；不能冒充服务器推理结果。旧v2.9评估入口及其源码锁保留；旧测试仅在其固定8fe81ec版本复现。当前完整dev多次参与开发，不是未参与方案选择的最终测试；未达到严格门槛不得扩三种子。
