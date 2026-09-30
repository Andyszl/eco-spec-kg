# 公式来源、角色复核与论文更新交付

本目录通过Git交付，服务器拉取后运行，无需手工上传ZIP。后续同类交付也使用Git；不要提交无关修改或临时缓存。

- `formula_review_20260929.zip`：24个文件，包含论文更新稿、来源/角色清单、独立训练候选包和回放校验脚本。
- `verify_formula_review_20260929.sh`：从自身目录读取ZIP；校验、复算并比较服务器与本地结果。
- ZIP SHA256：`e4dcdd48b55438dad74374cf222c77682fc7f79185810e45edcdaedac96cd1e4`。

ZIP内旧版说明中的“手工上传”步骤由本说明替代。为保留已验收内容与哈希，ZIP本体未修改。

## 在服务器运行

获取本目录对应的交付提交后，激活`ecospec-train`，执行：

```bash
bash /实际拉取目录/deliveries/formula_review_20260929/verify_formula_review_20260929.sh
```

脚本依赖之前已生成的：

```text
/home/hello/szl/eco-spec-results/paper_audit_20260929/session.env.sh
```

沿用其中的`DATA_PACKAGE`，使用固定候选代码提交`9bc751ff7281faf41e75193975159d429db76e71`。交付提交和候选算法提交用途不同；本次没有更改候选算法。

脚本复用或创建独立代码工作树`/home/hello/szl/eco-spec-kg-symbols-20260929`，在`/home/hello/szl/formula_review_20260929`解压数据。已有解压目录会先校验文件哈希，不重新覆盖。

结果输出到`/home/hello/szl/eco-spec-results/source_review_20260929_matrix`。若报告目录已存在，脚本停止并提示核查，不删除或覆盖已有结果。

预期修订器测试5项通过，693个训练单元中10个修订、683个保持；四组关系覆盖分别为607/799、602/794、640/799、662/794。最后显示“服务器与本地四格统计完全一致，复验通过”。

该包为助手局部技术复核的训练候选，未获领域专家签署。运行仅进行数据与覆盖复验，不训练、不调用模型服务、不覆盖冻结数据。
