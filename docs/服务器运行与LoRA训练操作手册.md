# EcoSpec-KG 服务器运行与 LoRA 训练操作手册

本文档用于指导将 EcoSpec-KG 项目上传到服务器后，从原始 PDF 语料开始，完成语料解析、候选三元组抽取、人工审核、数据划分、LoRA 训练、实验评估和 Gradio 演示运行。

## 1. 总体流程

完整流程如下：

```text
上传代码
-> 创建 Python 3.11 环境
-> 安装项目依赖
-> 上传原始 PDF
-> 解析 PDF 生成 chunks
-> 抽取候选三元组
-> 生成专家审核表
-> 人工审核三元组
-> 生成 reviewed_relations.jsonl
-> 按知识路径划分 train/validation/test/external_test
-> LoRA 烟雾训练
-> 三随机种子正式训练
-> 运行预测、评估、消融和报告
-> 启动 Gradio 演示
```

注意：LoRA 训练不能直接使用未审核的候选三元组，必须使用人工审核通过的数据。

## 2. 目录约定

推荐服务器目录如下：

```text
/home/ecospec/project/eco-spec-kg      # 项目代码目录，推荐放在 /home
/home/ecospec/venvs/ecospec            # Python 虚拟环境或 conda 环境
/home/ecospec/cache/pip                # pip 缓存
/work/ecospec/corpus                   # 原始 PDF 语料目录，推荐放在 /work
/work/ecospec/hf-cache                 # Hugging Face 模型缓存
/work/ecospec/tmp                      # pip、模型下载和训练临时目录
```

不建议把项目、虚拟环境、模型缓存或 PDF 语料放在 `/root`、`/tmp` 或根分区 `/` 下。当前服务器根分区约 70G，容易被 PyTorch、CUDA 包、Docker 缓存和 pip 临时文件占满；`/home` 和 `/work` 空间更充足。

如果你的服务器目录不同，把后续命令中的路径替换成实际路径即可。

## 3. 本地上传代码

如果 GitHub 可以访问，优先使用 Git：

```bash
mkdir -p /home/ecospec/project
cd /home/ecospec/project
git clone https://github.com/Andyszl/eco-spec-kg.git
cd eco-spec-kg
```

如果服务器不能访问 GitHub，可以从本地上传压缩包或 Git bundle。

本地 PowerShell 打包：

```powershell
Set-Location 'D:\else\lky\生态论文重写版本\eco-spec-kg'
git bundle create eco-spec-kg-main.bundle main
scp .\eco-spec-kg-main.bundle root@服务器IP:/home/ecospec/project/
```

服务器解包：

```bash
cd /home/ecospec/project
git clone eco-spec-kg-main.bundle eco-spec-kg
cd eco-spec-kg
```

### 3.1 服务器更新代码

服务器已有仓库时，更新代码：

```bash
cd /home/ecospec/project/eco-spec-kg
git pull
```

如果 `git pull` 报错，提示以下文件会被覆盖：

```text
src/ecospec_kg/__pycache__/cli.cpython-311.pyc
src/ecospec_kg/__pycache__/extraction.cpython-311.pyc
```

这是 Python 自动生成的缓存文件，不是源码。可以丢弃后再拉取：

```bash
git checkout -- src/ecospec_kg/__pycache__/cli.cpython-311.pyc
git checkout -- src/ecospec_kg/__pycache__/extraction.cpython-311.pyc
git pull
```

拉取后重新安装 editable 包，确保新命令生效：

```bash
conda activate ecospec
pip install -e ".[ml,ui,dev]"
eco-spec-kg --help
```

如果命令列表中出现 `extract-llm`，说明 LLM 抽取入口已更新成功。

## 4. 创建 Python 环境

项目要求 Python 3.11 或更高版本。不要使用 Python 3.9 环境，否则会出现 Gradio 5.x 无法安装的问题。

先检查 Python 版本：

```bash
python3 --version
```

### 4.1 推荐方式：使用 conda

```bash
cd /home/ecospec/project/eco-spec-kg

conda create -n ecospec python=3.11 -y
conda activate ecospec

python -m pip install --upgrade pip setuptools wheel
```

安装完整依赖：

```bash
pip install -e ".[ml,ui,dev]"
```

### 4.2 无 conda 时：使用 venv

Ubuntu 或 Debian：

```bash
apt update
apt install -y python3.11 python3.11-venv python3.11-dev
```

创建虚拟环境：

```bash
cd /home/ecospec/project/eco-spec-kg
mkdir -p /home/ecospec/venvs

python3.11 -m venv /home/ecospec/venvs/ecospec
source /home/ecospec/venvs/ecospec/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -e ".[ml,ui,dev]"
```

如果只需要先跑数据处理和命令行测试，不运行 Gradio 和 LoRA，可先安装基础依赖：

```bash
pip install -e ".[dev]"
```

如果只跑 LoRA，不启动网页演示：

```bash
pip install -e ".[ml,dev]"
```

### 4.3 验证安装

```bash
eco-spec-kg --help
python -c "import ecospec_kg; print('EcoSpec-KG import ok')"
```

如果要训练 LoRA，继续检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

输出 `True` 才说明当前环境能使用 CUDA。

## 5. 上传原始 PDF

原始 PDF 不进入 Git 仓库，需要单独上传到服务器。

本机原始 PDF 目录：

```text
D:\else\lky\生态论文重写版本\全国生态状况调查评估技术规范
```

从本机 PowerShell 上传到服务器：

```powershell
scp -r "D:\else\lky\生态论文重写版本\全国生态状况调查评估技术规范" root@服务器IP:/work/ecospec/corpus
```

服务器检查：

```bash
ls -lh /work/ecospec/corpus
```

设置环境变量：

```bash
export ECOSPEC_CORPUS_DIR=/work/ecospec/corpus
```

如果希望每次登录自动生效：

```bash
cat >> ~/.bashrc <<'EOF'
export PIP_CACHE_DIR=/home/ecospec/cache/pip
export HF_HOME=/work/ecospec/hf-cache
export TRANSFORMERS_CACHE=/work/ecospec/hf-cache
export TMPDIR=/work/ecospec/tmp
export ECOSPEC_CORPUS_DIR=/work/ecospec/corpus
EOF
source ~/.bashrc
```

## 6. 解析 PDF 生成 chunks

进入项目目录并激活环境：

```bash
cd /home/ecospec/project/eco-spec-kg
conda activate ecospec
```

如果使用 venv：

```bash
cd /home/ecospec/project/eco-spec-kg
source /home/ecospec/venvs/ecospec/bin/activate
```

执行 PDF 解析：

```bash
eco-spec-kg corpus \
  --source "$ECOSPEC_CORPUS_DIR" \
  --out data/processed
```

生成文件：

```text
data/processed/chunks.jsonl
data/processed/manifest.json
data/processed/summary.json
```

查看摘要：

```bash
cat data/processed/summary.json
```

正常情况下，去重后应识别 11 项 HJ 1166-HJ 1176 正式标准。若数量明显不对，优先检查 PDF 是否上传完整，以及是否把编制说明、DOCX 副本或重复文件混入了语料。

## 7. 抽取候选三元组

基于 chunks 抽取候选关系有两种方式：

```text
predict       # 规则基线，结果少，但可解释，作为 baseline
extract-llm   # LLM + Schema 约束抽取，正式扩充候选三元组使用
```

### 7.1 规则基线抽取

规则基线用于流程测试和对比实验：

```bash
eco-spec-kg predict \
  --chunks data/processed/chunks.jsonl \
  --out results/rule_predictions.jsonl
```

生成文件：

```text
results/rule_predictions.jsonl
results/rule_predictions.rejected.json
```

这里的记录不是普通裸三元组，而是带证据的关系记录，包含：

```text
头实体
关系类型
尾实体
标准编号
页码
章节
原文证据
抽取方法
置信度
审核状态
```

规则定义位置：

```text
src/ecospec_kg/extraction.py
```

具体类名：

```text
RuleExtractor
```

Schema 定义位置：

```text
src/ecospec_kg/schema.py
```

证据校验位置：

```text
src/ecospec_kg/evidence.py
```

### 7.2 LLM Schema 约束抽取

正式扩充三元组时，使用 LLM 抽取。当前阶段的目标是生成“候选三元组”，不要求必须使用本地模型；可以优先接入 DeepSeek、通义千问、智谱等 OpenAI-compatible 线上模型。线上模型负责候选生成，本地代码负责 Schema 校验、证据原文匹配和输出文件管理。

推荐优先级：

```text
线上强模型/OpenAI-compatible API  # 当前候选抽取阶段优先使用
本地 Transformers Qwen3-0.6B       # 后续本地基线、LoRA 前后对比使用
本地 vLLM OpenAI-compatible 服务   # 需要批量加速或统一服务接口时使用
```

#### 7.2.1 使用线上 DeepSeek/OpenAI-compatible 模型

先设置 API 环境变量。以下以 DeepSeek 兼容接口为例，模型名以你的服务商控制台实际可用模型为准：

```bash
export ECOSPEC_LLM_BASE_URL=https://api.deepseek.com
export ECOSPEC_LLM_API_KEY=你的API_KEY
export ECOSPEC_COMPLETION_MODEL=你的线上模型名
```

如果服务商要求 base URL 带 `/v1`，则设置为：

```bash
export ECOSPEC_LLM_BASE_URL=https://服务商域名/v1
```

先小批量试跑 5 个 chunk：

```bash
eco-spec-kg extract-llm \
  --provider openai \
  --chunks data/processed/chunks.jsonl \
  --standards "HJ 1172-2021" "HJ 1173-2021" \
  --out results/online_candidates_core_test.jsonl \
  --limit 5
```

`extract-llm` 默认会在终端打印逐 chunk 进度日志，格式类似：

```text
extract-llm start: provider=openai model=... chunks=5 standards=HJ 1172-2021,HJ 1173-2021 start=0 limit=5 out=...
[######----------------------] 1/5  20.0% chunk=HJ1172_2021-p2-c2 standard=HJ 1172-2021 page=2 section=- +accepted=2 +rejected=0 accepted=2 rejected=0
```

如果需要关闭进度日志，可加：

```bash
--quiet
```

检查候选数量和拒绝原因：

```bash
wc -l results/online_candidates_core_test.jsonl
cat results/online_candidates_core_test.rejected.json
head -n 3 results/online_candidates_core_test.jsonl
```

确认输出正常后，去掉 `--limit` 跑完整核心标准：

```bash
eco-spec-kg extract-llm \
  --provider openai \
  --chunks data/processed/chunks.jsonl \
  --standards "HJ 1172-2021" "HJ 1173-2021" \
  --out results/online_candidates_core.jsonl
```

参数说明：

```text
--provider transformers    # 在当前服务器上直接加载本地 Transformers 模型
--provider openai          # 调用 vLLM/OpenAI-compatible 服务
--model                    # provider=transformers 时指定本地模型；provider=openai 时通常由 ECOSPEC_COMPLETION_MODEL 指定
--standards                # 限定处理哪些标准
--limit                    # 限定处理多少个 chunk，先小批量试跑
--start                    # 从第几个 chunk 开始，便于断点分批
--max-relations-per-chunk  # 每个 chunk 最多保留多少条关系
```

`--standards "HJ 1172-2021" "HJ 1173-2021"` 表示只处理 HJ 1172 和 HJ 1173 两个核心标准。这样做是为了先围绕生态系统质量评估和生态系统服务功能评估构建核心训练候选，同时避免把 HJ 1171、HJ 1174、HJ 1175 这些跨规范测试集提前混入训练候选。

可以不跳过目录、附录标题和规范性引用文件。完整抽取保留这些候选有利于记录模型行为，后续在 `annotation_candidates.csv` 中将低价值或不适合训练的目录型关系标记为 `rejected` 即可。正式训练和测试只使用专家审核通过的 `accepted` 或 `consensus_accepted` 关系。

#### 7.2.2 使用本地 Transformers 模型

如果希望不用线上 API，也可以直接在 A100 服务器上加载本地模型：

```bash
eco-spec-kg extract-llm \
  --provider transformers \
  --model Qwen/Qwen3-0.6B \
  --chunks data/processed/chunks.jsonl \
  --standards "HJ 1172-2021" "HJ 1173-2021" \
  --out results/llm_candidates_core.jsonl
```

本地模型首次运行会下载模型文件，应提前设置缓存目录，避免占用根分区：

```bash
export HF_HOME=/work/ecospec/hf-cache
export TRANSFORMERS_CACHE=/work/ecospec/hf-cache
export TMPDIR=/work/ecospec/tmp
```

#### 7.2.3 使用本地 vLLM 服务

如果用 vLLM 服务，先启动 OpenAI-compatible endpoint，然后设置：

```bash
export ECOSPEC_LLM_BASE_URL=http://127.0.0.1:8000/v1
export ECOSPEC_LLM_API_KEY=local-token
export ECOSPEC_COMPLETION_MODEL=Qwen/Qwen3-0.6B
```

再运行：

```bash
eco-spec-kg extract-llm \
  --provider openai \
  --chunks data/processed/chunks.jsonl \
  --standards "HJ 1172-2021" "HJ 1173-2021" \
  --out results/llm_candidates_core.jsonl
```

LLM 抽取会自动执行 Schema 校验和证据校验。证据无法在原 chunk 中精确匹配的关系会写入与 `--out` 同名的拒绝文件。例如：

```text
results/online_candidates_core.rejected.json
results/llm_candidates_core.rejected.json
```

通过校验的候选关系会写入：

```text
results/online_candidates_core.jsonl
results/llm_candidates_core.jsonl
```

注意：LLM 输出仍然只是候选关系，不能直接作为训练集，必须进入专家审核。

## 8. 生成专家审核表

建议先合并规则基线和 LLM 候选，再生成 CSV 审核表：

```bash
cat results/rule_predictions.jsonl results/online_candidates_core.jsonl > results/candidates_all.jsonl
```

如果使用的是本地模型输出文件，则把 `results/online_candidates_core.jsonl` 替换为实际文件名，例如 `results/llm_candidates_core.jsonl`。

生成审核表：

```bash
eco-spec-kg annotate \
  --chunks data/processed/chunks.jsonl \
  --relations results/candidates_all.jsonl \
  --out data/annotations/annotation_candidates.csv
```

审核表位置：

```text
data/annotations/annotation_candidates.csv
```

如果需要下载到本地用 Excel 审核：

```powershell
scp root@服务器IP:/home/ecospec/project/eco-spec-kg/data/annotations/annotation_candidates.csv "D:\else\lky\生态论文重写版本\annotation_candidates.csv"
```

审核完成后再传回服务器：

```powershell
scp "D:\else\lky\生态论文重写版本\annotation_candidates.csv" root@服务器IP:/home/ecospec/project/eco-spec-kg/data/annotations/annotation_candidates.csv
```

## 9. 人工审核规则

重点填写以下列：

```text
reviewer_1_decision
reviewer_1_comment
reviewer_2_decision
reviewer_2_comment
consensus_decision
review_status
```

推荐使用以下取值：

```text
accepted              # 审核通过
consensus_accepted    # 双专家共识通过
rejected              # 审核不通过
pending               # 暂未审核
```

LoRA 训练只使用：

```text
accepted
consensus_accepted
```

审核判断标准：

1. 头实体、关系、尾实体是否符合原文。
2. 关系类型是否符合 Schema。
3. 证据文本是否能在对应页码和章节中找到。
4. 是否存在模型或规则额外推断。
5. 是否把计算公式、参数、单位、数据来源、质量要求混淆。

不确定的样本不要强行通过，先标为 `pending` 或 `rejected`。

## 10. 生成 reviewed_relations.jsonl

人工审核完成后，需要把审核表中的通过状态合并回 JSONL。

在服务器执行：

```bash
cd /home/ecospec/project/eco-spec-kg
conda activate ecospec
```

如果使用 venv：

```bash
source /home/ecospec/venvs/ecospec/bin/activate
```

执行转换脚本：

```bash
python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("/home/ecospec/project/eco-spec-kg")
csv_path = base / "data/annotations/annotation_candidates.csv"
pred_path = base / "results/candidates_all.jsonl"
out_path = base / "data/annotations/reviewed_relations.jsonl"

review = {}
with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        record_id = row["record_id"]
        status = (
            row.get("review_status")
            or row.get("consensus_decision")
            or ""
        ).strip().lower()
        if status in {"accept", "accepted", "consensus_accepted", "通过", "同意"}:
            review[record_id] = "consensus_accepted"
        elif status in {"reject", "rejected", "不通过", "拒绝"}:
            review[record_id] = "rejected"
        else:
            review[record_id] = "pending"

rows = []
with pred_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        obj["review_status"] = review.get(obj["relation_id"], "pending")
        if obj["review_status"] in {"accepted", "consensus_accepted"}:
            rows.append(obj)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="\n") as f:
    for obj in rows:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")

print(f"accepted={len(rows)}")
print(out_path)
PY
```

查看通过样本数量：

```bash
wc -l data/annotations/reviewed_relations.jsonl
```

正式 LoRA 训练建议至少准备 100 条以上通过样本，较稳妥是 300 条以上，并尽量覆盖以下类型：

```text
指标
公式
参数
单位
数据来源
处理方法
生态类型
时空范围
质量要求
标准条款
```

## 11. 划分训练集、验证集、测试集

划分原则：

1. 按完整知识路径分组划分，避免同一知识路径同时进入训练集和测试集。
2. HJ 1171、HJ 1174、HJ 1175 固定作为跨规范测试集。
3. LoRA 训练不能使用跨规范测试集样本。

执行：

```bash
python - <<'PY'
from pathlib import Path
from ecospec_kg.io_utils import read_jsonl, write_jsonl
from ecospec_kg.models import Relation
from ecospec_kg.split import group_split

base = Path("/home/ecospec/project/eco-spec-kg")
relations = [
    Relation.from_dict(row)
    for row in read_jsonl(base / "data/annotations/reviewed_relations.jsonl")
]

for seed in [42, 43, 44]:
    splits = group_split(relations, seed=seed)
    outdir = base / f"data/splits/seed_{seed}"
    for name, items in splits.items():
        write_jsonl(outdir / f"{name}.jsonl", (item.to_dict() for item in items))
    print(seed, {name: len(items) for name, items in splits.items()})
PY
```

生成文件：

```text
data/splits/seed_42/train.jsonl
data/splits/seed_42/validation.jsonl
data/splits/seed_42/test.jsonl
data/splits/seed_42/external_test.jsonl
data/splits/seed_43/...
data/splits/seed_44/...
```

## 12. LoRA 烟雾训练

烟雾训练用于确认：

1. 模型可以下载。
2. CUDA 可以使用。
3. 显存足够。
4. 训练代码能完整跑完。
5. adapter 可以保存。

执行 20 条样本烟雾测试：

```bash
eco-spec-kg train \
  --relations data/splits/seed_42/train.jsonl \
  --prepared data/training/qwen3_lora_seed42.jsonl \
  --out runs/lora/qwen3-0.6b/seed_42_smoke \
  --seed 42 \
  --smoke-limit 20 \
  --run
```

成功后应生成：

```text
runs/lora/qwen3-0.6b/seed_42_smoke/adapter
runs/lora/qwen3-0.6b/seed_42_smoke/run_manifest.json
```

查看运行清单：

```bash
cat runs/lora/qwen3-0.6b/seed_42_smoke/run_manifest.json
```

如果训练样本不足 20 条，可临时降低 `--smoke-limit`，例如：

```bash
eco-spec-kg train \
  --relations data/splits/seed_42/train.jsonl \
  --prepared data/training/qwen3_lora_seed42.jsonl \
  --out runs/lora/qwen3-0.6b/seed_42_smoke \
  --seed 42 \
  --smoke-limit 5 \
  --run
```

但正式实验不应使用过少训练样本支撑结论。

## 13. 三随机种子正式 LoRA 训练

烟雾测试通过后，运行正式训练：

```bash
for s in 42 43 44; do
  eco-spec-kg train \
    --relations data/splits/seed_${s}/train.jsonl \
    --prepared data/training/qwen3_lora_seed${s}.jsonl \
    --out runs/lora/qwen3-0.6b/seed_${s} \
    --seed ${s} \
    --run
done
```

默认 LoRA 参数：

```text
base model: Qwen/Qwen3-0.6B
rank: 8
alpha: 16
dropout: 0.05
learning rate: 2e-4
max epochs: 5
early stopping patience: 2
max sequence length: 2048
seeds: 42, 43, 44
```

这些参数会写入：

```text
runs/lora/qwen3-0.6b/seed_42/run_manifest.json
runs/lora/qwen3-0.6b/seed_43/run_manifest.json
runs/lora/qwen3-0.6b/seed_44/run_manifest.json
```

## 14. 显存不足时的处理

如果出现 CUDA out of memory，先确认 GPU：

```bash
nvidia-smi
```

可按顺序处理：

1. 停止其他占用 GPU 的任务。
2. 降低 batch size。
3. 降低 max sequence length。
4. 只跑 smoke-limit 确认流程。

当前 batch size 在源码中：

```text
src/ecospec_kg/training.py
```

可将：

```python
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=8
```

临时改为：

```python
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=16
```

修改训练参数后，论文中必须如实记录最终训练设置。

## 15. 建立检索索引

训练之外，也可以先构建原生 GraphRAG 索引用于演示和检索实验。

```bash
eco-spec-kg index \
  --adapter native \
  --chunks data/processed/chunks.jsonl \
  --relations results/candidates_all.jsonl \
  --out data/index/native
```

如果希望只使用审核通过关系：

```bash
eco-spec-kg index \
  --adapter native \
  --chunks data/processed/chunks.jsonl \
  --relations data/annotations/reviewed_relations.jsonl \
  --out data/index/native
```

## 16. 启动 Gradio 演示

```bash
eco-spec-kg serve \
  --data data \
  --host 0.0.0.0 \
  --port 7860
```

浏览器访问：

```text
http://服务器IP:7860
```

如果服务器有防火墙，需要开放端口：

```bash
ufw allow 7860/tcp
```

或用 SSH 端口转发：

```powershell
ssh -L 7860:127.0.0.1:7860 root@服务器IP
```

然后本地浏览器访问：

```text
http://127.0.0.1:7860
```

## 17. 运行评估

评估需要 gold 文件和 predictions 文件。最基础命令：

```bash
eco-spec-kg evaluate \
  --gold data/splits/seed_42/test.jsonl \
  --predictions results/rule_predictions.jsonl \
  --chunks data/processed/chunks.jsonl \
  --out results/evaluation_seed42.json
```

生成报告：

```bash
eco-spec-kg report \
  --results results/evaluation_seed42.json \
  --out results/evaluation_seed42.md
```

没有真实运行得到的指标不得手工填写到论文中。未运行结果继续保留 `待实验补充`。

## 18. 生成消融实验计划

```bash
eco-spec-kg ablate \
  --out results/ablation_plan.json
```

消融实验至少覆盖：

```text
去除 Schema
去除 GraphRAG
去除 LoRA
去除 few-shot
去除证据校验
不同 chunk 长度
不同 shot 数量
不同候选阈值
```

## 19. 后台运行

长期运行 Gradio：

```bash
nohup eco-spec-kg serve --data data --host 0.0.0.0 --port 7860 > gradio.out.log 2> gradio.err.log &
```

查看进程：

```bash
ps -ef | grep eco-spec-kg
```

查看日志：

```bash
tail -f gradio.out.log
tail -f gradio.err.log
```

停止服务：

```bash
pkill -f "eco-spec-kg serve"
```

## 20. 常见问题

### 20.1 gradio>=5 安装失败

典型报错：

```text
ERROR: No matching distribution found for gradio<7,>=5.0
```

原因通常是 Python 版本过低，例如 Python 3.9。

处理：

```bash
python --version
conda create -n ecospec python=3.11 -y
conda activate ecospec
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[ml,ui,dev]"
```

### 20.2 找不到 PDF

检查环境变量：

```bash
echo $ECOSPEC_CORPUS_DIR
ls -lh "$ECOSPEC_CORPUS_DIR"
```

如果为空：

```bash
export ECOSPEC_CORPUS_DIR=/work/ecospec/corpus
```

### 20.3 eco-spec-kg 命令不存在

说明环境未激活或项目未安装。

```bash
cd /home/ecospec/project/eco-spec-kg
conda activate ecospec
pip install -e ".[ml,ui,dev]"
eco-spec-kg --help
```

### 20.4 CUDA 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果 `nvidia-smi` 正常但 PyTorch 不识别 CUDA，需要安装匹配 CUDA 的 PyTorch 版本。

### 20.5 Hugging Face 模型下载慢或失败

可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新运行训练命令。

### 20.6 训练样本太少

如果出现：

```text
at least two reviewed training records are required
```

说明 `reviewed_relations.jsonl` 中审核通过样本不足。需要继续补充人工标注和审核。

### 20.7 git pull 被 __pycache__ 阻塞

如果出现：

```text
Your local changes to the following files would be overwritten by merge:
src/ecospec_kg/__pycache__/cli.cpython-311.pyc
src/ecospec_kg/__pycache__/extraction.cpython-311.pyc
```

执行：

```bash
cd /home/ecospec/project/eco-spec-kg
git checkout -- src/ecospec_kg/__pycache__/cli.cpython-311.pyc
git checkout -- src/ecospec_kg/__pycache__/extraction.cpython-311.pyc
git pull
pip install -e ".[ml,ui,dev]"
```

### 20.8 extract-llm 命令没有出现

说明服务器代码还没有更新到包含 LLM 抽取入口的版本，或 editable 安装没有刷新。

```bash
cd /home/ecospec/project/eco-spec-kg
git pull
pip install -e ".[ml,ui,dev]"
eco-spec-kg --help
```

命令列表中应包含：

```text
extract-llm
```

### 20.9 根分区空间不足

如果 pip 安装报错：

```text
OSError: [Errno 28] No space left on device
```

先检查：

```bash
df -h
du -xhd1 / 2>/dev/null | sort -h
```

不要把模型、虚拟环境和 PDF 放在 `/root` 或 `/tmp`。推荐：

```bash
export PIP_CACHE_DIR=/home/ecospec/cache/pip
export HF_HOME=/work/ecospec/hf-cache
export TRANSFORMERS_CACHE=/work/ecospec/hf-cache
export TMPDIR=/work/ecospec/tmp
```

## 21. 不应提交到 Git 的内容

以下内容不要提交到公开 Git 仓库：

```text
原始 PDF
data/processed/
data/index/
data/training/
runs/
results/rule_predictions*
results/*candidates*.jsonl
results/*.rejected.json
.env
.venv/
__pycache__/
*.pyc
```

这些内容应保留在服务器本地或通过内部数据盘管理。

## 22. 论文结论边界

实验和论文表述必须遵守以下边界：

1. 可以说明方法提升了规范知识组织、检索、补全和机器可读性。
2. 可以报告实体抽取、关系抽取、证据溯源、检索命中、专家接受率等指标。
3. 不能声称已经验证真实区域生态评估计算效果。
4. 不能声称替代遥感、GIS、野外调查或生态过程模型。
5. 未真实运行的实验结果必须保留为 `待实验补充`。

## 23. 推荐最小执行命令清单

如果只想从 PDF 到演示先跑通，执行：

```bash
cd /home/ecospec/project/eco-spec-kg
conda activate ecospec
export ECOSPEC_CORPUS_DIR=/work/ecospec/corpus

eco-spec-kg corpus --source "$ECOSPEC_CORPUS_DIR" --out data/processed
eco-spec-kg predict --chunks data/processed/chunks.jsonl --out results/rule_predictions.jsonl
eco-spec-kg extract-llm --provider openai --chunks data/processed/chunks.jsonl --standards "HJ 1172-2021" "HJ 1173-2021" --out results/online_candidates_core_test.jsonl --limit 5
eco-spec-kg extract-llm --provider openai --chunks data/processed/chunks.jsonl --standards "HJ 1172-2021" "HJ 1173-2021" --out results/online_candidates_core.jsonl
cat results/rule_predictions.jsonl results/online_candidates_core.jsonl > results/candidates_all.jsonl
eco-spec-kg annotate --chunks data/processed/chunks.jsonl --relations results/candidates_all.jsonl --out data/annotations/annotation_candidates.csv
eco-spec-kg index --adapter native --chunks data/processed/chunks.jsonl --relations results/candidates_all.jsonl --out data/index/native
eco-spec-kg serve --data data --host 0.0.0.0 --port 7860
```

如果要做 LoRA，必须先完成人工审核，再执行：

```bash
# 先按第 10 节生成 data/annotations/reviewed_relations.jsonl
# 再按第 11 节生成 data/splits/seed_42 等划分文件

eco-spec-kg train \
  --relations data/splits/seed_42/train.jsonl \
  --prepared data/training/qwen3_lora_seed42.jsonl \
  --out runs/lora/qwen3-0.6b/seed_42_smoke \
  --seed 42 \
  --smoke-limit 20 \
  --run
```
