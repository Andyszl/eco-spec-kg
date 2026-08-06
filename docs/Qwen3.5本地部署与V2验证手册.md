# Qwen3.5 本地部署与 V2 验证手册

本手册覆盖本地代码提交、服务器更新、Qwen3.5-9B 推理、Embedding 验收、
V2 开发集抽取验证和 LoRA 烟雾训练。模型、冻结数据、金标和运行结果不进入
公开 Git 仓库。

## 1. 本地提交代码

在 Windows PowerShell 中进入项目：

```powershell
Set-Location 'D:\else\lky\生态论文重写版本\eco-spec-kg'
git status --short
```

只暂存源码、配置、测试和文档：

```powershell
git add -- `
  .gitignore README.md pyproject.toml `
  src/ecospec_kg/cli.py `
  src/ecospec_kg/providers.py `
  src/ecospec_kg/training.py `
  src/ecospec_kg/training_v2.py `
  src/ecospec_kg/cli_v2.py `
  src/ecospec_kg/experiment_data_v2.py `
  src/ecospec_kg/experiment_io_v2.py `
  src/ecospec_kg/extractor_v2.py `
  src/ecospec_kg/evaluation_v2.py `
  src/ecospec_kg/prediction_contract_v2.py `
  src/ecospec_kg/prediction_validation_v2.py `
  src/ecospec_kg/ontology_v2.py `
  src/ecospec_kg/foundation_quality.py `
  src/ecospec_kg/layout_parser.py `
  src/ecospec_kg/source_units.py `
  src/ecospec_kg/annotation_v2.py `
  src/ecospec_kg/dataset_quality_v2.py `
  src/ecospec_kg/pilot_quality_v2.py `
  config/parse_pilot_v2.json `
  config/parse_acceptance_11_additions_v2.json `
  config/formula_overrides_v2.json `
  config/experiments_v2/rule_baseline.json `
  config/experiments_v2/qwen35_9b_zero_shot.json `
  tests/test_experiment_chain_v2.py `
  tests/test_dataset_v2.py `
  tests/test_foundations_v2.py `
  tests/test_qwen_runtime.py `
  docs/V2正式实验命令链.md `
  docs/服务器运行与LoRA训练操作手册.md `
  docs/Qwen3.5本地部署与V2验证手册.md
```

检查暂存内容，确认没有 `gold`、PDF、模型或运行结果：

```powershell
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only | Select-String 'data/frozen|results|runs|\.pdf|safetensors'
```

最后一条命令应无输出。提交并推送：

```powershell
git commit -m "Add Qwen3.5 V2 extraction and LoRA runtime"
git push origin main
```

## 2. 私下传输冻结数据

冻结包包含训练标注和测试金标，不能推送到公开 Git。使用 SCP 单独传输：

```powershell
ssh hello@服务器IP "mkdir -p /home/hello/szl/eco-spec-data/frozen"
scp -r '.\data\frozen\v2.1' hello@服务器IP:/home/hello/szl/eco-spec-data/frozen/
```

测试金标仅供最后的 `evaluate-v2` 进程读取，不得复制到模型服务目录、GraphRAG
索引或抽取提示中。

## 3. 服务器拉取代码

```bash
export PROJECT=/home/hello/szl/eco-spec-kg
export DATA_PACKAGE=/home/hello/szl/eco-spec-data/frozen/v2.1
export QWEN_LLM=/home/hello/szl/models/Qwen3.5-9B
export QWEN_EMBED=/home/hello/szl/models/Qwen3-Embedding-0.6B
export ECOSPEC_RUNS=/home/hello/szl/eco-spec-runs
export ECOSPEC_LOGS=/home/hello/szl/eco-spec-logs

mkdir -p /home/hello/szl/eco-spec-data/training "$ECOSPEC_RUNS" "$ECOSPEC_LOGS"
if [ -d "$PROJECT/.git" ]; then
  git -C "$PROJECT" pull --ff-only origin main
else
  git clone https://github.com/Andyszl/eco-spec-kg.git "$PROJECT"
fi
cd "$PROJECT"
```

已有仓库时只需执行 `git pull --ff-only origin main`。不要在服务器代码目录保留
手工源码修改。

## 4. 建立两个隔离环境

vLLM 与训练框架分别安装，避免它们对 PyTorch、Transformers 的版本要求互相覆盖。

推理环境：

```bash
conda create -n ecospec-vllm python=3.12 -y
conda activate ecospec-vllm
python -m pip install -U pip uv
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
```

训练和项目环境：

```bash
conda create -n ecospec-train python=3.12 -y
conda activate ecospec-train
cd "$PROJECT"
python -m pip install -U pip
pip install -e ".[ml,qwen35,dev]"
```

安装后 `ms-swift` 必须是 4.x。若镜像源错误地安装了 1.x，使用官方 PyPI
重新安装：

```bash
pip uninstall -y ms-swift
pip install --no-cache-dir --index-url https://pypi.org/simple \
  "ms-swift>=4.1,<5"
pip install -e ".[ml,qwen35,dev]"
```

验收命令入口：

```bash
eco-spec-kg --help
eco-spec-kg-v2 --help
swift sft --help >/dev/null && echo "ms-swift OK"
python -c "from importlib.metadata import version; print('ms-swift', version('ms-swift'))"
python -m pytest
git status --short
```

测试应全部通过；`git status --short` 应无输出。

## 5. 模型和 GPU 验收

```bash
test -f "$QWEN_LLM/config.json" && echo "LLM files OK"
test -f "$QWEN_EMBED/config.json" && echo "Embedding files OK"
nvidia-smi
```

训练环境中检查 CUDA：

```bash
conda activate ecospec-train
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("bf16", torch.cuda.is_bf16_supported())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

验收要求为 CUDA 可用、两张 GPU 可见且 BF16 为真。

## 6. Embedding 验收

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(os.environ["QWEN_EMBED"], device="cuda")
texts = [
    "生态系统质量指数的计算方法",
    "生态系统服务功能评估的数据来源",
    "今天的天气情况",
]
vectors = model.encode(texts, normalize_embeddings=True)
print("shape", vectors.shape)
print("related", float(vectors[0] @ vectors[1]))
print("unrelated", float(vectors[0] @ vectors[2]))
PY
```

要求输出形状为 `(3, 1024)`，并且没有权重或模型类型错误。

## 7. 启动 Qwen3.5-9B

```bash
conda activate ecospec-vllm
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$QWEN_LLM" \
  --served-model-name Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --language-model-only \
  > "$ECOSPEC_LOGS/qwen35-vllm.log" 2>&1 &
echo $! > "$ECOSPEC_LOGS/qwen35-vllm.pid"
```

查看启动日志和模型列表：

```bash
tail -n 50 "$ECOSPEC_LOGS/qwen35-vllm.log"
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

非思考 JSON 验收：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-token' \
  -d '{
    "model":"Qwen3.5-9B",
    "messages":[{"role":"user","content":"只输出JSON：{\"status\":\"ok\"}"}],
    "temperature":0,
    "max_tokens":128,
    "chat_template_kwargs":{"enable_thinking":false}
  }' | python -m json.tool
```

## 8. 开发集 V2 抽取与验证

```bash
conda activate ecospec-train
cd "$PROJECT"
export ECOSPEC_LLM_BASE_URL=http://127.0.0.1:8000/v1
export ECOSPEC_LLM_API_KEY=local-token
export ECOSPEC_COMPLETION_MODEL=Qwen3.5-9B
export ECOSPEC_LLM_ENABLE_THINKING=false
unset ECOSPEC_LLM_APPEND_NO_THINK

python -m ecospec_kg.cli_v2 extract-v2 \
  --units "$DATA_PACKAGE/blind/dev_units.jsonl" \
  --config config/experiments_v2/qwen35_9b_zero_shot.json \
  --out "$ECOSPEC_RUNS/v2.1/qwen35_zero_shot/dev_seed_42"

python -m ecospec_kg.cli_v2 validate-predictions-v2 \
  --units "$DATA_PACKAGE/blind/dev_units.jsonl" \
  --predictions "$ECOSPEC_RUNS/v2.1/qwen35_zero_shot/dev_seed_42/predictions.jsonl" \
  --schema "$DATA_PACKAGE/schema_v2.json" \
  --out "$ECOSPEC_RUNS/v2.1/qwen35_zero_shot/dev_seed_42/validation"
```

`validation_report.json` 中必须为 `"passed": true`，且失败数为 0。

开发集验证通过后，可以检查完整评测链是否可运行：

```bash
python -m ecospec_kg.cli_v2 evaluate-v2 \
  --gold "$DATA_PACKAGE/gold/dev_annotations.jsonl" \
  --predictions "$ECOSPEC_RUNS/v2.1/qwen35_zero_shot/dev_seed_42/validation/validated_predictions.jsonl" \
  --units "$DATA_PACKAGE/blind/dev_units.jsonl" \
  --validation-report "$ECOSPEC_RUNS/v2.1/qwen35_zero_shot/dev_seed_42/validation/validation_report.json" \
  --out "$ECOSPEC_RUNS/v2.1/qwen35_zero_shot/dev_seed_42/evaluation"

python -m json.tool \
  "$ECOSPEC_RUNS/v2.1/qwen35_zero_shot/dev_seed_42/evaluation/metrics.json"
```

该命令只用于开发集调试。测试集评测必须等方法、Prompt、阈值和模型版本全部冻结后
执行一次。

## 9. 准备 V2 LoRA 数据

```bash
python -m ecospec_kg.cli_v2 prepare-lora-v2 \
  --units "$DATA_PACKAGE/blind/train_units.jsonl" \
  --annotations "$DATA_PACKAGE/gold/train_annotations.jsonl" \
  --out /home/hello/szl/eco-spec-data/training/qwen35_v2_train.jsonl

python -m json.tool \
  /home/hello/szl/eco-spec-data/training/qwen35_v2_train.manifest.json
```

命令只接受明确标记为 `split=train` 的标注。当前包预期为 693 条训练记录；
候选关系召回上限约 0.7622。若结果差异明显，先核对冻结包哈希和 Git 版本。

## 10. LoRA 烟雾训练

运行训练前停止 vLLM，释放 GPU 0：

```bash
kill "$(cat "$ECOSPEC_LOGS/qwen35-vllm.pid")"
nvidia-smi
```

先使用 20 条样本：

```bash
conda activate ecospec-train
cd "$PROJECT"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
eco-spec-kg train \
  --prepared /home/hello/szl/eco-spec-data/training/qwen35_v2_train.jsonl \
  --out "$ECOSPEC_RUNS/lora/qwen35-9b/seed_42_smoke" \
  --model "$QWEN_LLM" \
  --trainer swift \
  --precision bf16 \
  --max-length 4096 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --learning-rate 2e-4 \
  --epochs 1 \
  --seed 42 \
  --smoke-limit 20 \
  --run
```

验证：

```bash
python -m json.tool \
  "$ECOSPEC_RUNS/lora/qwen35-9b/seed_42_smoke/run_manifest.json"
find "$ECOSPEC_RUNS/lora/qwen35-9b/seed_42_smoke" \
  -name adapter_config.json -o -name adapter_model.safetensors
nvidia-smi
```

只有运行清单为 `complete`、适配器文件存在、训练日志没有 NaN/OOM，才进入三随机
种子正式训练。当前候选覆盖上限不足，因此完成烟雾训练后应先改进候选生成器，不能
直接把结果写入论文正式实验表。
