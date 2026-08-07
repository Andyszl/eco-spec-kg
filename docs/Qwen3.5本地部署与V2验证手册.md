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
conda create -n ecospec-vllm-cu130 python=3.12 -y
conda activate ecospec-vllm-cu130
python -m pip install -U pip uv
uv pip install "vllm==0.20.1" --torch-backend=cu130

python -c "import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda)"
python -c "import vllm._C; print('vLLM CUDA extension OK')"
```

该组合应输出 vLLM 0.20.1、PyTorch `2.11.0+cu130` 和 CUDA `13.0`。
不要使用 `--torch-backend=auto`；它可能选择 cu129，而 vLLM 0.20.1 的默认扩展
是按 cu130 构建的，混用会导致缺少 `libcudart.so.13`。

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
conda activate ecospec-vllm-cu130
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

零样本配置将 `max_tokens` 固定为 3072。开发集中最长已知请求约含 4023 个输入
token，与 3072 个输出 token 合计不超过当前 8192 token 上下文；同时 Prompt 要求
模型输出单行紧凑 JSON，避免候选较多时因格式化换行耗尽输出预算。若响应仍以
`finish_reason=length` 结束，应视为抽取失败，不能绕过验证门直接评测。

## 9. 准备 V2 LoRA 数据

```bash
python -m ecospec_kg.cli_v2 prepare-lora-v2 \
  --units "$DATA_PACKAGE/blind/train_units.jsonl" \
  --annotations "$DATA_PACKAGE/gold/train_annotations.jsonl" \
  --out /home/hello/szl/eco-spec-data/training/qwen35_v21_train.jsonl

python -m json.tool \
  /home/hello/szl/eco-spec-data/training/qwen35_v21_train.manifest.json
```

命令只接受明确标记为 `split=train` 的标注。当前包预期为 693 条训练记录。
`structure-aware-rule-v2.1` 在当前冻结训练包上的候选实体召回上限约为 0.9019，
关系召回上限约为 0.8148；开发集两项候选召回均为 1.0。若结果差异明显，先核对
冻结包哈希、候选生成器版本和 Git 版本。关系候选上限仍未达到 0.9，因此正式训练
结果必须同时报告该上限，不能将模型漏检与候选缺失混为一项误差。

## 10. LoRA 烟雾训练

Qwen3.5 包含线性注意力层。训练环境先安装并验证官方建议的 FLA 内核：

```bash
conda activate ecospec-train
python -m pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation

python - <<'PY'
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print("flash-linear-attention OK")
PY
```

项目生成的 `swift sft` 命令会显式设置 `padding_free=false`、`packing=false` 和
`sequence_parallel_size=1`。这符合当前单卡文本 LoRA 实验设计，也避免未声明地切换
训练语义；FLA 仍用于 Qwen3.5 线性注意力的高效实现。

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
  --prepared /home/hello/szl/eco-spec-data/training/qwen35_v21_train.jsonl \
  --out "$ECOSPEC_RUNS/lora/qwen35-9b/seed_42_smoke" \
  --model "$QWEN_LLM" \
  --trainer swift \
  --precision bf16 \
  --max-length 8192 \
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

当前 693 条 V2.1 训练记录的实测长度分布为：P95=1034、P99=3447、最大值=7293；
其中 5 条超过 4096，没有记录超过 8192。因此训练统一使用 `max-length 8192`，不能
以 4096 静默截断最长样本。使用 20 份最长样本进行的 8192-token 压力测试已完成
2 个训练 step，峰值显存约 23.04 GiB，未出现 OOM 或 NaN。该重复样本运行只用于
容量验收，不作为模型效果实验。

若训练在第一个 step 前出现
`Qwen3.5 linear attention padding free/sequence parallel requires flash-linear-attention`，
先执行本节的 FLA 导入验证。不要通过开启 packing 或 sequence parallel 绕过该错误。
