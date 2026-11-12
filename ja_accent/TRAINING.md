# 在 NVIDIA GPU 上训练日语音调微调

数据准备（`data/build_dataset.py` → `ja_accent.ft.prepare`）已在别处完成，产物是 `data/prep/`。
训练只需要这个目录和项目代码，所有输出都写在本项目的 `runs/<名字>/` 下。

## 1. 把项目拷到 GPU 机器

`data/prep/`（约 130 MB）已提交到仓库，原始录音不在仓库里，训练也用不到：

```bash
git clone git@github.com:Ksuriuri/kokoro.git && cd kokoro
```

## 2. 环境

- NVIDIA 驱动：`nvidia-smi` 显示的 CUDA Version ≥ 12.4。
- `pyopenjtalk`、`pyworld` 在 Linux 上从源码编译，需要：
  `sudo apt install build-essential cmake python3-dev`
- uv：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 首次运行会联网下载：Kokoro-82M 权重（Hugging Face）和 OpenJTalk 词典（GitHub）。
  Hugging Face 访问慢可设 `export HF_ENDPOINT=https://hf-mirror.com`。

锁定的是 torch 2.6.0（CUDA 12.4 版），支持到 Hopper / Ada（H100、A100、RTX 4090/3090 等）。
**RTX 50 系 / B200 等 Blackwell 卡**需要 CUDA 12.8 版 torch，装好后用 `NO_SYNC=1` 启动，否则会被 `uv sync` 换回去：

```bash
uv sync --extra ja --extra ja-train
uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
NO_SYNC=1 bash scripts/train_cuda.sh
```

## 3. 启动

```bash
cd ~/kokoro
bash scripts/train_cuda.sh                                   # → runs/exp1，默认超参
RUN=exp2 GPU=1 bash scripts/train_cuda.sh --steps 4000 --w-logf0 3
```

脚本会依次：`uv sync`、确认 torch 能用这张卡、在 GPU 上跑一遍 smoke 自检（只在该 run 第一次启动时跑）、后台启动训练。
断开 ssh 不影响训练。同一个 `RUN` 再执行一次会从 `runs/$RUN/ckpt_last.pt` 续训；想从头训就换个 `RUN` 名。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `RUN` | `exp1` | 输出目录 `runs/$RUN` |
| `GPU` | `0` | 使用哪张卡 |
| `DATA` | `data/prep` | prepare 的输出目录 |
| `NO_SYNC=1` | | 不执行 `uv sync`（保留手动装的 torch） |
| `SKIP_SMOKE=1` | | 跳过 GPU 自检 |

其余参数原样传给 `python -m ja_accent.ft.train`，全部选项见 `--help`。

## 4. 监控

```bash
tail -f runs/exp1/train.out     # 训练日志
kill $(cat runs/exp1/train.pid) # 停止（之后可续训，最多丢 --ckpt-every 步）
```

`runs/exp1/` 内容：

- `log.jsonl`：每 10 步的各项 loss；每 250 步的验证 loss（`val_*`）和 `eval_accent`
- `eval/stepXXXXXX_NN.wav`：每次评估时合成的最小对立句（箸/橋/端 等），直接听
- `ckpt_last.pt`（每 500 步）、`ckpt_best.pt`（验证 loss 最低），每个约 800 MB

`eval_accent` 是合成语音的音高走向与声调标记一致的比例。原版 Kokoro（`jf_alpha`）在这组句子上约 0.84，训练后应高于它。

**选 checkpoint**：`ckpt_best.pt` 只按 mel + STFT + 时长 loss 选，不看音调。看音调准确率最高的是哪一步：

```bash
uv run --no-sync python -c "
import json
rows = [json.loads(l) for l in open('runs/exp1/log.jsonl') if 'eval_accent' in l]
for r in rows: print(r['step'], r['eval_accent'], r.get('val_loss'))
"
```

## 5. 导出与试听

```bash
uv run --no-sync python -m ja_accent.ft.infer --ckpt runs/exp1/ckpt_best.pt --text 箸と橋と端。 --out out.wav
uv run --no-sync python -m ja_accent.ft.export --ckpt runs/exp1/ckpt_best.pt --out export/
```

## 6. 常见问题

- **日志里出现 `OOM on ...`**：这条被跳过，偶尔几条无妨。很多的话减小 `--max-frames`（默认 800 帧 = 20 秒），会丢掉更长的句子。
- **训练集过小**：这份数据约 0.74 小时，默认 6000 步 × 8 条 ≈ 80 轮。如果 `val_loss` 先降后升、或 `eval_accent` 不再上升，用更少的 `--steps`。
