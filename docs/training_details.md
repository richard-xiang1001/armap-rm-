# training_details.md — Text-only Reward Model (RM) Training Protocol

> 目标：复现 ARMAP 的 **Reward Model 训练方法**（不追求论文指标对齐），产出一套**可开源、可审计、可迁移**的训练流程与细节说明。

本仓库实现的是一个 **text-only** RM：输入 `(instruction, trajectory)` 输出标量 reward，并用**偏好对** `(pos, neg)` 的 pairwise loss 训练。

---

## 0. 训练对象与目标函数

- 评分函数：`R(x, h)`
  - `x`: 指令/目标（instruction）
  - `h`: 轨迹（trajectory），通常由多步交互/推理/动作组成
- 训练数据：三元组 `(x, h_pos, h_neg)`，其中 `h_pos` 比 `h_neg` 更符合目标。
- 优化目标（pairwise logistic）：

\[
\mathcal{L} = -\mathbb{E}[\log \sigma(R(x,h^+) - R(x,h^-))]
\]

实现中使用 `softplus(-(r_pos-r_neg))` 计算，数值更稳定。

---

## 1. 设备与资源配置（上限：单卡 4090）

### 1.1 推荐硬件

- GPU：
  - **最小可跑**：12GB 显存（例如 3060 12GB）
  - **推荐**：24GB（RTX 4090）
- CPU：8 vCPU 以上
- RAM：16GB 以上（推荐 32GB）
- 磁盘：20GB 以上（推荐 50GB）

### 1.2 为什么 4090 足够

本 repo 的 text-only RM 不依赖超大模型，默认参数量较小（char embedding + BiGRU）。
即使你后续换成 1B–3B 的开源 LLM 做 backbone，4090 仍可通过：
- LoRA / QLoRA
- 梯度累积
- max_len 控制
实现稳定训练。

---

## 2. 环境与可复现性

### 2.1 Python/依赖

- Python ≥ 3.10
- 依赖见 `requirements.txt`

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2.2 可复现设置（务必公开）

开源时请在 `docs/reproducibility.md` 补全：
- OS/Driver/CUDA 版本
- torch 版本
- seed
- 完整命令行
- git commit hash

---

## 3. 数据协议（Dataset Schema）

每行一个 JSON（jsonl）：

```json
{
  "instruction_raw": "...",
  "instruction_refined": "...",
  "traj_pos": "...",
  "traj_neg": "...",
  "meta": {
    "env": "...",
    "id": 123,
    "construction_method": "...",
    "checks": {"...": true}
  }
}
```

### 3.1 本 repo 的最小数据生成

`scripts/prepare_data.py` 生成合成算术任务：
- `traj_pos`：格式一致、步骤合理、最终答案正确
- `traj_neg`：与正例“看起来一样像在做题”，但在最终或关键步骤上微小偏离

推荐在训练前执行数据体检：

```bash
python3 scripts/lint_data.py \
  --path data/train.jsonl \
  --max_len 512 \
  --last_k_steps 3 \
  --step_markers "step,action:,observation:,obs:"
```

若你的环境有自定义泄漏字段（如 `task_success` / `episode_return`），可追加：

```bash
python3 scripts/lint_data.py \
  --path data/train.jsonl \
  --leak_terms_file configs/leak_terms.txt \
  --fail_on_leak
```

### 3.2 你后续迁移到真实环境（Webshop/你自研 env）时要补的字段

建议在 `meta` 中补充：
- `steps`: 结构化 step 列表（obs/action/metadata），并保留原始文本版本
- `termination`: done / timeout / error
- `env_success`: 由环境判定的成功标签（⚠️ 注意：训练输入里要剥离/掩码，避免泄漏）

---

## 4. 模型与输入拼接

### 4.1 输入拼接规则

训练时每条样本会构造两段文本：
- `text_pos = instruction + SEP + traj_pos`
- `text_neg = instruction + SEP + traj_neg`

默认 `SEP = "\n\n---\n\n"`。

### 4.2 模型结构（text-only baseline）

- CharTokenizer（字符级 tokenizer，完全离线、稳定可复现）
- TextRewardModel：
  - Embedding
  - BiGRU encoder
  - masked mean pooling
  - linear head -> scalar reward

> 这对应论文的“backbone + scalar head”结构，只是把多模态 VLM backbone 替换为轻量文本 backbone。

---

## 5. 训练配置（建议默认值）

### 5.1 建议超参

- `max_len`: 512（轨迹更长可上 1024）
- `batch_size`: 64（显存不足用 16/32 + 梯度累积）
- `lr`: 3e-4（小模型）；若换成大 backbone，常见范围 1e-5 ~ 2e-4
- `weight_decay`: 0.01
- `epochs`: 3
- `grad_clip`: 1.0

### 5.2 训练命令

```bash
python3 scripts/prepare_data.py --out_dir data --n_train 20000 --n_valid 2000 --seed 42

python3 -m rm.train \
  --train_path data/train.jsonl \
  --valid_path data/valid.jsonl \
  --save_dir results/exp1 \
  --max_len 512 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 3 \
  --seed 42
```

训练会输出：
- `results/exp1/metrics.jsonl`：每个 epoch 的 valid 指标
- `results/exp1/rm.pt`：最优 checkpoint

---

## 6. 评估指标（不追论文数值，但要证明“学到了偏好”）

本 repo 默认输出：

1) **Pair Accuracy**：`P(R(pos) > R(neg))`
- 期望：明显高于 0.5

2) **Reward Gap**：`E[R(pos) - R(neg)]`
- 期望：均值为正，且方差不过度爆炸

3) **Reward Gap 分位数**：`p10/p50/p90` of `R(pos)-R(neg)`
- 用于判断是否少量极端样本拉动均值

4) （建议你加到后续版本）Calibration：
- 按 reward 分桶，看高分桶的真实成功率是否更高（需要少量环境回放）

> 口径说明：`avg_reward_gap` 是辅助监控项，跨实验不建议直接比较绝对值；主要关注其符号、分位数稳定性、以及是否出现发散。

评估命令：

```bash
python3 -m rm.eval --ckpt results/exp1/rm.pt --valid_path data/valid.jsonl --max_len 512
```

在受限 macOS/sandbox 环境，若出现 `torch_shm_manager ... Operation not permitted`，可将训练/评估命令补充：

```bash
--num_workers 0
```

---

## 7. 关键工程细节（论文通常不写，但你开源必须写透）

### 7.1 防止“标签泄漏”

严禁让训练输入中出现如下字段的原文：
- `reward=...`, `success=true`, `done=true`, `score=...`
- 任何显式的“正确/错误/失败/成功”标记

处理方式：
- 轨迹输入前做 regex 清洗/掩码
- 把环境返回的 reward/success 放在 `meta`，但不拼进 `traj_*` 文本

### 7.2 长轨迹截断策略（建议写成可配置）

简单截前 N tokens 可能截掉关键结尾。
建议策略（按优先级）：
- 保留末段 + 保留关键 action step
- 或基于 step 边界做“按步截断”

### 7.3 Hard Negatives 的比例

如果负例太蠢，RM 学到的是“识别胡话”，而不是“识别接近目标”。
建议：
- 30% easy negatives（明显错）
- 70% hard negatives（差一点成功，或只在关键一步偏离）

### 7.4 采样均衡

按轨迹长度、任务类型分桶采样，避免训练被某一类样本主导。

---

## 8. Troubleshooting：低于预期时的排障清单（Plan B）

### 症状 A：valid pair_accuracy ≈ 0.50（学不动）

可能原因与修复：
1) **数据过噪 / 正负差异太随机**
- 修复：提高 hard negative 的“结构相似度”，降低随机负例比例。

2) **输入截断导致关键信息被截掉**
- 修复：max_len 提高到 1024；或改用“保留末段”的截断策略。
- 经验：本 repo 合成数据若用 `max_len=128` 可能截掉 `Final` 行，导致 `traj_pos/traj_neg` 差异丢失，pair_accuracy 退化到随机附近。
- 可量化前置检查：
  - `truncation_risk_ratio_final_line`
  - `truncation_risk_last_k_steps`（不依赖 `Final:`，适用于通用 step 轨迹）

#### Failure Story（真实踩坑记录）

一次 smoke run 使用 `max_len=128` 时，模型验证指标出现：
- `pair_accuracy` 接近随机
- `avg_reward_gap` 接近 0

根因是拼接后的 `(instruction + SEP + trajectory)` 被截断，`Final:` 行落在可见窗口之外，正负样本差异被抹平。
将 `max_len` 提升到 `512` 后，指标恢复到明显高于随机水平。该问题已通过 `scripts/lint_data.py` 的 `truncation_risk_ratio_final_line` 与 `truncation_risk_last_k_steps` 指标前置暴露。

3) **学习率不合适**
- 修复：尝试 lr ∈ {1e-4, 3e-4, 1e-3}（小模型）；加 warmup。

4) **模型容量太小**
- 修复：提升 `d_model/hidden_size/num_layers`；或换更强文本 backbone。

### 症状 B：pair_accuracy 很高，但泛化差（过拟合/走捷径）

1) **泄漏**：轨迹里含 success/reward 字样
- 修复：强制清洗。

2) **负例太容易**
- 修复：增加 hard negative，占比提高。

### 症状 C：reward_gap 爆炸/训练不稳定

- 修复：
  - 降低 lr
  - 增加 `grad_clip`
  - 增加 weight_decay
  - 检查 batch 中是否有极端长文本

---

## 9. 开源清单（你承诺公开的“训练细节”建议逐条对齐）

请在开源时至少公开：
- 数据生成版本、过滤规则、schema
- 训练命令、所有超参、seed
- 日志字段定义与解释
- 训练曲线（pair_acc, reward_gap, loss）
- checkpoint 与推理接口（至少离线 scoring）
