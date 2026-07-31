# READER 🔍

**Dynamic LLM Provenance from Query-Varying Interactions**

[English](README.md) | [简体中文](README_zh-CN.md)

[📄 预印本](https://arxiv.org/abs/2606.10794) · [🌐 READER Atlas](https://leojeshua.github.io/READER-Atlas/) · [📊 实验结果](results/) · [🧪 复现指南](docs/REPRODUCING.md)

READER 研究如何从 query-varying 的黑盒交互中识别已注册的来源大模型。冻结的代理 LLM 读取每个 prompt-response 对，将 response-token 激活轨迹编码为紧凑的频域指纹，再由轻量级线性探针产生来源证据。Bayesian Evidence Accumulation 可以在不同查询预算下持续累积多次交互的判断。

<p align="center">
  <a href="https://leojeshua.github.io/READER-Atlas/">
    <img src="results/paper_figures/app.png" width="92%" alt="READER Atlas：交互式模型指纹可视化">
  </a>
</p>

## ✨ 核心特点

- **动态溯源：**直接处理自然变化的查询，无需在推理时使用固定诊断问题集。
- **时序指纹：**长度归一化 DCT-II 从完整 response-token 轨迹中提取互补的 DC 与 AC 证据。
- **轻量注册：**代理模型全程冻结，只训练面向候选模型集合的线性来源探针。
- **弹性证据预算：**同一个探针适用于任意可用的查询数量 `K`。
- **统一模型表征：**同一频域指纹还支持静态关系分析、模型生态几何与跨域诊断。

## 🧠 方法概览

1. **读取交互：**冻结的代理模型处理 prompt 和 response，并在选定层提取隐藏状态。
2. **构造指纹：**正交 DCT-II 使用 DC–AC 表征概括完整的 response-token 轨迹。
3. **解码证据：**fold-local 线性探针将单条指纹映射为候选来源证据。
4. **累积判断：**Bayesian Evidence Accumulation 随查询预算增加聚合多条响应。

标准管线使用全部响应 token 和 full-rank DC–AC 指纹，不采用 PCA、GRP 或 SVD。

## 📦 发布内容

| 组成 | 规模与协议 |
|---|---|
| **Agent500** | 每个来源对应 500 条异构 Agent prompts 的动态溯源基准 |
| **来源集合** | 相互嵌套的 50-way、100-way 与 165-way 候选生态 |
| **查询预算** | `K ∈ {1, 5, 10, 20, 50, 100}` |
| **压力测试** | 受控响应长度与无重训 Math100 迁移 |
| **Bench-A** | pair、model-disjoint 与 family-disjoint 静态关系协议 |
| **公开数据** | 139,200 条 prompt-response 记录及校验和与 roster 元数据 |
| **结果产物** | 紧凑实验报告、表格、论文图和回归测试值 |

仓库还提供适用于 OpenAI-compatible API 的 MMLU-Pro 评测器。公开文件不包含密钥、私有端点、模型权重、集群路径或调度命令。

## 🚀 快速开始

环境要求为 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[plots,dev]'
```

复现 DNA baseline 时安装额外的 sentence-encoder 依赖：

```bash
pip install -e '.[baselines,plots,dev]'
```

代理特征提取需要 CUDA 设备以及对应 Hugging Face checkpoint 的访问权限。指纹提取完成后，评测阶段可使用 CPU 或 CUDA。

### 校验完整发布

```bash
reader-data --data-root data validate --full
PYTHONPATH=src python tools/validate_release.py --full-data
pytest
```

完整校验覆盖 139,200 条记录、嵌套 roster、结果校验和、论文关键数值、软链接和公开边界。

## 🧪 复现实验

### Agent500 动态溯源

使用 Qwen3.5-9B 代理运行标准 100-way 实验：

```bash
python workflows/agent500.py \
  --proxy-tag qwen35_9b \
  --variant 100-way \
  --stage all \
  --device cuda \
  --early-exit
```

输出位于 `outputs/agent500/100-way/qwen35_9b/`，包括指纹、fold-local 探针、out-of-fold 来源证据和完整查询预算报告。已有指纹时可使用 `--stage evaluate --device cpu` 跳过代理模型前向。该工作流同样支持 `50-way` 和 `165-way`。

### 无重训压力测试

```bash
python workflows/stress_tests.py \
  --proxy-tag qwen35_9b \
  --variant 100-way \
  --condition all \
  --stage all \
  --device cuda \
  --early-exit
```

### 静态关系评测

```bash
python workflows/bench_a.py \
  --proxy-tag qwen35_9b \
  --stage all \
  --device cuda \
  --early-exit
```

[`configs/proxies.yaml`](configs/proxies.yaml) 中的 `layer` 与 `bench_a_layer` 分别记录任务侧选择的读取层。两类任务共享相同的 DC–AC 指纹构造。

## 📈 分析与论文产物

```bash
python workflows/input_ablation.py --proxy-tag qwen35_9b --stage all
python workflows/layer_scan.py --proxy-tag qwen35_9b --stage all

MPLCONFIGDIR=.cache/matplotlib reader-report \
  --results results \
  --proxy-config configs/proxies.yaml \
  --capabilities configs/capabilities.yaml \
  --output-dir outputs/paper
```

`reader-report` 可重新生成归因、压力测试、分量、层、方差与能力相关性图表。`reader-confusion`、`reader-geometry` 和 `reader-statistics` 分别提供 out-of-fold 混淆分析、模型几何以及 bootstrap、sign-flip 和 Holm 校正统计。

完整协议和命令见 [`docs/PROTOCOL.md`](docs/PROTOCOL.md) 与 [`docs/REPRODUCING.md`](docs/REPRODUCING.md)，已发布论文产物由 [`results/paper_map.json`](results/paper_map.json) 索引。

## 🗂️ 仓库结构

```text
configs/                  模型 roster 与实验配置
data/                     prompts、responses、splits 和 checksums
results/                  紧凑实验报告与论文图
src/reader_provenance/    数值核心与实验入口
workflows/                端到端实验编排
tests/                    单元、集成与论文数值测试
tools/                    发布校验工具
mmlu_pro_api/             OpenAI-compatible MMLU-Pro 评测
```

独立的 [READER Atlas](https://leojeshua.github.io/READER-Atlas/) 支持交互式查看不同代理、方法与领域下的模型指纹几何。

## 📄 数据与许可

代码采用 [MIT License](LICENSE)。Prompt 和 response 数据用于研究复现，并继续受对应模型及服务提供方条款约束。数据格式、数量、校验和与保留的 Bench-A 空响应样本见 [`data/README.md`](data/README.md)。

## 📝 引用

```bibtex
@misc{liu2026reader,
  title         = {READER: Dynamic LLM Provenance from Query-Varying Interactions},
  author        = {Liu, Jiaxu and Mu, Sunnan and Huang, Dong and Wang, Liuyin and Shao, Jing and Zhang, Jie},
  year          = {2026},
  eprint        = {2606.10794},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2606.10794}
}
```
