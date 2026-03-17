<div align="center">
  <h1><img src="assets/image.png" alt="MemCast logo" style="height: 1em; width: auto; vertical-align: -0.15em; margin-right: 0.4em;">AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning</h1>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/AnomaMind-TS/stargazers">
    <img src="https://img.shields.io/github/stars/Xiaoyu-Tao/AnomaMind-TS?style=social" alt="Stars">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/AnomaMind-TS/pulls">
    <img src="https://img.shields.io/badge/PRs-Welcome-green" alt="PRs Welcome">
  </a>
</div>

---

AnomaMind is a novel framework that reformulates **Time Series Anomaly Detection (TSAD)** as an **evidence-driven sequential decision-making process**. It combines a structured coarse-to-fine workflow with tool-augmented reasoning and a hybrid inference mechanism that couples general-purpose LLM reasoning with task-specific anomaly decision learning via reinforcement learning.

> 📝 "AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning"  
> **Preprint** | [📄 Paper]()

---

## 🔍 Overview

Existing TSAD methods often treat anomaly detection as a static discriminative task with fixed feature inputs, lacking adaptive evidence gathering and iterative refinement. AnomaMind introduces an agentic paradigm:

- **Coarse-to-Fine Workflow**: Progressively localizes anomalous intervals (coarse evidence acquisition), then performs adaptive evidence construction, reasoning-based detection, and iterative refinement.
- **Tool-Augmented Reasoning**: Relies on a **Detection Toolkit**—interval localization (visual-aware), feature extraction (statistical and structural), and knowledge memory—for context-aware diagnostic analysis.
- **Hybrid Inference**: General-purpose LLMs handle autonomous tool invocation and self-reflective refinement; core anomaly detection decisions are learned through **reinforcement learning** under workflow-level feedback.

<p align="center">
  <img src="assets/main.png" width="800">
</p>

---

## ✨ Key Features

- **Structured Coarse-to-Fine Workflow**: Coarse evidence acquisition → adaptive evidence construction → reasoning-based anomaly detection → iterative refinement.
- **Detection Toolkit**: Interval localization (VLM-based), statistical and structural feature extraction, and knowledge memory for domain and tool semantics.
- **Hybrid Decision Mechanism**: General-purpose models for tool orchestration and self-reflection; RL-optimized detector for precise anomaly boundaries.
- **Self-Reflective Refinement**: Evaluator validates intermediate results and triggers refinement when needed, reducing false alarms and improving consistency.
- **Strong Benchmark Performance**: Consistently improves over statistical, deep learning, foundation-model, and LLM-based baselines on YAHOO, TODS, IOPS, and WSD.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Xiaoyu-Tao/AnomaMind-TS
cd AnomaMind-TS
```

### 2. Environment Setup

```bash
conda create -n anomalmind python=3.10
conda activate anomalmind
sh install.sh
```

Create `scripts/.env` with your API config:

```
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.example.com/v1
AUX_LLM_MODEL=your_model_name
```

### 3. Prepare Data

AnomaMind is evaluated on four TSAD benchmarks with diverse anomaly types (point, contextual, collective, sequence):
<p align="center">
  <img src="assets/dataset.png" width="800">
</p>

Datasets can be downloaded from [here](https://www.thedatum.org/datasets/TSB-AD-U.zip). 

Extract the CSV files and split them into training and testing sets.

Please organize the files into the following directory structure:

- Training Data: Place CSV files in `dataset/train/raw/`
- Testing Data: Place CSV files in `dataset/test/raw/`


### 4. Run Anomaly Detection

**Step 1: Preprocess Data**

```bash
cd ./scripts
python preprocess.py ../dataset/train/raw ../dataset/train/processed --segment_size 100 --sample_ratio 1.0
python preprocess.py ../dataset/test/raw ../dataset/test/processed --segment_size 100 --sample_ratio 1.0
```

**Step 2: RL Train**

```bash
python train_full_workflow.py --train_data ../dataset/train/processed --model Qwen/Qwen3-8B --output_dir ../model --epochs 2
```

**Step 3: Serve the trained model with vLLM**

```bash
# Replace the path with your checkpoint
vllm serve ./model/PATH/TO/HUGGINGFACE --port 8000 --max-model-len 11000 --gpu-memory-utilization 0.95 --enable-auto-tool-choice --tool-call-parser hermes
```

**Step 4: Inference**

```bash
python infer.py -i ../dataset/test/processed -o ../results -m Qwen/Qwen3-8B -u http://localhost:8000/v1 --enable_checking -w 4
```

**Step 5: Evaluate**

```bash
python eval.py --results_dir ../results --output ../evaluation_results.json
```

## 📊 Benchmark Results

**Main Results (F1 / Best-F1):**

AnomaMind achieves competitive or best average performance across Precision, Recall, F1, and Best-F1 on YAHOO, TODS, IOPS, and WSD, outperforming statistical (FFT-AD, SR), deep learning (CNN, M2N2, LSTMAD, TransAD), foundation (TimesFM, Chronos), and LLM-based (LLMAD, LLM-TSAD, OFA) baselines.

<p align="center">
  <img src="assets/1-main.png" width="800">
</p>

**Ablation Studies:**

- **Detection tools**: Removing interval localization, feature extraction, or knowledge memory each degrades performance; feature extraction has the largest impact.
- **Workflow**: Removing the evaluator (iterative refinement) consistently hurts F1 and Best-F1 across datasets.
- **Reinforcement learning**: Removing RL for anomaly decision learning leads to substantial performance drop; RL is critical for task-specific decision boundaries.
<p align="center">
  <img src="assets/2-tool.png" width="800">
</p>

<p align="center">
  <img src="assets/3-rl.png" width="45%">
  &emsp;
  <img src="assets/4-agent.png" width="45%">
</p>

---

## 📁 Project Structure

```
AnomaMind-TS/
├── README.md
├── install.sh
├── requirements.txt
├── dataset/           # Place benchmark data here
├── assets/            # Figures for README (main.png, ablation plots, etc.)
└── scripts/           # Run scripts for each benchmark
```

## 🤖 Model Zoo

We provide pre-trained detector models for different scales:

| Model | Description | Hugging Face Repo |
|-------|-------------|-------------------|
| **AnomaMind-0.6B** | The model based on Qwen3-8B | [AnomaMind-0.6B](https://huggingface.co/YuChongZ/AnomaMind-0.6B) |
| **AnomaMind-8B** | The model based on Qwen3-8B | [AnomaMind-8B](https://huggingface.co/YuChongZ/AnomaMind-8B) |

You can download and use these models directly:

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{tao2026anomamind,
  title={AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning},
  author={Tao, Xiaoyu and Wu, Yuchong and Cheng, Mingyue and Guo, Ze and Gao, Tian},
  journal={arXiv preprint arXiv:2602.13807},
  year={2026}
}
```
