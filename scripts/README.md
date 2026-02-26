# Anomaly Detector Agent — One-Stop Guide

End-to-end workflow: CSV data → preprocess → train → serve with vLLM → batch evaluate → compute metrics.

---

## Step 1: Prepare CSV Data

Place your time series CSV files in `data/`. Each file must have:

- **`Data`** — numeric time series values
- **`Label`** — 0 (normal) or 1 (anomaly)

Example:

```csv
Data,Label
1.23,0
1.45,0
2.10,1
1.98,1
1.50,0
```

```bash
mkdir -p data
# Copy your CSV files into data/
```

---

## Step 2: Preprocess

Run `preprocess.py` to normalize, segment, and generate visualizations. Output goes to `processed_data/`.

```bash
python preprocess.py data processed_data --segment_size 100 --sample_ratio 1.0
```

| Argument        | Default | Description                          |
|----------------|---------|--------------------------------------|
| `--segment_size`  | 100   | Points per segment                    |
| `--sample_ratio`  | 1.0   | Fraction of segments to process (0–1) |

---

## Step 3: Train

Train the Detector using the full workflow. Uses `processed_data` as input.

```bash
python train_full_workflow.py --train_data processed_data --model Qwen/Qwen3-0.6B --output_dir ./output --epochs 2
```

| Argument      | Default            | Description                |
|---------------|--------------------|----------------------------|
| `--train_data`  | (required)       | Path to processed segment dir |
| `--model`       | Qwen/Qwen3-0.6B  | Base model path            |
| `--output_dir`  | ./output         | Checkpoint output directory |
| `--epochs`      | 2                | Training epochs            |

Configure `LLM_API_KEY`, `LLM_BASE_URL`, `AUX_LLM_MODEL` in `.env` for aux agents (Localization, Locator, Evaluator, Actor).

---

## Step 4: Start vLLM Server

Serve the trained model with vLLM. Replace the model path with your checkpoint (e.g. `./output/.../actor/huggingface`).

```bash
vllm serve ./output/experiment_name/checkpoint_XXX/actor/huggingface \
    --port 8000 \
    --max-model-len 11000 \
    --gpu-memory-utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

---

## Step 5: Batch Inference

Run batch inference on processed data. Input: `processed_data`, output: result directory.

```bash
python batch_process.py --input processed_data --output ./batch_results \
    --model_name Qwen/Qwen3-0.6B \
    --base_url http://localhost:8000/v1 \
    --enable_checking \
    --max_workers 4
```

| Argument         | Default                  | Description                 |
|------------------|--------------------------|-----------------------------|
| `--input` / `-i` | (required)               | Input folder (processed_data) |
| `--output` / `-o`| (required)               | Output results directory    |
| `--model_name`   | Qwen/Qwen3-0.6B          | Detector model name         |
| `--base_url`     | http://localhost:8000/v1 | vLLM API base URL           |
| `--enable_checking` | (flag)               | Enable Evaluator            |
| `--max_workers`  | 1                        | Parallel threads            |

---

## Step 6: Evaluate

Compute Precision, Recall, F1, and BestF1 from batch results.

```bash
python eval.py --results_dir ./batch_results --output ./evaluation_results.json
```

| Argument        | Default                 | Description                |
|-----------------|-------------------------|----------------------------|
| `--results_dir` | (required)              | Batch results directory    |
| `--output`      | ./evaluation_results.json | Output JSON path         |

---

## Quick Reference (Copy-Paste)

```bash
# 1. Preprocess
python preprocess.py data processed_data --segment_size 100 --sample_ratio 1.0

# 2. Train
python train_full_workflow.py --train_data processed_data --model Qwen/Qwen3-0.6B --output_dir ./output --epochs 2

# 3. Serve (replace path with your checkpoint)
vllm serve ./output/.../actor/huggingface --port 8000 --max-model-len 11000 --gpu-memory-utilization 0.95 --enable-auto-tool-choice --tool-call-parser hermes

# 4. Batch inference
python batch_process.py -i processed_data -o ./batch_results -m Qwen/Qwen3-0.6B -u http://localhost:8000/v1 --enable_checking -w 4

# 5. Evaluate
python eval.py --results_dir ./batch_results --output ./evaluation_results.json
```

---

## Environment

Create `.env` in the Agent directory:

```
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.example.com/v1
AUX_LLM_MODEL=grok-4-1-fast-non-reasoning
```

Aux agents (Localization, Locator, Evaluator, Actor) use these; the Detector uses `--model_name` and `--base_url` from the batch_process command.
