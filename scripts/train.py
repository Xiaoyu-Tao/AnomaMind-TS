# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

# Load .env before other imports that may use env vars
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import agentlightning as agl
from agentlightning.adapter import TracerTraceToTriplet

from TrainerAgent import TrainerAgent
from llm_config import LLMConfig

import shutil
nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = str(Path(nvcc_path).parent.parent)
    os.environ["CUDA_HOME"] = cuda_home
if "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_" + os.getlogin()

os.environ["WANDB_MODE"] = "offline"


def get_training_config(
    train_dir: str,
    model: str,
    output_dir: str,
    epochs: int,
) -> Dict[str, Any]:
    """Training config with overrides."""
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": True,
            "kl_coeff": 0.0,
            "kl_ctrl": {"type": "fixed", "kl_coef": 0.0},
        },
        "data": {
            "train_segment_dir": train_dir,
            "val_segment_dir": train_dir,
            "train_batch_size": 8,
            "max_prompt_length": 7000,
            "max_response_length": 3000,
            "truncation": "left",
        },
        "actor_rollout_ref": {
            "rollout": {
                "free_cache_engine": True,
                "tensor_model_parallel_size": 1,
                "n": 1,
                "log_prob_micro_batch_size_per_gpu": 1,
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 35000,
                "enable_chunked_prefill": True,
                "sampling_params": {"temperature": 1, "top_p": 0.9, "top_k": 150},
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                    }
                },
            },
            "actor": {
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "grad_clip": 2.0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "clip_ratio_c": 10.0,
                "optim": {"lr": 2e-6, "weight_decay": 0.01},
                "checkpoint": {
                    "save_contents": ["hf_model"],
                    "load_contents": ["hf_model"],
                    "async_save": False,
                },
            },
            "ref": {"log_prob_micro_batch_size_per_gpu": 1},
            "model": {
                "path": model,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 4,
            "val_before_train": False,
            "logger": ["console", "wandb"],
            "project_name": "AgentTraining",
            "experiment_name": "AgentTraining",
            "nnodes": 1,
            "test_freq": 1000,
            "total_epochs": epochs,
            "save_freq": 100,
            "default_local_dir": output_dir,
        },
    }


def load_segment_folders(segment_dir: str) -> List[Dict[str, Any]]:

    base = Path(segment_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Segment directory does not exist: {segment_dir}")

    tasks = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if (p / "segment_data.csv").exists() and (p / "segment_clean.jpg").exists():
            tasks.append({"segment_folder": str(p), "data_id": p.name})
        else:
            print(f"[WARN] Skipping incomplete segment: {p.name}")

    return tasks


def get_aux_llm_config() -> LLMConfig:
    """LLM config for all non-detector nodes (from .env)."""
    return LLMConfig(
        model_name=os.getenv("AUX_LLM_MODEL", "grok-4-1-fast-non-reasoning"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", ""),
        temperature=0.3,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full workflow training (train detect node only)."
    )
    parser.add_argument(
        "--train_data", "-d",
        type=str,
        required=True,
        help="Train data folder (segment dir with segment_data.csv, segment_clean.jpg)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model path (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Total epochs (default: 2)",
    )
    return parser.parse_args()


def train_full_workflow(
    train_data: str,
    model: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "./output",
    epochs: int = 2,
):
    """Train using full workflow (only detect node gets gradient updates)."""
    print("=" * 80)
    print("Starting full workflow training (detect node only)")
    print("=" * 80)

    config = get_training_config(
        train_dir=train_data,
        model=model,
        output_dir=output_dir,
        epochs=epochs,
    )

    train_dir = train_data
    val_dir = train_data

    train_data = load_segment_folders(train_dir)
    val_data = load_segment_folders(val_dir)

    if len(train_data) == 0:
        raise ValueError(f"Train data is empty: {train_dir}")
    if len(val_data) == 0:
        raise ValueError(f"Val data is empty: {val_dir}")

    print(f"Train segments: {len(train_data)}")
    print(f"Val segments: {len(val_data)}")

    aux_config = get_aux_llm_config()

    agent = TrainerAgent(
        val_temperature=0.3,
        max_turns=3,
        localization_config=aux_config,
        locator_config=aux_config,
        evaluator_config=aux_config,
        actor_config=aux_config,
        detector_config=None,
        enable_evaluator=False,
    )

    algorithm = agl.VERL(config)
    adapter = TracerTraceToTriplet(
        agent_match=r"detect",
        _skip_empty_token_spans=True,
    )
    trainer = agl.Trainer(
        n_runners=128,
        algorithm=algorithm,
        adapter=adapter,
        port=42120,
    )

    print("Starting training...")
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)


if __name__ == "__main__":
    args = parse_args()
    train_full_workflow(
        train_data=args.train_data,
        model=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )
