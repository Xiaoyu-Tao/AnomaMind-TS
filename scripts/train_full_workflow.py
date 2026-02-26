# -*- coding: utf-8 -*-
"""
Full workflow training script: uses TrainerAgent.LiteAgent, trains only the fine_grained_reasoning node.

- Data format: segment_folder (preprocessed dir with segment_data.csv, segment_clean.jpg, ground_truth.csv)
- Workflow: load_images -> localization -> locate -> actor -> detect -> evaluate -> finalize
- Training target: detect node only (reward=0 when upstream nodes fail)

pkill -f "python train_full_workflow.py"
ray stop --force
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Load .env before other imports that may use env vars
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import agentlightning as agl
from agentlightning.adapter import TracerTraceToTriplet

from TrainerAgent import LiteAgent
from llm_config import LLMConfig

import shutil
nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = str(Path(nvcc_path).parent.parent)
    os.environ["CUDA_HOME"] = cuda_home
if "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_" + os.getlogin()

os.environ["WANDB_MODE"] = "offline"


# ==================== Logging ====================
_log_file_handle = None


class Tee:
    """Output to both console and file (thread-safe)."""
    def __init__(self, *files):
        self.files = files
        self.lock = threading.Lock()

    def write(self, obj):
        with self.lock:
            for f in self.files:
                try:
                    if hasattr(f, 'closed') and f.closed:
                        continue
                    f.write(obj)
                    f.flush()
                except (ValueError, OSError, AttributeError):
                    pass

    def flush(self):
        with self.lock:
            for f in self.files:
                try:
                    if hasattr(f, 'closed') and f.closed:
                        continue
                    f.flush()
                except (ValueError, OSError, AttributeError):
                    pass

    def isatty(self):
        if self.files and hasattr(self.files[0], 'isatty'):
            return self.files[0].isatty()
        return False

    def fileno(self):
        if self.files and hasattr(self.files[0], 'fileno'):
            return self.files[0].fileno()
        return 2


def setup_logging(log_file: Optional[str] = None, enable_logging: bool = True):
    global _log_file_handle
    if not enable_logging:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(console_handler)
        store_client_server_logger = logging.getLogger("agentlightning.store.client_server")
        store_client_server_logger.setLevel(logging.WARNING)
        return root_logger

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(original_stdout, _log_file_handle)
        sys.stderr = Tee(original_stderr, _log_file_handle)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(console_handler)
    if log_file:
        print(f"All output will be saved to: {Path(log_file).absolute()}")

    # Reduce HTTP access log noise (same as train.py)
    store_client_server_logger = logging.getLogger("agentlightning.store.client_server")
    store_client_server_logger.setLevel(logging.WARNING)

    return root_logger


logger = logging.getLogger(__name__)


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
            logger.warning(f"Skipping incomplete segment: {p.name}")

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
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path (default: auto-generated)",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable log file, console only",
    )
    return parser.parse_args()


def train_full_workflow(
    train_data: str,
    model: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "./output",
    epochs: int = 2,
    log_file: Optional[str] = None,
    enable_logging: bool = True,
):
    """Train using full workflow (only detect node gets gradient updates)."""
    if enable_logging:
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"training_full_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(str(log_file), enable_logging=True)
        logger.info("=" * 80)
        logger.info("Starting full workflow training (detect node only)")
        logger.info("=" * 80)
    else:
        setup_logging(enable_logging=False)

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

    logger.info(f"Train segments: {len(train_data)}")
    logger.info(f"Val segments: {len(val_data)}")

    aux_config = get_aux_llm_config()

    agent = LiteAgent(
        val_temperature=0.3,
        max_turns=3,
        localization_config=aux_config,
        locator_config=aux_config,
        evaluator_config=aux_config,
        actor_config=aux_config,
        detector_config=None,
        rollout_data_dir="rollout_full_workflow",
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

    logger.info("Starting training...")
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)


if __name__ == "__main__":
    args = parse_args()
    train_full_workflow(
        train_data=args.train_data,
        model=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        log_file=args.log_file,
        enable_logging=not args.no_log,
    )
