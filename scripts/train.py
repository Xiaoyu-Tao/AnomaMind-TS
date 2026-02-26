# -*- coding: utf-8 -*-
"""
Simplified training script: trains only the fine_grained_reasoning module of ActionAgent.
pkill -f "python train.py"
pkill -f AgentLightning-AgentOpsServer
ray stop --force
"""

import argparse
import logging
import os
import sys
import threading
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Load .env before other imports that may use env vars
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import agentlightning as agl
from agentlightning.adapter import TracerTraceToTriplet

from FineGrainedTrainerAgent import FineGrainedLiteAgent

import shutil
nvcc_path = shutil.which("nvcc")

if nvcc_path:
    cuda_home = str(Path(nvcc_path).parent.parent)
    os.environ["CUDA_HOME"] = cuda_home
if "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_" + os.getlogin()

os.environ["WANDB_MODE"] = "offline"


# ==================== Logging ====================
# Global variable to hold log file handle (prevent GC)
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
        """Check if first file (usually raw stdout/stderr) is a terminal."""
        if self.files:
            first_file = self.files[0]
            if hasattr(first_file, 'isatty'):
                return first_file.isatty()
        return False
    
    def fileno(self):
        """Return file descriptor of first file (usually raw stdout/stderr)."""
        if self.files:
            first_file = self.files[0]
            if hasattr(first_file, 'fileno'):
                return first_file.fileno()
        return 2  # stderr descriptor is usually 2

def setup_logging(log_file: Optional[str] = None, enable_logging: bool = True):
    """Configure logging and optionally save all output to log file.

    Args:
        log_file: Log file path; if None, output to console only
        enable_logging: If False, no log file, no output redirection
    """
    global _log_file_handle
    
    # If logging disabled, configure basic console logger only
    if not enable_logging:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        agl_logger = logging.getLogger("agentlightning")
        agl_logger.setLevel(logging.INFO)
        store_client_server_logger = logging.getLogger("agentlightning.store.client_server")
        store_client_server_logger.setLevel(logging.WARNING)
        return root_logger

    # Create logs directory
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        tee_stdout = Tee(original_stdout, _log_file_handle)
        tee_stderr = Tee(original_stderr, _log_file_handle)
        
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        print(f"All output will be saved to: {Path(log_file).absolute()}")
    agl_logger = logging.getLogger("agentlightning")
    agl_logger.setLevel(logging.INFO)
    store_client_server_logger = logging.getLogger("agentlightning.store.client_server")
    store_client_server_logger.setLevel(logging.WARNING)
    return root_logger

logger = logging.getLogger(__name__)

def get_training_config(
    model: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "./output",
    epochs: int = 2,
) -> Dict[str, Any]:
    """Get training config with overrides."""
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": True,
            "kl_coeff": 0.0,
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": 0.0,
            }
        },
        "data": {
            "train_files": None,
            "val_files": None,
            "train_batch_size": 4,
            "max_prompt_length": 6000,
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
                "sampling_params": {
                    "temperature": 1,
                    "top_p": 0.9,
                    "top_k": 150,
                },
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                    }
                },
            },
            "actor": {
                "ppo_mini_batch_size": 4,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "grad_clip": 2.0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "clip_ratio_c": 10.0,
                "optim": {
                    "lr": 2e-6,
                    "weight_decay": 0.01
                },
                "checkpoint": {
                    "save_contents": ["hf_model"],
                    "load_contents": ["hf_model"],
                    "async_save": False,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 2,
            },
            "model": {
                "path": model,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 4,
            "val_before_train": False,
            "logger": ["console", "wandb"],
            "project_name": "VLMTimeSSeriesAgent",
            "experiment_name": "vlm_tsss_training",
            "nnodes": 1,
            "test_freq": 1000,
            "total_epochs": epochs,
            "save_freq": 240,
            "default_local_dir": output_dir,
        },
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train fine_grained_reasoning (Detector) module."
    )
    parser.add_argument(
        "--train_data", "-d",
        type=str,
        required=True,
        help="Train data path (parquet file)",
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


def train_fine_grained(
    train_data: str,
    model: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "./output",
    epochs: int = 2,
    log_file: Optional[str] = None,
    enable_logging: bool = True,
):
    """Train fine_grained_reasoning module."""
    if enable_logging:
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"training_fine_grained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        setup_logging(str(log_file), enable_logging=True)
        logger.info("=" * 80)
        logger.info("Starting Fine-Grained Reasoning training")
        logger.info("=" * 80)
        logger.info(f"Log file: {Path(log_file).absolute()}")
        print(f"All output saved to: {Path(log_file).absolute()}")
    else:
        setup_logging(enable_logging=False)
        logger.info("=" * 80)
        logger.info("Starting Fine-Grained Reasoning training (logging disabled)")
        logger.info("=" * 80)

    config = get_training_config(
        model=model,
        output_dir=output_dir,
        epochs=epochs,
    )
    config["data"]["train_files"] = train_data
    config["data"]["val_files"] = train_data
    agent = FineGrainedLiteAgent(
        val_temperature=0.3,
        detector_config=None,  
        rollout_data_dir="rollout_6B",
        save_rollout_records=False
    )

    algorithm = agl.VERL(config)
    adapter = TracerTraceToTriplet(
        agent_match=r"detect",
        _skip_empty_token_spans=True
    )
    trainer = agl.Trainer(
        n_runners=128,
        algorithm=algorithm,
        adapter=adapter,
        port=42119
    )

    train_file = config["data"]["train_files"]
    val_file = config["data"]["val_files"]

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not Path(val_file).exists():
        raise FileNotFoundError(f"Val file not found: {val_file}")

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    
    logger.info(f"Train file: {train_file}")
    logger.info(f"Val file: {val_file}")
    logger.info(f"Train rows: {len(train_df)}")
    logger.info(f"Val rows: {len(val_df)}")

    if len(train_df) == 0:
        raise ValueError(f"Train file is empty: {train_file}")
    if len(val_df) == 0:
        raise ValueError(f"Val file is empty: {val_file}")

    required_columns = ["prompt", "ground_truth"]
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"Train data missing required columns: {missing_columns}")

    logger.info(f"Train columns: {train_df.columns.tolist()}")
    train_data = train_df.to_dict(orient="records")
    val_data = val_df.to_dict(orient="records")
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")

    logger.info("Starting training...")
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)


if __name__ == "__main__":
    args = parse_args()
    train_fine_grained(
        train_data=args.train_data,
        model=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        log_file=args.log_file,
        enable_logging=not args.no_log,
    )

