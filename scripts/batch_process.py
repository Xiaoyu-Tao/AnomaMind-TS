# -*- coding: utf-8 -*-
"""
Batch process samples using TrainerAgent.
Reads folder, processes samples in parallel, saves results.
"""

import argparse
import json
import os
import shutil
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Load .env before other imports
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from TrainerAgent import TrainerAgent
from utils import load_segment_data
from llm_config import LLMConfig


def find_all_samples(data_folder: str) -> List[str]:
    """Find all sample folders.

    Args:
        data_folder: Data folder path

    Returns:
        List of sample folder paths
    """
    data_path = Path(data_folder)
    samples = []

    for item in data_path.iterdir():
        if item.is_dir():
            csv_file = item / "segment_data.csv"
            segment_img = item / "segment_clean.jpg"

            if csv_file.exists() and segment_img.exists():
                samples.append(str(item))

    samples.sort()
    return samples


def is_sample_already_processed(sample_output_dir: Path) -> bool:
    """Check if sample has been successfully processed."""
    result_file = sample_output_dir / "detection_result.json"
    gt_file = sample_output_dir / "groundtruth.json"

    if result_file.exists() and gt_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                if not result_data.get("error", False):
                    return True
        except Exception:
            pass

    return False


def get_aux_llm_config() -> LLMConfig:
    """LLM config for non-detector nodes (from .env)."""
    return LLMConfig(
        model_name=os.getenv("AUX_LLM_MODEL", "grok-4-1-fast-non-reasoning"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        temperature=0.3,
    )


def process_single_sample(
    segment_folder: str,
    output_dir: str,
    trainer_agent: TrainerAgent,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    skip_if_exists: bool = True
) -> Dict[str, Any]:
    """Process a single sample.

    Args:
        segment_folder: Sample folder path
        output_dir: Output directory
        trainer_agent: TrainerAgent instance
        max_retries: Max retries (default 3)
        retry_delay: Retry delay in seconds (default 1.0)
        skip_if_exists: Skip if already processed (default True)

    Returns:
        Result dict
    """
    sample_name = Path(segment_folder).name
    sample_output_dir = Path(output_dir) / sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and is_sample_already_processed(sample_output_dir):
        print(f"\n{'='*80}")
        print(f"Skipping already processed: {sample_name}")
        print(f"{'='*80}")
        return {
            "sample_name": sample_name,
            "folder_path": segment_folder,
            "success": True,
            "skipped": True,
            "error": None,
            "result": None,
            "timestamp": datetime.now().isoformat()
        }

    print(f"\n{'='*80}")
    print(f"Processing: {sample_name}")
    print(f"{'='*80}")

    result = {
        "sample_name": sample_name,
        "folder_path": segment_folder,
        "success": False,
        "error": None,
        "result": None,
        "retry_count": 0,
        "timestamp": datetime.now().isoformat()
    }

    last_error = None
    try:
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"\n[Retry {attempt}/{max_retries-1}] Waiting {retry_delay}s...")
                import time
                time.sleep(retry_delay)
                print(f"[Retry {attempt}/{max_retries-1}] Retrying: {sample_name}")

            result["retry_count"] = attempt

            try:
                segment_data = load_segment_data(Path(segment_folder))
                ground_truth = segment_data["ground_truth"]

                workflow = trainer_agent.build_workflow()
                initial_state = {
                    "data": segment_data["segment_data"],
                    "context_data": segment_data["context_data"],
                    "local_view_path": segment_data["segment_image_path"],
                    "local_view_base64": segment_data["segment_image_base64"],
                    "localization_anomaly_intervals": [],
                    "localization_anomaly_types": [],
                    "localization_anomaly_reasons": [],
                    "visual_description": "",
                    "plan": "",
                    "current_interval_index": -1,
                    "detector_anomaly_intervals": [],
                    "detector_anomaly_types": [],
                    "explanations": [],
                    "confidences": [],
                    "actor_conversation": [],
                    "check_result": {},
                    "needs_refinement": False,
                    "refinement_count": 0,
                    "agent_prompts_responses": {},
                    "has_error": False,
                    "error_message": "",
                    "final_output": {}
                }

                final_state = workflow.invoke(initial_state)
                detection_result = final_state.get("final_output", {})

                result["success"] = True
                result["result"] = detection_result

                result_file = sample_output_dir / "detection_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(detection_result, f, ensure_ascii=False, indent=2, default=str)

                gt_data = {"ground_truth": ground_truth}
                gt_file = sample_output_dir / "groundtruth.json"
                with open(gt_file, 'w', encoding='utf-8') as f:
                    json.dump(gt_data, f, ensure_ascii=False, indent=2)

                segment_image_path = segment_data["segment_image_path"]
                if segment_image_path and os.path.exists(segment_image_path):
                    shutil.copy2(segment_image_path, sample_output_dir / "segment_clean.jpg")

                csv_source = Path(segment_folder) / "segment_data.csv"
                if csv_source.exists():
                    shutil.copy2(csv_source, sample_output_dir / "segment_data.csv")

                agent_prompts_responses = final_state.get("agent_prompts_responses", {})
                if agent_prompts_responses:
                    prompts_file = sample_output_dir / "agent_prompts_responses.json"
                    with open(prompts_file, 'w', encoding='utf-8') as f:
                        json.dump(agent_prompts_responses, f, ensure_ascii=False, indent=2)

                state_to_save = final_state.copy()
                if "local_view_base64" in state_to_save:
                    state_to_save["local_view_base64"] = "[BASE64_DATA_OMITTED]"
                state_file = sample_output_dir / "final_state.json"
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state_to_save, f, ensure_ascii=False, indent=2, default=str)

                print(f"\nDone: {sample_name}")
                print(f"  Results saved to: {sample_output_dir}")
                break

            except Exception as e:
                error_msg = str(e) if e else "Unknown error"
                error_trace = traceback.format_exc()
                last_error = error_msg
                result["error"] = error_msg
                result["error_trace"] = error_trace

                if attempt < max_retries - 1:
                    print(f"\nFailed {sample_name} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Retrying in {retry_delay}s...")
                else:
                    print(f"\nFailed {sample_name} (after {max_retries} retries): {error_msg}")
                    error_file = sample_output_dir / "error.json"
                    with open(error_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "error": error_msg,
                            "traceback": error_trace,
                            "retry_count": attempt + 1,
                            "max_retries": max_retries,
                            "timestamp": datetime.now().isoformat()
                        }, f, ensure_ascii=False, indent=2)

        if not result["success"]:
            result["error"] = last_error
            result["final_error"] = f"Failed after {max_retries} retries: {last_error}"

    except Exception as e:
        result["error"] = str(e)
        result["error_trace"] = traceback.format_exc()

    return result


def batch_process_with_trainer(
    data_folder: str,
    output_dir: str,
    detector_model_name: str,
    detector_base_url: str,
    enable_checking: bool = False,
    max_workers: int = 1,
    max_samples: Optional[int] = None,
    start_from: Optional[int] = None,
    skip_existing: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_failed_only: bool = False,
    max_turns: int = 5,
) -> Dict[str, Any]:
    """Batch process samples using TrainerAgent.

    Non-detector agents (localization, locator, evaluator, actor) load config from .env.
    Detector uses detector_model_name and detector_base_url from args.

    Args:
        data_folder: Input data folder
        output_dir: Output directory
        detector_model_name: Model name for Detector
        detector_base_url: Base URL for Detector
        enable_checking: Enable Evaluator
        max_workers: Parallel thread count
        max_samples: Max samples to process (None = all)
        start_from: Start index (for resume)
        skip_existing: Skip already processed samples
        max_retries: Max retries per sample
        retry_delay: Retry delay (seconds)
        retry_failed_only: Only retry failed samples
        max_turns: Max workflow turns

    Returns:
        Summary dict
    """
    print("=" * 80)
    print("Batch processing (TrainerAgent)")
    print("=" * 80)

    aux_config = get_aux_llm_config()
    detector_config = LLMConfig(
        model_name=detector_model_name,
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=detector_base_url,
        temperature=0.3,
        max_tokens=5000,
    )

    print(f"\nScanning: {data_folder}")
    all_samples = find_all_samples(data_folder)
    total_samples = len(all_samples)
    print(f"Found {total_samples} samples")

    if retry_failed_only:
        output_path = Path(output_dir)
        failed = [s for s in all_samples if not is_sample_already_processed(output_path / Path(s).name)]
        all_samples = failed
        print(f"Retrying failed only: {len(all_samples)} samples")

    if start_from:
        all_samples = all_samples[start_from:]
        print(f"Starting from index {start_from + 1}")

    if max_samples:
        all_samples = all_samples[:max_samples]
        print(f"Limiting to {max_samples} samples")

    if skip_existing:
        output_path = Path(output_dir)
        samples_to_process = [s for s in all_samples
                             if not is_sample_already_processed(output_path / Path(s).name)]
        skipped = len(all_samples) - len(samples_to_process)
        all_samples = samples_to_process
        if skipped > 0:
            print(f"Skipped already processed: {skipped}")
        print(f"To process: {len(all_samples)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTrainerAgent config:")
    print(f"  Aux (from .env): {aux_config.model_name} @ {aux_config.base_url}")
    print(f"  Detector: {detector_model_name} @ {detector_base_url}")
    print(f"  Enable Evaluator: {enable_checking}")
    print(f"  Max workers: {max_workers}")

    trainer_agent = TrainerAgent(
        max_turns=max_turns,
        localization_config=aux_config,
        locator_config=aux_config,
        evaluator_config=aux_config,
        actor_config=aux_config,
        detector_config=detector_config,
        training_mode=False,
        enable_evaluator=enable_checking,
    )

    config_file = output_path / "batch_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            "data_folder": data_folder,
            "output_dir": str(output_dir),
            "total_samples": total_samples,
            "processed_count": len(all_samples),
            "start_from": start_from,
            "max_samples": max_samples,
            "max_workers": max_workers,
            "enable_checking": enable_checking,
            "detector_model_name": detector_model_name,
            "detector_base_url": detector_base_url,
            "timestamp": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)

    results = []
    success_count = 0
    error_count = 0
    skipped_count = 0
    stats_lock = threading.Lock()
    results_lock = threading.Lock()
    console_lock = threading.Lock()

    def process_with_stats(sample_folder: str, index: int) -> Dict[str, Any]:
        nonlocal success_count, error_count, skipped_count
        thread_name = threading.current_thread().name or f"Thread-{threading.current_thread().ident}"
        sample_name = Path(sample_folder).name

        try:
            result = process_single_sample(
                segment_folder=sample_folder,
                output_dir=output_dir,
                trainer_agent=trainer_agent,
                max_retries=max_retries,
                retry_delay=retry_delay,
                skip_if_exists=skip_existing
            )
        except Exception as e:
            result = {
                "sample_name": sample_name,
                "folder_path": sample_folder,
                "success": False,
                "error": str(e),
                "error_trace": traceback.format_exc(),
                "result": None,
                "retry_count": 0,
                "timestamp": datetime.now().isoformat()
            }

        with stats_lock:
            if result.get("skipped", False):
                skipped_count += 1
            elif result["success"]:
                success_count += 1
            else:
                error_count += 1
            completed = success_count + error_count + skipped_count

        with results_lock:
            results.append(result)

        with console_lock:
            status = 'OK' if result.get('success') else 'FAIL' if not result.get('skipped') else 'SKIP'
            print(f"\n[{thread_name}] [{completed}/{len(all_samples)}] {status} | {sample_name}")
            print(f"  Success: {success_count} | Failed: {error_count} | Skipped: {skipped_count}")

        return result

    if max_workers > 1 and len(all_samples) > 1:
        print(f"\nParallel processing with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(process_with_stats, sf, i + 1): sf
                for i, sf in enumerate(all_samples)
            }
            for future in as_completed(future_to_sample):
                try:
                    future.result()
                except Exception as e:
                    sf = future_to_sample[future]
                    with console_lock:
                        print(f"\nException for {Path(sf).name}: {e}")
                    with stats_lock:
                        error_count += 1
    else:
        print(f"\nSingle-thread processing {len(all_samples)} samples...")
        for i, sf in enumerate(all_samples, 1):
            process_with_stats(sf, i)

    summary = {
        "total_samples": len(all_samples),
        "success_count": success_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "success_rate": success_count / len(all_samples) if all_samples else 0,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
        "max_workers": max_workers,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

    summary_file = output_path / "batch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 80)
    print("Batch processing complete")
    print("=" * 80)
    print(f"Total: {len(all_samples)}")
    print(f"Success: {success_count}")
    print(f"Failed: {error_count}")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count}")
    print(f"Success rate: {success_count / len(all_samples) * 100:.2f}%" if all_samples else "N/A")

    failed_samples = [r["sample_name"] for r in results if not r.get("success") and not r.get("skipped")]
    if failed_samples:
        print(f"\nFailed samples ({len(failed_samples)}):")
        for name in failed_samples[:10]:
            print(f"  - {name}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more")
        print("\nTip: Use --retry_failed_only to retry only failed samples")

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process samples using TrainerAgent."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input data folder (segment dirs)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output results directory",
    )
    parser.add_argument(
        "--enable_checking",
        action="store_true",
        help="Enable Evaluator (checking)",
    )
    parser.add_argument(
        "--max_workers", "-w",
        type=int,
        default=1,
        help="Parallel thread count (default: 1)",
    )
    parser.add_argument(
        "--model_name", "-m",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Detector model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--base_url", "-u",
        type=str,
        default="http://localhost:8000/v1",
        help="Detector API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to process (default: all)",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=None,
        help="Start from index (for resume)",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Do not skip already processed samples",
    )
    parser.add_argument(
        "--retry_failed_only",
        action="store_true",
        help="Only retry previously failed samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_process_with_trainer(
        data_folder=args.input,
        output_dir=args.output,
        detector_model_name=args.model_name,
        detector_base_url=args.base_url,
        enable_checking=args.enable_checking,
        max_workers=args.max_workers,
        max_samples=args.max_samples,
        start_from=args.start_from,
        skip_existing=not args.no_skip,
        retry_failed_only=args.retry_failed_only,
    )
