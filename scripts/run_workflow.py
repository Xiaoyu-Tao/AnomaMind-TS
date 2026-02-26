# -*- coding: utf-8 -*-
"""
Example script to run the anomaly detection workflow.
"""

import pandas as pd
import os
from llm_config import LLMConfig
from typing import Dict, Any, Optional
import json
from utils import image_to_base64,get_column_names
from TrainerAgent import TrainerAgent
def load_preprocessed_data(segment_folder: str) -> tuple:
    """Load data from preprocessed folder (base64 image, context data).

    Args:
        segment_folder: Path (contains segment_data.csv, segment_clean.jpg, context_data.csv)

    Returns:
        (df, timestamp_col, value_col, segment_image_base64, segment_image_path, context_df)
    """
    from pathlib import Path

    folder = Path(segment_folder)
    csv_path = folder / "segment_data.csv"
    segment_image_path = folder / "segment_clean.jpg"
    context_csv_path = folder / "context_data.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"segment_data.csv not found in {folder}")
    if not segment_image_path.exists():
        raise FileNotFoundError(f"segment_clean.jpg not found in {folder}")

    df = pd.read_csv(csv_path)
    context_df = None
    if context_csv_path.exists():
        context_df = pd.read_csv(context_csv_path)

    timestamp_col = None
    value_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'timestamp' in col.lower() or 'date' in col.lower() or col.lower() == 'index':
            timestamp_col = col
            break

    for col in df.columns:
        if col.lower() in ['value', 'data', 'val', 'values']:
            value_col = col
            break

    if value_col is None and len(df.columns) >= 2:
        value_col = df.columns[1] if df.columns[0] == timestamp_col else df.columns[0]

    if timestamp_col is None:
        df['Index'] = df.index
        timestamp_col = 'Index'

    segment_image_base64 = image_to_base64(str(segment_image_path))
    return (df, timestamp_col, value_col,
            segment_image_base64,
            str(segment_image_path), context_df)


def run_anomaly_detection(
    segment_folder: str,
    task_prompt: str = "Detect anomalies in time series",
    attribute_semantics: str = "Time series data with timestamp and value",
    llm_config: Optional[LLMConfig] = None,
    vlm_config: Optional[LLMConfig] = None
) -> Dict[str, Any]:
    """
    Run anomaly detection workflow (using preprocessed files).

    Args:
        segment_folder: Preprocessed folder (segment_data.csv, segment_clean.jpg)
        task_prompt: Task prompt
        attribute_semantics: Attribute semantics
        llm_config: LLM config
        vlm_config: VLM config

    Returns:
        Final detection result
    """
    print("=" * 60)
    print("Starting anomaly detection workflow (preprocessed)")
    print("=" * 60)

    print(f"\n[1] Loading preprocessed data: {segment_folder}")
    (df, _, _,
     segment_image_base64,
     segment_image_path, context_df) = load_preprocessed_data(segment_folder)

    if context_df is not None:
        print(f"  Context data: {len(context_df)} points")
    else:
        print(f"  Context data: not found (optional)")

    timestamp_col, value_col = get_column_names(df)
    print(f"Data shape: {df.shape}")
    print(f"Timestamp col: {timestamp_col}")
    print(f"Value col: {value_col}")
    print(f"Segment image: {segment_image_path}")
    print(f"Image loaded and converted to base64")
    
    initial_state = {
        "data": df,
        "context_data": context_df,
        "local_view_path": segment_image_path,
        "local_view_base64": segment_image_base64,
        "localization_anomaly_intervals": [],
        "localization_anomaly_types": [],
        "localization_anomaly_reasons": [],
        "visual_description": "",
        "plan": "",
        "current_interval_index": -1,
        "fine_anomaly_intervals": [],
        "fine_anomaly_types": [],
        "explanations": [],
        "confidences": [],
        "fine_grained_conversation_history": [],
        "tool_call_conversation": [],
        "check_result": {},
        "needs_refinement": False,
        "refinement_count": 0,
        "agent_prompts_responses": {},
        "agent_timings": {},
        "has_error": False,
        "error_message": "",
        "final_output": {}
    }

    print("\n[2] Building workflow...")
    plan_config = LLMConfig(
        model_name="grok-4-1-fast-non-reasoning",
        api_key="sk-WRkycVK2fsKY2z4zZlaDslWY4ZG37XLIlWg07i5LMfGy8f4b",
        base_url="https://api2.aigcbest.top/v1",
        temperature=0.3
    )
    
    vlm_config = LLMConfig(
        model_name="gemini-2.5-flash-lite-preview-09-2025",
        api_key="sk-WRkycVK2fsKY2z4zZlaDslWY4ZG37XLIlWg07i5LMfGy8f4b",
        base_url="https://api2.aigcbest.top/v1",
        temperature=0.3
    )

    refinement_config = LLMConfig(
        model_name="gemini-2.5-flash-lite-nothinking",
        api_key="sk-WRkycVK2fsKY2z4zZlaDslWY4ZG37XLIlWg07i5LMfGy8f4b",
        base_url="https://api2.aigcbest.top/v1",
        temperature=0.3
    )

    action_config = LLMConfig(
        model_name="gemini-2.5-flash-lite-nothinking",
        api_key="123",
        base_url="http://localhost:8001/v1",
        temperature=0.3
    )
    from knowledge_base import KnowledgeBase

    workflow = TrainerAgent(
        localization_config=plan_config,
        evaluator_config=plan_config,
        locator_config=plan_config,
        actor_config=plan_config,
        detector_config=plan_config,
        knowledgeBase=KnowledgeBase(),
        training_mode=True,
        enable_evaluator=False
    ).build_workflow()

    print("\n[3] Executing workflow...")
    print("-" * 60)
    final_state = workflow.invoke(initial_state)

    print("\n" + "=" * 60)
    print("Workflow execution complete")
    print("=" * 60)

    final_output = final_state.get("final_output", {}) if final_state else {}
    print("\nDetection results:")
    print(f"  Anomaly intervals: {len(final_output.get('anomaly_intervals', []))}")

    for i, (interval, anomaly_type, explanation, confidence) in enumerate(zip(
        final_output.get('anomaly_intervals', []),
        final_output.get('anomaly_types', []),
        final_output.get('explanations', []),
        final_output.get('confidences', [])
    )):
        print(f"\n  Anomaly #{i+1}:")
        print(f"    Interval: {interval}")
        print(f"    Type: {anomaly_type}")
        print(f"    Confidence: {confidence:.2f}")
        print(f"    Explanation: {explanation[:100]}..." if len(explanation) > 100 else f"    Explanation: {explanation}")

    return final_output


if __name__ == "__main__":
    segment_folder = "./TrainData/WSD/068_WSD_id_40_WebService_tr_4549_1st_13322_seg165"

    if not os.path.exists(segment_folder):
        print(f"Error: folder does not exist: {segment_folder}")
        print("Set segment_folder to a valid preprocessed folder")
        print("Preprocessed folder should contain: segment_data.csv, segment_clean.jpg")
        exit(1)

    result = run_anomaly_detection(segment_folder=segment_folder)
    output_file = "detection_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")

