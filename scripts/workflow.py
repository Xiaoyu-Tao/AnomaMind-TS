from knowledge_base import KnowledgeBase
from llm_config import LLMConfig, create_llm
from typing import Optional
from langgraph.graph import StateGraph, END
from utils import get_column_names,image_to_base64
from workflowstate import WorkflowState
from Localization import Localization
from Actor import Actor
from Detector import Detector
from Locator import Locator
from Evaluator import Evaluator


def should_continue_analysis(state: WorkflowState) -> str:

    if state.get("has_error", False):
        print(f"\n[Workflow] Error detected, stopping: {state.get('error_message', 'Unknown error')}")
        return "finalize"

    # Check if refinement needed
    needs_refinement = state.get("needs_refinement", False)
    refinement_count = state.get("refinement_count", 0)
    max_refinements = 3

    print(f"\n[Workflow] ========== should_continue_analysis ==========")
    print(f"[Workflow] needs_refinement: {needs_refinement}")
    print(f"[Workflow] refinement_count: {refinement_count}")
    print(f"[Workflow] max_refinements: {max_refinements}")
    print(f"[Workflow] check_result: {state.get('check_result', {})}")
    
    if needs_refinement and refinement_count < max_refinements:
        print(f"\n[Workflow] Evaluator suggests refinement, returning to Locator (round {refinement_count + 1})")
        check_result = state.get("check_result", {})
        print(f"[Workflow] check_result: {check_result}")

        if check_result:
            # Add feedback to plan for Locator
            suggestions = check_result.get("suggestions", [])
            issues = check_result.get("issues", [])
            if suggestions or issues:
                refinement_feedback = "\n\n## Previous Analysis Feedback:\n"
                if issues:
                    refinement_feedback += f"Issues found: {', '.join(issues)}\n"
                if suggestions:
                    refinement_feedback += f"Suggestions: {', '.join(suggestions)}\n"
                # Add feedback for Locator
                current_plan = state.get("plan", "")
                state["plan"] = current_plan + refinement_feedback
        print(f"[Workflow] ============================================\n")
        return "locate"
    elif needs_refinement:
        print(f"\n[Workflow] Max refinements ({max_refinements}) reached, continuing to finalize")
        return "finalize"

    print(f"\n[Workflow] Analysis complete, no refinement needed, continuing to finalize")
    return "finalize"


def build_workflow(
    locator_config: Optional[LLMConfig] = None,
    plan_config: Optional[LLMConfig] = None,
    vlm_config: Optional[LLMConfig] = None,
    evaluator_config: Optional[LLMConfig] = None,
    check_config: Optional[LLMConfig] = None,
    actor_config: Optional[LLMConfig] = None,
    detector_config: Optional[LLMConfig] = None,
    action_config: Optional[LLMConfig] = None,
) -> StateGraph:
    knowledge_base = KnowledgeBase()
    _actor_cfg = actor_config or action_config
    _detector_cfg = detector_config or action_config
    _evaluator_cfg = evaluator_config or check_config
    _locator_cfg = locator_config or plan_config

    localization_agent = Localization(llm=create_llm(vlm_config) if vlm_config else None, knowledge_base=knowledge_base)
    locator = Locator(llm=create_llm(_locator_cfg) if _locator_cfg else None, knowledge_base=knowledge_base)
    actor = Actor(knowledge_base=knowledge_base, llm=create_llm(_actor_cfg) if _actor_cfg else None)
    detector = Detector(knowledge_base=knowledge_base, llm=create_llm(_detector_cfg) if _detector_cfg else None)
    evaluator = Evaluator(llm=create_llm(_evaluator_cfg) if _evaluator_cfg else None)

    workflow = StateGraph(WorkflowState)
    workflow.add_node("load_images", _load_preprocessed_images)
    workflow.add_node("localization", localization_agent.analyze)
    workflow.add_node("locate", locator.locate)
    workflow.add_node("actor_call", actor.tool_call)
    workflow.add_node("detect", detector.detect)
    workflow.add_node("evaluate", evaluator.evaluate)
    workflow.add_node("finalize", _finalize_output)

    workflow.set_entry_point("load_images")
    workflow.add_edge("load_images", "localization")
    workflow.add_edge("localization", "locate")
    workflow.add_edge("locate", "actor_call")
    workflow.add_edge("actor_call", "detect")
    workflow.add_edge("detect", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        should_continue_analysis,
        {
            "locate": "locate",
            "finalize": "finalize"
        }
    )
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


def _load_preprocessed_images(state: WorkflowState) -> WorkflowState:
    """Verify local image path exists, load local_view_base64 if needed."""
    print("\n[Load images] Checking local image path...")

    local_path = state.get("local_view_path", "")
    if not local_path:
        raise ValueError("local_view_path must be set in initial_state")

    import os
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Image file not found: {local_path}")

    local_base64 = state.get("local_view_base64", "")
    if not local_base64:
        print(f"[Load images] Loading image from path: {local_path}")
        state["local_view_base64"] = image_to_base64(local_path)
    else:
        print("[Load images] local_view_base64 ready")
    
    return state


def _finalize_output(state: WorkflowState) -> WorkflowState:
    """Generate final output."""
    print("\n[Finalize] Organizing results...")

    # Check for errors
    if state.get("has_error", False):
        state["final_output"] = {
            "error": True,
            "error_message": state.get("error_message", "Unknown error"),
            "anomaly_intervals": state.get("detector_anomaly_intervals", []) or state.get("fine_anomaly_intervals", []),
            "anomaly_types": state.get("detector_anomaly_types", []) or state.get("fine_anomaly_types", []),
            "explanations": state.get("explanations", []),
            "confidences": state.get("confidences", []),
            "anomaly_points": _extract_anomaly_points(state)
        }
        print(f"[Finalize] Workflow terminated with error: {state.get('error_message', 'Unknown error')}")
    else:
        state["final_output"] = {
            "error": False,
            "anomaly_intervals": state.get("detector_anomaly_intervals", []) or state.get("fine_anomaly_intervals", []),
            "anomaly_types": state.get("detector_anomaly_types", []) or state.get("fine_anomaly_types", []),
            "explanations": state.get("explanations", []),
            "confidences": state.get("confidences", []),
            "anomaly_points": _extract_anomaly_points(state),
            "check_result": state.get("check_result", {}),
            "refinement_count": state.get("refinement_count", 0),
            "needs_refinement": state.get("needs_refinement", False)
        }
    
    return state


def _extract_anomaly_points(state: WorkflowState) -> list:
    """Extract anomaly point values."""
    intervals = state.get("detector_anomaly_intervals", []) or state.get("fine_anomaly_intervals", [])
    data = state["data"]
    timestamp_col, value_col = get_column_names(data)
    
    anomaly_points = []
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) < 1:
            continue
        start = interval[0]
        end = interval[1] if len(interval) >= 2 else start
        # start/end in intervals are CSV Index values, map to DataFrame row indices
        if timestamp_col in data.columns:
            # Find rows matching Index values
            start_rows = data[data[timestamp_col] == start]
            end_rows = data[data[timestamp_col] == end]
            if len(start_rows) > 0 and len(end_rows) > 0:
                start_row = start_rows.index[0]
                end_row = end_rows.index[0]
                # Ensure start_row <= end_row
                if start_row > end_row:
                    start_row, end_row = end_row, start_row
                interval_data = data.iloc[start_row:end_row+1]
                points = interval_data[value_col].tolist()
                anomaly_points.extend(points)
        else:
            # Fallback: if no Index column, assume start/end are row indices
            if 0 <= start < len(data) and 0 <= end < len(data):
                # Ensure start <= end
                if start > end:
                    start, end = end, start
                interval_data = data.iloc[start:end+1]
                points = interval_data[value_col].tolist()
                anomaly_points.extend(points)
    
    return anomaly_points

