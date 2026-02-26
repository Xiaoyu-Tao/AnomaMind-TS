import pandas as pd
from typing import TypedDict, Annotated, Sequence, Optional, Tuple


class WorkflowState(TypedDict):
    """Workflow state"""
    data: pd.DataFrame
    context_data: Optional[pd.DataFrame]  # Context data (first 100 points, for sine fitting)

    # Visualization results (local only)
    local_view_path: str
    local_view_base64: str

    # Localization analysis results
    localization_anomaly_intervals: list  # [(start, end), ...]
    localization_anomaly_types: list  # [str, ...]
    localization_anomaly_reasons: list  # [str, ...]
    visual_description: str  # Description from visual model (<description> tag)

    # Planning results
    plan: str
    current_interval_index: int

    # Detector analysis results
    detector_anomaly_intervals: list
    detector_anomaly_types: list
    explanations: list
    confidences: list
    detector_conversation_history: list  # Full Detector conversation (includes Actor tool calls)
    actor_conversation: list  # Actor tool-call phase conversation (Detector input)
    tool_call_conversation: list  # Alias for actor_conversation

    check_result: dict
    needs_refinement: bool
    refinement_count: int  # Refinement round count, avoids infinite loop

    # Agent prompt/response records (for debugging and evaluation)
    agent_prompts_responses: dict  # {"localization": {"prompt": ..., "response": ...}, ...}
    agent_timings: dict  # Per-Agent execution time

    # Error handling
    has_error: bool
    error_message: str

    # Final output
    final_output: dict
