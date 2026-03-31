from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from Actor import Actor
from Detector import Detector
from Evaluator import Evaluator
from knowledge_base import KnowledgeBase
from llm_config import LLMConfig, create_llm
from Localization import Localization
from Locator import Locator
from utils import get_column_names, image_to_base64
from workflowstate import WorkflowState

logger = logging.getLogger(__name__)


class AnomaAgent:

    def __init__(
        self,
        max_turns: int = 5,
        localization_config: LLMConfig = None,
        evaluator_config: LLMConfig = None,
        locator_config: LLMConfig = None,
        actor_config: LLMConfig = None,
        detector_config: LLMConfig = None,
        knowledgeBase: KnowledgeBase = None,
        enable_evaluator: bool = True,
        max_refinements: int = 3,
    ):
        self.max_turns = max_turns
        self.enable_evaluator = enable_evaluator
        self.max_refinements = max_refinements

        if not self.enable_evaluator:
            logger.info("[AnomaAgent] Evaluator disabled, skipping evaluation step, using Detector result directly")

        self.knowledgeBase = knowledgeBase if knowledgeBase else KnowledgeBase()

        self.localization_llm = create_llm(localization_config)
        self.locator_llm = create_llm(locator_config)
        self.evaluator_llm = create_llm(evaluator_config)
        self.actor_llm = create_llm(actor_config)
        self.detector_llm = create_llm(detector_config)

        self.localization_agent = Localization(llm=self.localization_llm, knowledge_base=self.knowledgeBase)
        self.locator = Locator(llm=self.locator_llm, knowledge_base=self.knowledgeBase)
        self.actor = Actor(knowledge_base=self.knowledgeBase, llm=self.actor_llm)
        self.detector = Detector(knowledge_base=self.knowledgeBase, llm=self.detector_llm)
        self.evaluator = Evaluator(llm=self.evaluator_llm)

    def load_images(self, state: WorkflowState) -> WorkflowState:
        return self._load_preprocessed_images(state)

    def localization(self, state: WorkflowState) -> WorkflowState:
        result = self.localization_agent.analyze(state)
        return result

    def locate(self, state: WorkflowState) -> WorkflowState:
        result = self.locator.locate(state)
        return result

    def actor_call(self, state: WorkflowState) -> WorkflowState:
        result = self.actor.tool_call(state)
        return result

    def detect(self, state: WorkflowState) -> WorkflowState:
        result = self.detector.detect(state)
        return result

    def evaluate(self, state: WorkflowState) -> WorkflowState:
        result = self.evaluator.evaluate(state)
        return result

    def finalize(self, state: WorkflowState) -> WorkflowState:
        return self._finalize_output(state)

    def build_workflow(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        workflow.add_node(self.load_images)  # type: ignore
        workflow.add_node(self.localization)  # type: ignore
        workflow.add_node(self.locate)  # type: ignore
        workflow.add_node(self.actor_call)  # type: ignore
        workflow.add_node(self.detect)  # type: ignore
        workflow.add_node(self.finalize)  # type: ignore
        if self.enable_evaluator:
            workflow.add_node(self.evaluate)  # type: ignore

        workflow.set_entry_point("load_images")
        workflow.add_edge("load_images", "localization")
        workflow.add_edge("localization", "locate")
        workflow.add_edge("locate", "actor_call")
        workflow.add_edge("actor_call", "detect")
        if self.enable_evaluator:
            workflow.add_edge("detect", "evaluate")
            workflow.add_conditional_edges(
                "evaluate",
                self.should_continue_analysis,
                {"locate": "locate", "finalize": "finalize"},
            )
        else:
            workflow.add_edge("detect", "finalize")

        workflow.add_edge("finalize", END)
        return workflow.compile()

    def should_continue_analysis(self, state: WorkflowState) -> str:
        if state.get("has_error", False):
            logger.info(f"[Workflow] Error detected, stopping: {state.get('error_message', 'Unknown error')}")
            return "finalize"

        needs_refinement = state.get("needs_refinement", False)
        refinement_count = state.get("refinement_count", 0)

        if needs_refinement and refinement_count < self.max_refinements:
            logger.info(f"[Workflow] Evaluator suggests refinement, returning to Locator (round {refinement_count + 1})")
            check_result = state.get("check_result", {})
            if check_result:
                suggestions = check_result.get("suggestions", [])
                issues = check_result.get("issues", [])
                if suggestions or issues:
                    refinement_feedback = "\n\n## Previous Analysis Feedback:\n"
                    if issues:
                        refinement_feedback += f"Issues found: {', '.join(issues)}\n"
                    if suggestions:
                        refinement_feedback += f"Suggestions: {', '.join(suggestions)}\n"
                    current_plan = state.get("plan", "")
                    state["plan"] = current_plan + refinement_feedback
            return "locate"
        elif needs_refinement:
            logger.info(f"[Workflow] Max refinements ({self.max_refinements}) reached, continuing to finalize")
            return "finalize"

        logger.info("[Workflow] Analysis complete, no refinement needed, continuing to finalize")
        return "finalize"

    def _load_preprocessed_images(self, state: WorkflowState) -> WorkflowState:
        local_path = state.get("local_view_path", "")
        if not local_path:
            raise ValueError("Set local_view_path in initial_state")

        import os

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Image file not found: {local_path}")

        local_base64 = state.get("local_view_base64", "")
        if not local_base64:
            state["local_view_base64"] = image_to_base64(local_path)

        return state

    def _finalize_output(self, state: WorkflowState) -> WorkflowState:
        final_has_error = state.get("has_error", False)

        if final_has_error:
            state["final_output"] = {
                "error": True,
                "error_message": state.get("error_message", "Unknown error"),
                "anomaly_intervals": state.get("detector_anomaly_intervals", []),
                "anomaly_types": state.get("detector_anomaly_types", []),
                "explanations": state.get("explanations", []),
                "confidences": state.get("confidences", []),
                "anomaly_points": self._extract_anomaly_points(state),
            }
        else:
            state["final_output"] = {
                "error": False,
                "anomaly_intervals": state.get("detector_anomaly_intervals", []),
                "anomaly_types": state.get("detector_anomaly_types", []),
                "explanations": state.get("explanations", []),
                "confidences": state.get("confidences", []),
                "anomaly_points": self._extract_anomaly_points(state),
                "check_result": state.get("check_result", {}),
                "refinement_count": state.get("refinement_count", 0),
                "needs_refinement": state.get("needs_refinement", False),
            }

        return state

    def _extract_anomaly_points(self, state: WorkflowState) -> list:
        intervals = state.get("detector_anomaly_intervals", [])
        data = state["data"]
        timestamp_col, value_col = get_column_names(data)

        anomaly_points = []
        for interval in intervals:
            if not isinstance(interval, (list, tuple)) or len(interval) == 0:
                continue
            if len(interval) == 1:
                start = end = interval[0]
            else:
                start, end = interval[0], interval[1]

            if timestamp_col in data.columns:
                start_rows = data[data[timestamp_col] == start]
                end_rows = data[data[timestamp_col] == end]
                if len(start_rows) > 0 and len(end_rows) > 0:
                    start_row = start_rows.index[0]
                    end_row = end_rows.index[0]
                    if start_row > end_row:
                        start_row, end_row = end_row, start_row
                    interval_data = data.iloc[start_row : end_row + 1]
                    anomaly_points.extend(interval_data[value_col].tolist())
            else:
                if 0 <= start < len(data) and 0 <= end < len(data):
                    if start > end:
                        start, end = end, start
                    interval_data = data.iloc[start : end + 1]
                    anomaly_points.extend(interval_data[value_col].tolist())

        return anomaly_points

