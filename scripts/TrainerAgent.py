from __future__ import annotations
import logging
import time
import threading
import queue
import sys
from typing import Any, Dict, List, Optional, cast
from knowledge_base import KnowledgeBase
from llm_config import LLMConfig, create_llm
from langgraph.graph import StateGraph, END
from pathlib import Path
from utils import get_column_names, image_to_base64, load_segment_data, save_rollout_records
from workflowstate import WorkflowState
from Localization import Localization
from Actor import Actor
from Detector import Detector
from Locator import Locator
from Evaluator import Evaluator
from reward import reward
import agentlightning as agl

logger = logging.getLogger(__name__)


class TrainerAgent:
    def __init__(
        self,
        max_turns: int = 5,
        localization_config: LLMConfig = None,
        evaluator_config: LLMConfig = None,
        locator_config: LLMConfig = None,
        actor_config: LLMConfig = None,
        detector_config: LLMConfig = None,
        knowledgeBase: KnowledgeBase = None,
        training_mode: bool = False,
        enable_evaluator: bool = True,
        enable_checking: bool = None,
        max_refinements: int = 3,
    ):
        self.max_turns = max_turns
        self.training_mode = training_mode
        _enable_eval = enable_evaluator if enable_checking is None else enable_checking
        self.enable_evaluator = _enable_eval
        self.max_refinements = max_refinements
        if not self.enable_evaluator:
            logger.info("[TrainerAgent] Evaluator disabled, skipping evaluation step, using Detector result directly")
        _actor_cfg = actor_config
        _detector_cfg = detector_config
        _evaluator_cfg = evaluator_config
        _locator_cfg = locator_config
        self.localization_llm = create_llm(localization_config)
        self.locator_llm = create_llm(_locator_cfg)
        self.evaluator_llm = create_llm(_evaluator_cfg)
        self.actor_llm = create_llm(_actor_cfg)
        self.detector_llm = create_llm(_detector_cfg)
        self.knowledgeBase = knowledgeBase if knowledgeBase else KnowledgeBase()
        self.localization_agent = Localization(llm=self.localization_llm, knowledge_base=self.knowledgeBase)
        self.locator = Locator(llm=self.locator_llm, knowledge_base=self.knowledgeBase)
        self.actor = Actor(knowledge_base=self.knowledgeBase, llm=self.actor_llm)
        self.detector = Detector(knowledge_base=self.knowledgeBase, llm=self.detector_llm)
        self.evaluator = Evaluator(llm=self.evaluator_llm)
    
    def load_images(self, state: WorkflowState) -> WorkflowState:
        """Load preprocessed images node."""
        return self._load_preprocessed_images(state)
    
    def localization(self, state: WorkflowState) -> WorkflowState:
        """Localization node."""
        start_time = time.time()
        result = self.localization_agent.analyze(state)
        end_time = time.time()
        execution_time = end_time - start_time
        if "agent_timings" not in result:
            result["agent_timings"] = {}
        result["agent_timings"]["localization"] = {
            "start_time": start_time,
            "end_time": end_time,
            "execution_time_seconds": execution_time
        }
        return result
    
    def locate(self, state: WorkflowState) -> WorkflowState:
        """Locator planning node."""
        start_time = time.time()
        result = self.locator.locate(state)
        end_time = time.time()
        execution_time = end_time - start_time
        if "agent_timings" not in result:
            result["agent_timings"] = {}
        result["agent_timings"]["locate"] = {
            "start_time": start_time,
            "end_time": end_time,
            "execution_time_seconds": execution_time
        }
        return result
    
    def actor_call(self, state: WorkflowState) -> WorkflowState:
        """Actor tool call node."""
        start_time = time.time()
        result = self.actor.tool_call(state)
        end_time = time.time()
        execution_time = end_time - start_time
        
        if "agent_timings" not in result:
            result["agent_timings"] = {}
        result["agent_timings"]["actor"] = {
            "start_time": start_time,
            "end_time": end_time,
            "execution_time_seconds": execution_time
        }
        return result

    def detect(self, state: WorkflowState) -> WorkflowState:
        """Detector node (training target)."""
        start_time = time.time()
        result = self.detector.detect(state)
        end_time = time.time()
        execution_time = end_time - start_time

        if "agent_timings" not in result:
            result["agent_timings"] = {}
        result["agent_timings"]["detect"] = {
            "start_time": start_time,
            "end_time": end_time,
            "execution_time_seconds": execution_time
        }
        return result
    
    def evaluate(self, state: WorkflowState) -> WorkflowState:
        """Evaluator node."""
        start_time = time.time()
        result = self.evaluator.evaluate(state)
        end_time = time.time()
        execution_time = end_time - start_time
        if "agent_timings" not in result:
            result["agent_timings"] = {}
        result["agent_timings"]["evaluate"] = {
            "start_time": start_time,
            "end_time": end_time,
            "execution_time_seconds": execution_time
        }
        return result
    
    def finalize(self, state: WorkflowState) -> WorkflowState:
        """Final output node."""
        return self._finalize_output(state)
    
    def build_workflow(self) -> StateGraph:
        """Build workflow graph."""
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
                {
                    "locate": "locate",
                    "finalize": "finalize"
                }
            )
        else:
            workflow.add_edge("detect", "finalize")
            logger.debug("[TrainerAgent] Workflow configured: skipping Evaluator, detect goes to finalize")
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def should_continue_analysis(self, state: WorkflowState) -> str:
        """Decide whether to continue analysis (conditional routing).

        If refinement needed, return to Locator for re-planning.
        """
        if state.get("has_error", False):
            logger.info(f"[Workflow] Error detected, stopping: {state.get('error_message', 'Unknown error')}")
            return "finalize"

        needs_refinement = state.get("needs_refinement", False)
        refinement_count = state.get("refinement_count", 0)
        logger.debug(
            f"[Workflow] should_continue_analysis: needs_refinement={needs_refinement}, "
            f"refinement_count={refinement_count}, max_refinements={self.max_refinements}"
        )

        if needs_refinement and refinement_count < self.max_refinements:
            logger.info(f"[Workflow] Evaluator suggests refinement, returning to Locator (round {refinement_count + 1})")
            check_result = state.get("check_result", {})
            logger.debug(f"[Workflow] check_result: {check_result}")

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
        """Verify local image path exists, load local_view_base64 if needed."""
        logger.debug("[Load images] Checking local image path...")

        local_path = state.get("local_view_path", "")
        if not local_path:
            raise ValueError("Set local_view_path in initial_state")

        import os
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Image file not found: {local_path}")

        local_base64 = state.get("local_view_base64", "")
        if not local_base64:
            logger.info(f"[Load images] Loading image from path: {local_path}")
            state["local_view_base64"] = image_to_base64(local_path)
        else:
            logger.debug("[Load images] local_view_base64 ready")
        
        return state
    
    def _finalize_output(self, state: WorkflowState) -> WorkflowState:
        """Generate final output. Uses Detector results from state."""
        logger.debug("[Finalize] Organizing results...")
        final_has_error = state.get("has_error", False)
        
        if final_has_error:
            state["final_output"] = {
                "error": True,
                "error_message": state.get("error_message", "Unknown error"),
                "anomaly_intervals": state.get("detector_anomaly_intervals", []),
                "anomaly_types": state.get("detector_anomaly_types", []),
                "explanations": state.get("explanations", []),
                "confidences": state.get("confidences", []),
                "anomaly_points": self._extract_anomaly_points(state)
            }
            logger.warning(f"[Finalize] Workflow terminated with error: {state.get('error_message', 'Unknown error')}")
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
                "needs_refinement": state.get("needs_refinement", False)
            }
        
        return state
    
    def _extract_anomaly_points(self, state: WorkflowState) -> list:
        """Extract anomaly point values."""
        intervals = state.get("detector_anomaly_intervals", [])
        data = state["data"]
        timestamp_col, value_col = get_column_names(data)

        anomaly_points = []
        for interval in intervals:
            if not isinstance(interval, (list, tuple)) or len(interval) == 0:
                logger.warning(f"[_extract_anomaly_points] Skipping invalid interval format: {interval}")
                continue
            if len(interval) == 1:
                start = end = interval[0]
            elif len(interval) >= 2:
                start, end = interval[0], interval[1]
            if timestamp_col in data.columns:
                start_rows = data[data[timestamp_col] == start]
                end_rows = data[data[timestamp_col] == end]
                if len(start_rows) > 0 and len(end_rows) > 0:
                    start_row = start_rows.index[0]
                    end_row = end_rows.index[0]
                    if start_row > end_row:
                        start_row, end_row = end_row, start_row
                    interval_data = data.iloc[start_row:end_row+1]
                    points = interval_data[value_col].tolist()
                    anomaly_points.extend(points)
            else:
                if 0 <= start < len(data) and 0 <= end < len(data):
                    if start > end:
                        start, end = end, start
                    interval_data = data.iloc[start:end+1]
                    points = interval_data[value_col].tolist()
                    anomaly_points.extend(points)
        
        return anomaly_points



class LiteAgent(agl.LitAgent[Dict[str, Any]]):
    _task_counter = 0
    _task_id_map: Dict[str, str] = {}

    def __init__(
        self,
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        localization_config: LLMConfig = None,
        locator_config: LLMConfig = None,
        evaluator_config: LLMConfig = None,
        actor_config: LLMConfig = None,
        detector_config: LLMConfig = None,
        rollout_data_dir: Optional[str] = None,
        enable_evaluator: bool = True,
        enable_checking: bool = None,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.localization_config = localization_config
        self.locator_config = locator_config
        self.evaluator_config = evaluator_config
        self.actor_config = actor_config
        self.detector_config = detector_config
        self.enable_evaluator = enable_evaluator if enable_checking is None else enable_checking
        self.knowledgeBase = KnowledgeBase()
        self.rollout_data_dir = Path(rollout_data_dir) if rollout_data_dir else Path("rollout_records")

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        """Execute rollout and record all Agent I/O."""
        rollout_id = rollout.rollout_id
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        
        segment_folder = task.get('segment_folder')
        if not segment_folder:
            logger.error(f"[Rollout {rollout_id}] Missing segment_folder in task")
            return 0.0
        
        segment_folder_path = Path(segment_folder).resolve()
        
        original_task_id = segment_folder_path.name
        if task.get('data_id'):
            original_task_id = task.get('data_id')
        
        if original_task_id not in LiteAgent._task_id_map:
            LiteAgent._task_counter += 1
            LiteAgent._task_id_map[original_task_id] = f"Task{LiteAgent._task_counter}"
        
        task_id = LiteAgent._task_id_map[original_task_id]
        
        rollout_dir = self.rollout_data_dir / task_id / rollout_id
        rollout_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            segment_data = load_segment_data(segment_folder_path)
            ground_truth = segment_data["ground_truth"]
            logger.debug(f"[Rollout {rollout_id}] Loaded data: {len(segment_data['segment_data'])} points, {len(ground_truth)} ground truth intervals")
        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] Failed to load data: {e}", exc_info=True)
            return 0.0
        
        is_training = rollout.mode == "train"
        if is_training:
            self.detector_config = LLMConfig(
                model_name=llm.model,
                api_key="dummy",
                base_url=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
                temperature=1,
                max_tokens=5000,
                extra_params={
                    "repetition_penalty": 1.1,
                }
            )
        else:
            self.detector_config = LLMConfig(
                model_name=llm.model,
                api_key="dummy",
                base_url=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
                temperature=(
                    self.val_temperature
                    if self.val_temperature is not None
                    else llm.sampling_parameters.get("temperature", 0.3)
                ),
                max_tokens=5000,
            )
        
        agent = TrainerAgent(
            max_turns=self.max_turns,
            knowledgeBase=self.knowledgeBase,
            locator_config=self.locator_config,
            localization_config=self.localization_config,
            evaluator_config=self.evaluator_config,
            actor_config=self.actor_config,
            detector_config=self.detector_config,
            training_mode=is_training,
            enable_evaluator=self.enable_evaluator
        ).build_workflow()

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
            "detector_conversation_history": [],
            "actor_conversation": [],
            "check_result": {},
            "needs_refinement": False,
            "refinement_count": 0,
            "agent_prompts_responses": {},
            "has_error": False,
            "error_message": "",
            "final_output": {},
            "agent_timings": {}
        }

        WORKFLOW_TIMEOUT_SECONDS = 900.0
        workflow_start_time = time.time()
        
        logger.info(f"[Rollout {rollout_id}] Starting workflow execution (timeout: {WORKFLOW_TIMEOUT_SECONDS}s)")
        sys.stdout.flush()
        sys.stderr.flush()
        
        final_state = None
        workflow_exception = None
        workflow_timed_out = False
        
        result_queue = queue.Queue()

        def run_workflow():
            """Run workflow in separate thread."""
            try:
                handler = self.tracer.get_langchain_handler()
                invoke_result = agent.invoke(
                    initial_state,
                    {"callbacks": [handler] if handler else [], "recursion_limit": 100},
                )
                if isinstance(invoke_result, tuple):
                    state_result = invoke_result[0] if len(invoke_result) > 0 else {}
                    logger.warning(f"[Rollout {rollout_id}] agent.invoke() returned tuple instead of dict, using first element")
                elif isinstance(invoke_result, dict):
                    state_result = invoke_result
                else:
                    logger.error(f"[Rollout {rollout_id}] agent.invoke() returned unexpected type: {type(invoke_result)}")
                    state_result = invoke_result if isinstance(invoke_result, dict) else {}
                
                result_queue.put(("success", state_result, None))
            except Exception as e:
                result_queue.put(("error", None, e))
        
        workflow_thread = threading.Thread(target=run_workflow, daemon=True)
        workflow_thread.start()

        try:
            result_type, state_result, error = result_queue.get(timeout=WORKFLOW_TIMEOUT_SECONDS)
            workflow_duration = time.time() - workflow_start_time
            
            if result_type == "success":
                final_state = state_result
                logger.info(f"[Rollout {rollout_id}] ✅ Workflow completed in {workflow_duration:.2f}s")
            elif result_type == "error":
                workflow_exception = error
                logger.error(f"[Rollout {rollout_id}] ❌ Workflow failed after {workflow_duration:.2f}s: {error}", exc_info=True)
                
        except queue.Empty:
            workflow_timed_out = True
            workflow_duration = time.time() - workflow_start_time
            logger.error(f"[Rollout {rollout_id}] ⚠️ WORKFLOW TIMEOUT after {workflow_duration:.2f}s (limit: {WORKFLOW_TIMEOUT_SECONDS}s)")
            logger.error(f"[Rollout {rollout_id}] This rollout will be marked as failed with reward 0.0")
            logger.warning(f"[Rollout {rollout_id}] The workflow thread may still be running in the background")

        if workflow_timed_out or workflow_exception is not None:
            failure_reason = "workflow_timeout" if workflow_timed_out else f"workflow_exception: {str(workflow_exception)}"
            failure_final_state = {
                "has_error": True,
                "error_message": failure_reason,
                "final_output": {
                    "error": True,
                    "error_message": failure_reason,
                    "anomaly_intervals": [],
                    "anomaly_types": [],
                    "explanations": [],
                    "confidences": [],
                },
                "agent_prompts_responses": {},
                "agent_timings": {},
            }
            final_state = failure_final_state

            try:
                save_rollout_records(
                    rollout_id=rollout_id,
                    initial_state=initial_state,
                    final_state=failure_final_state,
                    reward_value=0.0,
                    rollout_dir=rollout_dir,
                    ground_truth=ground_truth,
                    action_token_count=None,
                    failure_reasons=[failure_reason]
                )
            except Exception as e:
                logger.error(f"[Rollout {rollout_id}] Failed to save error records: {e}", exc_info=True)
            
            return 0.0

        if final_state is None:
            logger.error(f"[Rollout {rollout_id}] Unexpected: final_state is None")
            return 0.0

        end_time_rollout = time.time()
        reward_value = 0.0
        
        try:
            final_output = final_state.get("final_output", {}) if final_state else {}
            
            if final_output.get("error", False):
                reward_value = 0.0
                error_msg = final_output.get("error_message", "Unknown error")
                logger.warning(
                    f"[Rollout {rollout_id}] ActionAgent execution failed: {error_msg}, reward=0.0"
                )
            else:
                anomaly_intervals = final_output.get("anomaly_intervals", [])
                reward_value = float(reward(anomaly_intervals, ground_truth))
                
                logger.info(
                    f"[Rollout {rollout_id}] ActionAgent produced valid results: "
                    f"intervals={len(anomaly_intervals)}, reward={reward_value:.2f}"
                )
        except Exception as e:
            logger.error(
                f"[Rollout {rollout_id}] Failed to calculate reward: {e}", 
                exc_info=True
            )
            reward_value = 0.0

        anomaly_intervals = final_output.get("anomaly_intervals", [])
        logger.info(
            f"[Rollout {rollout_id}] Completed: "
            f"anomalies={len(anomaly_intervals)}, reward={reward_value:.2f}, "
            f"duration={end_time_rollout - start_time:.2f}s"
        )
        
        try:
            failure_reasons = None
            if final_output.get("error", False):
                error_msg = final_output.get("error_message", "")
                if error_msg:
                    failure_reasons = [error_msg]
            
            save_rollout_records(
                rollout_id=rollout_id,
                initial_state=initial_state,
                final_state=final_state if final_state else {},
                reward_value=reward_value,
                rollout_dir=rollout_dir,
                ground_truth=ground_truth,
                action_token_count=None,
                failure_reasons=failure_reasons
            )
        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] Failed to save rollout records: {e}", exc_info=True)
        
        return reward_value

