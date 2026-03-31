from __future__ import annotations
from typing import Any, Dict, Optional, cast
from knowledge_base import KnowledgeBase
from llm_config import LLMConfig
from pathlib import Path
from utils import load_segment_data
from reward import reward
import agentlightning as agl
import logging
import queue
import sys
import threading
import time

from AnomaAgent import AnomaAgent

logger = logging.getLogger(__name__)


class TrainerAgent(agl.LitAgent[Dict[str, Any]]):
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
        enable_evaluator: bool = True,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.localization_config = localization_config
        self.locator_config = locator_config
        self.evaluator_config = evaluator_config
        self.actor_config = actor_config
        self.detector_config = detector_config
        self.enable_evaluator = enable_evaluator
        self.knowledgeBase = KnowledgeBase()

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
        
        if original_task_id not in TrainerAgent._task_id_map:
            TrainerAgent._task_counter += 1
            TrainerAgent._task_id_map[original_task_id] = f"Task{TrainerAgent._task_counter}"
        
        task_id = TrainerAgent._task_id_map[original_task_id]
        
        try:
            segment_data = load_segment_data(segment_folder_path)
            ground_truth = segment_data["ground_truth"]
            logger.debug(f"[Rollout {rollout_id}] Loaded data: {len(segment_data['segment_data'])} points, {len(ground_truth)} ground truth intervals")
        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] Failed to load data: {e}", exc_info=True)
            return 0.0
        
        self.detector_config = LLMConfig(
            model_name=llm.model,
            api_key="dummy",
            base_url=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
            temperature=(
                self.val_temperature
                if self.val_temperature is not None
                else llm.sampling_parameters.get("temperature", 1)
            ),
            max_tokens=5000,
        )
        
        agent = AnomaAgent(
            max_turns=self.max_turns,
            knowledgeBase=self.knowledgeBase,
            locator_config=self.locator_config,
            localization_config=self.localization_config,
            evaluator_config=self.evaluator_config,
            actor_config=self.actor_config,
            detector_config=self.detector_config,
            enable_evaluator=self.enable_evaluator,
        ).build_workflow()

        initial_state = {
            "data": segment_data["segment_data"],
            "local_view_path": segment_data["segment_image_path"],
            "local_view_base64": segment_data["segment_image_base64"],
            "localization_anomaly_intervals": [],
            "localization_anomaly_types": [],
            "localization_anomaly_reasons": [],
            "visual_description": "",
            "plan": "",
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
            }
            final_state = failure_final_state
            
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
        
        return reward_value

