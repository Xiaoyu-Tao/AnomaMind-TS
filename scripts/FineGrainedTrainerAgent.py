# -*- coding: utf-8 -*-
"""
DetectorTrainerAgent: Trains only the Detector module.
Reads prompt and ground_truth from CSV/parquet.
"""

from __future__ import annotations
import logging
import time
import json
import threading
import queue
import sys
import re
import collections
import uuid
from typing import Any, Dict, List, Optional, cast
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

from knowledge_base import KnowledgeBase
from llm_config import LLMConfig, create_llm
from utils import save_rollout_records
from workflowstate import WorkflowState
from Detector import Detector
from reward import reward

import agentlightning as agl

logger = logging.getLogger(__name__)

class DetectorTrainerAgent:
    """Training Agent with only the Detector node."""

    def __init__(
        self,
        detector_config: LLMConfig = None,
        knowledgeBase: KnowledgeBase = None,
    ):
        self.detector_llm = create_llm(detector_config)
        self.knowledgeBase = knowledgeBase if knowledgeBase else KnowledgeBase()
        self.detector = Detector(knowledge_base=self.knowledgeBase, llm=self.detector_llm)

    def parse_chatml_to_messages(self, chatml_str: str) -> List:
        messages = []
        chatml_str = chatml_str.replace('""', '"')

        pattern = r'<\|im_start\|>(system|user|assistant|tool)\n(.*?)\n<\|im_end\|>'
        matches = re.finditer(pattern, chatml_str, re.DOTALL)
        pending_tool_ids = collections.deque()
        
        for match in matches:
            role = match.group(1)
            content = match.group(2).strip()
            
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                try:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        t_calls = parsed.get("tool_calls", [])
                        if t_calls:
                            for tc in t_calls:
                                if "id" in tc:
                                    pending_tool_ids.append(tc["id"])
                            
                            messages.append(AIMessage(
                                content=parsed.get("text", ""),
                                tool_calls=t_calls
                            ))
                            continue
                    messages.append(AIMessage(content=content))
                except Exception:
                    messages.append(AIMessage(content=content))
                    
            elif role == "tool":
                if pending_tool_ids:
                    current_id = pending_tool_ids.popleft()
                else:
                    current_id = str(uuid.uuid4())
                    
                messages.append(ToolMessage(content=content, tool_call_id=current_id))
        
        return messages
    
    def parse_prompt_string_to_messages(self, prompt_str: str) -> List:
        """Parse prompt string (ChatML format or plain text)."""
        if "<|im_start|>" in prompt_str and "<|im_end|>" in prompt_str:
            return self.parse_chatml_to_messages(prompt_str)
        else:
            return [HumanMessage(content=prompt_str)]
    
    def detect(self, state: WorkflowState) -> WorkflowState:
        """Detector detection node."""
        start_time = time.time()
        messages = state.get("tool_call_conversation", []) or state.get("actor_conversation", [])

        if not messages:
            logger.error("[DetectorTrainerAgent] tool_call_conversation is empty")
            state["has_error"] = True
            state["error_message"] = "tool_call_conversation is empty"
            return state

        try:
            response = self.detector.llm.invoke(messages)
            final_content = response.content if hasattr(response, 'content') else str(response)
            if not isinstance(final_content, str):
                final_content = str(final_content)
            self.detector._parse_response(final_content, state)
            state["tool_call_conversation"] = messages + [response]
            if "agent_prompts_responses" not in state:
                state["agent_prompts_responses"] = {}
            state["agent_prompts_responses"]["detector"] = {
                "prompt": messages[-1].content if len(messages) >= 1 else "",
                "response": final_content
            }

        except Exception as e:
            logger.error(f"[DetectorTrainerAgent] Inference failed: {e}", exc_info=True)
            state["has_error"] = True
            state["error_message"] = f"detect failed: {str(e)}"
            state.update({
                "detector_anomaly_intervals": [],
                "detector_anomaly_types": [],
                "explanations": [],
                "confidences": []
            })
        
        end_time = time.time()
        if "agent_timings" not in state:
            state["agent_timings"] = {}
        state["agent_timings"]["detect"] = {
            "start_time": start_time,
            "end_time": end_time,
            "execution_time_seconds": end_time - start_time
        }
        
        return state
    
    def _parse_detector_response(self, final_content: str, state: WorkflowState):
        """Parse Detector response (handled by Detector._parse_response, kept for compatibility)."""
        self.detector._parse_response(final_content, state)

    def finalize(self, state: WorkflowState) -> WorkflowState:
        """Final output node."""
        has_err = state.get("has_error", False)
        state["final_output"] = {
            "error": has_err,
            "error_message": state.get("error_message", ""),
            "anomaly_intervals": state.get("detector_anomaly_intervals", []),
            "anomaly_types": state.get("detector_anomaly_types", []),
            "explanations": state.get("explanations", []),
            "confidences": state.get("confidences", []),
        }
        return state

    def build_workflow(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        workflow.add_node("detect", self.detect)
        workflow.add_node("finalize", self.finalize)
        workflow.set_entry_point("detect")
        workflow.add_edge("detect", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()


# Backward compatibility
FineGrainedTrainerAgent = DetectorTrainerAgent


class FineGrainedLiteAgent(agl.LitAgent[Dict[str, Any]]):
    """Simplified LiteAgent that trains only the Detector."""

    _task_counter = 0
    _task_id_map: Dict[str, str] = {}

    def __init__(
        self,
        val_temperature: Optional[float] = None,
        detector_config: LLMConfig = None,
        fine_grained_config: LLMConfig = None,
        rollout_data_dir: Optional[str] = None,
        save_rollout_records: bool = False,
        dismatch_punish: float = 0,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.detector_config = detector_config or fine_grained_config
        self.knowledgeBase = KnowledgeBase()
        self.rollout_data_dir = Path(rollout_data_dir) if rollout_data_dir else Path("rollout_records")
        self.save_rollout_records = save_rollout_records
        self.dismatch_punish = dismatch_punish
    
    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        """
        Execute rollout (reads prompt and ground_truth from CSV).

        task should contain:
        - prompt: ChatML string or plain text (from CSV)
        - ground_truth: JSON string (from CSV)
        - segment_folder: optional, for logging
        """
        rollout_id = rollout.rollout_id
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        
        prompt_str = task.get('prompt')
        ground_truth_str = task.get('ground_truth')
        segment_folder = task.get('segment_folder', '')
        
        if not prompt_str:
            logger.error(f"[Rollout {rollout_id}] Missing 'prompt' in task")
            return self.dismatch_punish
        
        if not ground_truth_str:
            logger.error(f"[Rollout {rollout_id}] Missing 'ground_truth' in task")
            return self.dismatch_punish
        
        original_task_id = segment_folder if segment_folder else f"task_{rollout_id}"
        if task.get('data_id'):
            original_task_id = task.get('data_id')
        
        if original_task_id not in FineGrainedLiteAgent._task_id_map:
            FineGrainedLiteAgent._task_counter += 1
            FineGrainedLiteAgent._task_id_map[original_task_id] = f"Task{FineGrainedLiteAgent._task_counter}"
        
        task_id = FineGrainedLiteAgent._task_id_map[original_task_id]
        
        rollout_dir = None
        if self.save_rollout_records:
            rollout_dir = self.rollout_data_dir / task_id / rollout_id
            rollout_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            ground_truth_dict = json.loads(ground_truth_str)
            ground_truth = self._convert_gt_dict_to_intervals(ground_truth_dict)
        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] Failed to parse ground_truth: {e}", exc_info=True)
            return self.dismatch_punish
        
        is_training = rollout.mode == "train"
        
        if is_training:
            self.detector_config = LLMConfig(
                model_name=llm.model,
                api_key="dummy",
                base_url=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
                temperature=1,
                max_tokens=5000,
                extra_params={
                    "repetition_penalty": 1.0,
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
        
        agent = DetectorTrainerAgent(
            detector_config=self.detector_config,
            knowledgeBase=self.knowledgeBase,
        ).build_workflow()

        trainer_agent = DetectorTrainerAgent(
            detector_config=self.detector_config,
            knowledgeBase=self.knowledgeBase,
        )
        messages = trainer_agent.parse_prompt_string_to_messages(prompt_str)

        initial_state: WorkflowState = {
            "tool_call_conversation": messages,
            "detector_anomaly_intervals": [],
            "detector_anomaly_types": [],
            "explanations": [],
            "confidences": [],
            "agent_prompts_responses": {},
            "has_error": False,
            "error_message": "",
            "final_output": {},
            "agent_timings": {},
            "data": None,
        }

        WORKFLOW_TIMEOUT_SECONDS = 300.0
        workflow_start_time = time.time()
        
        logger.info(f"[Rollout {rollout_id}] 🚀 Starting Detector workflow (timeout: {WORKFLOW_TIMEOUT_SECONDS}s)")
        sys.stdout.flush()
        sys.stderr.flush()
        
        final_state = None
        workflow_exception = None
        workflow_timed_out = False
        
        result_queue = queue.Queue()
        def run_workflow():
            try:
                handler = self.tracer.get_langchain_handler()
                invoke_result = agent.invoke(
                    initial_state,
                    {"callbacks": [handler] if handler else [], "recursion_limit": 100},
                )
                if isinstance(invoke_result, tuple):
                    state_result = invoke_result[0] if len(invoke_result) > 0 else {}
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
            logger.error(f"[Rollout {rollout_id}] ⚠️ WORKFLOW TIMEOUT after {workflow_duration:.2f}s")
        
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

            if self.save_rollout_records and rollout_dir:
                try:
                    self._save_rollout_data(
                        rollout_dir=rollout_dir,
                        prompt_str=prompt_str,
                        ground_truth_str=ground_truth_str,
                        final_state=failure_final_state,
                        reward_value=self.dismatch_punish,
                        rollout_id=rollout_id,
                    )
                except Exception as e:
                    logger.error(f"[Rollout {rollout_id}] Failed to save error records: {e}", exc_info=True)
            
            return self.dismatch_punish
        
        if final_state is None:
            logger.error(f"[Rollout {rollout_id}] Unexpected: final_state is None")
            return self.dismatch_punish

        end_time_rollout = time.time()
        reward_value = self.dismatch_punish
        
        try:
            final_output = final_state.get("final_output", {}) if final_state else {}
            
            if final_output.get("error", False):
                reward_value = self.dismatch_punish
                error_msg = final_output.get("error_message", "Unknown error")
                logger.warning(f"[Rollout {rollout_id}] Execution failed: {error_msg}, reward={self.dismatch_punish}")
            else:
                anomaly_intervals = final_output.get("anomaly_intervals", [])
                reward_value = float(reward(anomaly_intervals, ground_truth))
                
                logger.info(
                    f"[Rollout {rollout_id}] Produced valid results: "
                    f"intervals={len(anomaly_intervals)}, reward={reward_value:.2f}"
                )
        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] Failed to calculate reward: {e}", exc_info=True)
            reward_value = self.dismatch_punish
        
        anomaly_intervals = final_output.get("anomaly_intervals", [])
        logger.info(
            f"[Rollout {rollout_id}] Completed: "
            f"anomalies={len(anomaly_intervals)}, reward={reward_value:.2f}, "
            f"duration={end_time_rollout - start_time:.2f}s"
        )

        if self.save_rollout_records and rollout_dir:
            try:
                self._save_rollout_data(
                    rollout_dir=rollout_dir,
                    prompt_str=prompt_str,
                    ground_truth_str=ground_truth_str,
                    final_state=final_state if final_state else {},
                    reward_value=reward_value,
                    rollout_id=rollout_id,
                )
            except Exception as e:
                logger.error(f"[Rollout {rollout_id}] Failed to save rollout data: {e}", exc_info=True)

            try:
                failure_reasons = None
                if reward_value == self.dismatch_punish and final_output.get("error", False):
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
    
    def _save_rollout_data(
        self,
        rollout_dir: Path,
        prompt_str: str,
        ground_truth_str: str,
        final_state: Dict[str, Any],
        reward_value: float,
        rollout_id: str,
    ) -> None:
        """Save rollout data to files (compact format).

        Structure:
        rollout_dir/
          ├── prompt_answer.json
          └── model_response.json
        """
        try:
            final_output = final_state.get("final_output", {})
            agent_responses = final_state.get("agent_prompts_responses", {})
            fine_grained_response = agent_responses.get("fine_grained", {})

            prompt_answer_data = {
                "rollout_id": rollout_id,
                "prompt": prompt_str,
                "answer": fine_grained_response.get("response", ""),
                "ground_truth": ground_truth_str,
                "reward": reward_value,
                "timestamp": time.time(),
            }
            
            prompt_answer_file = rollout_dir / "prompt_answer.json"
            with open(prompt_answer_file, "w", encoding="utf-8") as f:
                json.dump(prompt_answer_data, f, ensure_ascii=False, indent=2)

            model_response_data = {
                "rollout_id": rollout_id,
                "anomaly_intervals": final_output.get("anomaly_intervals", []),
                "anomaly_types": final_output.get("anomaly_types", []),
                "explanations": final_output.get("explanations", []),
                "confidences": final_output.get("confidences", []),
                "has_error": final_output.get("error", False),
                "error_message": final_output.get("error_message", ""),
                "agent_timings": final_state.get("agent_timings", {}),
                "reward": reward_value,
            }
            
            model_response_file = rollout_dir / "model_response.json"
            with open(model_response_file, "w", encoding="utf-8") as f:
                json.dump(model_response_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[Rollout {rollout_id}] ✓ Saved to {rollout_dir}")
            
        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] Failed to save rollout data: {e}", exc_info=True)
    
    def _convert_gt_dict_to_intervals(self, gt_dict: Dict[str, int]) -> List[List[int]]:
        """Convert ground truth dict to interval list.

        gt_dict: {"0": 0, "1": 1, "2": 1, "3": 0, ...}
        Returns: [[1, 2], [5, 7], ...]
        """
        intervals = []
        current_start = None
        sorted_keys = sorted([int(k) for k in gt_dict.keys() if gt_dict[k] == 1])
        
        if not sorted_keys:
            return intervals
        
        current_start = sorted_keys[0]
        current_end = sorted_keys[0]
        
        for i in range(1, len(sorted_keys)):
            if sorted_keys[i] == current_end + 1:
                current_end = sorted_keys[i]
            else:
                intervals.append([current_start, current_end])
                current_start = sorted_keys[i]
                current_end = sorted_keys[i]

        intervals.append([current_start, current_end])
        
        return intervals

