# -*- coding: utf-8 -*-
"""
Detector: Reasoning and detection phase - anomaly detection from tool call results.
"""

import json
import re
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from knowledge_base import KnowledgeBase
from analysis_tools import get_available_tools_description
from utils import get_column_names
from workflowstate import WorkflowState


class Detector:

    def __init__(self, knowledge_base: KnowledgeBase, llm=None):
        self.knowledge_base = knowledge_base
        self.llm = llm

    def detect(self, state: WorkflowState) -> WorkflowState:
        messages = state.get("actor_conversation", []) or state.get("tool_call_conversation", [])
        if not messages:
            print(f"\n[Detector] ❌ error: actor_conversation is empty")
            state["has_error"] = True
            return state

        used_tool_names = set()
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    used_tool_names.add(tc['name'])

        print(f"[Detector] 🛠️ tool: {used_tool_names}")

        full_desc = get_available_tools_description()
        filtered_tool_desc = "### Logic of Tools Used in Previous Steps:\n"
        tool_blocks = full_desc.split("\n\n")
        for block in tool_blocks:
            for name in used_tool_names:
                if f"**{name}" in block or f"{name}(" in block:
                    filtered_tool_desc += block + "\n\n"
                    break

        data = state["data"]
        timestamp_col, value_col = get_column_names(data)
        value_series = data[value_col]
        all_indices = data[timestamp_col].tolist() if timestamp_col in data.columns else list(range(len(data)))
        data_dict = {str(all_indices[i]): float(value_series.iloc[i]) for i in range(len(value_series))}
        local_view_path = state.get("local_view_path", "")
        dataset_type = None
        if local_view_path:
            detected_type = self.knowledge_base.detect_dataset_type(folder_path=str(Path(local_view_path).parent))
            if detected_type != "UNKNOWN":
                dataset_type = detected_type

        detector_knowledge = self.knowledge_base.get_agent_knowledge(
            agent_type="detector",
            dataset_type=dataset_type
        )

        final_answer_prompt = f"""
You have received the tool call results in the conversation history above.
(DO NOT CALL TOOL AGAIN)
Now provide your concise reasoning and final answer based on the analysis plan, the historical tool results, and the raw time series data.
{filtered_tool_desc}

{detector_knowledge if detector_knowledge else "Follow standard time series anomaly detection rules."}

## Analysis Plan
{state["plan"]}

## Your Task
Based on plan and tool result, time series value,
give a concise reasoning and then output result.

## Output format:
<think>
Your concise analysis of tool result and plan here.
</think>
[
    {{
        "interval": [start_index, end_index],
        "type": "anomaly_type_name",
        "explanation": "Specific reason for this anomaly",
        "confidence": 1-3
    }}
]

## IMPORTANT:
- If no anomalies detected, return empty arrays.
- Use actual Index values from the data above.
- Confidence should be an integer from 1 to 3.
"""

        messages.append(HumanMessage(content=final_answer_prompt))

        if not hasattr(self, 'llm') or self.llm is None:
            raise ValueError("Detector llm is not initialized")
        response = self.llm.invoke(messages)
        messages.append(response)

        final_content = response.content if hasattr(response, 'content') else str(response)
        if not isinstance(final_content, str):
            final_content = str(final_content)

        if "agent_prompts_responses" not in state:
            state["agent_prompts_responses"] = {}
        state["agent_prompts_responses"]["detector"] = {
            "system_message": "You are a professional time series anomaly detection expert.",
            "prompt": final_answer_prompt,
            "response": final_content
        }

        if not final_content or len(final_content.strip()) == 0:
            state["has_error"] = True
            state["error_message"] = "Detector parse failed: empty response"
            state["detector_anomaly_intervals"] = []
            state["detector_anomaly_types"] = []
            state["explanations"] = []
            state["confidences"] = []
            return state

        conversation_history = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                conversation_history.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                conversation_history.append({"role": "user", "content": str(msg.content) if isinstance(msg.content, str) else str(msg.content)})
            elif isinstance(msg, AIMessage):
                msg_content = {"text": msg.content}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    msg_content["tool_calls"] = [{"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")} for tc in msg.tool_calls]
                conversation_history.append({"role": "assistant", "content": msg_content})
            elif isinstance(msg, ToolMessage):
                conversation_history.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})

        state["detector_conversation_history"] = conversation_history
        if "detector_anomaly_intervals" not in state:
            state["detector_anomaly_intervals"] = []
        if "detector_anomaly_types" not in state:
            state["detector_anomaly_types"] = []
        if "explanations" not in state:
            state["explanations"] = []
        if "confidences" not in state:
            state["confidences"] = []

        try:
            reasoning_match = re.search(r'<think>(.*?)</think>', final_content, re.DOTALL)
            search_start_pos = reasoning_match.end() if reasoning_match else 0

            json_match = re.search(r'\[\s*\{.*\}\s*\]|\[\s*\]', final_content[search_start_pos:], re.DOTALL)
            if not json_match:
                json_match = re.search(r'\[\s*\{.*\}\s*\]|\[\s*\]', final_content, re.DOTALL)
            if not json_match:
                raise ValueError("no json detected")

            anomalies_list = json.loads(json_match.group())
            if not isinstance(anomalies_list, list):
                raise ValueError("JSON is not list type")

            state["detector_anomaly_intervals"] = []
            state["detector_anomaly_types"] = []
            state["explanations"] = []
            state["confidences"] = []

            for item in anomalies_list:
                if not isinstance(item, dict):
                    continue
                raw_interval = item.get("interval", [])
                if isinstance(raw_interval, list) and len(raw_interval) >= 1:
                    try:
                        start = int(raw_interval[0])
                        end = int(raw_interval[1]) if len(raw_interval) >= 2 else start
                        if start > end:
                            start, end = end, start
                        state["detector_anomaly_intervals"].append([start, end])
                        state["detector_anomaly_types"].append(str(item.get("type", "Unknown")))
                        state["explanations"].append(str(item.get("explanation", "")))
                        state["confidences"].append(int(float(item.get("confidence", 1))))
                    except (ValueError, TypeError, IndexError):
                        pass

            print(f"[Detector] ✓ parsing  {len(state['detector_anomaly_intervals'])} intervals")

        except Exception as e:
            print(f"[Detector] ❌ parsing error: {e}")
            state["has_error"] = True
            state["error_message"] = str(e)
            state["detector_anomaly_intervals"] = []
            state["detector_anomaly_types"] = []
            state["explanations"] = []
            state["confidences"] = []

        return state

    def _parse_response(self, final_content: str, state: WorkflowState) -> None:
        if not final_content or len(final_content.strip()) == 0:
            state["has_error"] = True
            state["error_message"] = "Detector parsing error"
            state["detector_anomaly_intervals"] = []
            state["detector_anomaly_types"] = []
            state["explanations"] = []
            state["confidences"] = []
            return
        try:
            reasoning_match = re.search(r'<(think|redacted_reasoning)>(.*?)</\1>', final_content, re.DOTALL)
            search_start_pos = reasoning_match.end() if reasoning_match else 0
            json_match = re.search(r'\[\s*\{.*\}\s*\]|\[\s*\]', final_content[search_start_pos:], re.DOTALL)
            if not json_match:
                json_match = re.search(r'\[\s*\{.*\}\s*\]|\[\s*\]', final_content, re.DOTALL)
            if not json_match:
                raise ValueError("no JSON list")
            anomalies_list = json.loads(json_match.group())
            if not isinstance(anomalies_list, list):
                raise ValueError("JSON is not list type")
            state["detector_anomaly_intervals"] = []
            state["detector_anomaly_types"] = []
            state["explanations"] = []
            state["confidences"] = []
            for item in anomalies_list:
                if not isinstance(item, dict):
                    continue
                raw_interval = item.get("interval", [])
                if isinstance(raw_interval, list) and len(raw_interval) >= 1:
                    try:
                        start = int(raw_interval[0])
                        end = int(raw_interval[1]) if len(raw_interval) >= 2 else start
                        state["detector_anomaly_intervals"].append([min(start, end), max(start, end)])
                        state["detector_anomaly_types"].append(str(item.get("type", "Unknown")))
                        state["explanations"].append(str(item.get("explanation", "")))
                        state["confidences"].append(int(float(item.get("confidence", 1))))
                    except (ValueError, TypeError, IndexError):
                        pass
        except Exception as e:
            state["has_error"] = True
            state["error_message"] = str(e)
            state["detector_anomaly_intervals"] = []
            state["detector_anomaly_types"] = []
            state["explanations"] = []
            state["confidences"] = []
