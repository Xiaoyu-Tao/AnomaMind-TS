# -*- coding: utf-8 -*-
"""
Actor: Tool calling phase - invokes time series analysis tools per the plan.
"""

import json
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from knowledge_base import KnowledgeBase
from analysis_tools import create_all_tools, get_available_tools_description
from utils import get_column_names
from workflowstate import WorkflowState


class Actor:
    """Tool-calling Actor that invokes tools per the analysis plan."""

    def __init__(self, knowledge_base: KnowledgeBase, llm=None):
        self.knowledge_base = knowledge_base
        self.llm = llm

    def tool_call(self, state: WorkflowState) -> WorkflowState:
        """Execute tool calling phase."""
        if state.get("has_error", False):
            return state

        messages = None
        try:
            data = state["data"]
            timestamp_col, value_col = get_column_names(data)
            segment_folder_path = None
            local_view_path = state.get("local_view_path", "")
            if local_view_path:
                segment_folder_path = str(Path(local_view_path).parent)

            if not hasattr(self, 'llm') or self.llm is None:
                raise ValueError("Actor llm is not initialized")

            tools = create_all_tools(data, segment_folder_path=segment_folder_path)
            bound_llm = self.llm.bind_tools(tools)
            tools_dict = {tool.name: tool for tool in tools}

            value_series = data[value_col]
            all_values = value_series.tolist()
            all_indices = data[timestamp_col].tolist() if timestamp_col in data.columns else list(range(len(data)))

            data_dict = {}
            for i in range(len(all_values)):
                idx = all_indices[i] if i < len(all_indices) else i
                val = all_values[i]
                try:
                    data_dict[str(idx)] = float(val) if isinstance(val, (int, float, str)) else val
                except (ValueError, TypeError):
                    data_dict[str(idx)] = val

            values_str = json.dumps(data_dict, indent=2, ensure_ascii=False)
            tool_call_prompt = f"""
You are a time series anomaly detection expert. Please call the necessary tools to analyze the time series data based on the analysis plan.

## Analysis Plan
{state["plan"]}

## Complete Time Series Values (index: value)
{values_str}

## Important
- The index values shown above are the actual Index values from the CSV file. The data index range is [{min(all_indices) if all_indices else 0}, {max(all_indices) if all_indices else len(all_values)-1}]. When referencing data points or calling tools, use these Index values directly.

## Your Task
You should call tools based on plan. Call all you tools you need in one turn.
"""

            messages = [
                SystemMessage(content="You are a professional time series anomaly detection expert. You have access to tools for time series analysis."),
                HumanMessage(content=tool_call_prompt)
            ]

            if "agent_prompts_responses" not in state:
                state["agent_prompts_responses"] = {}
            state["agent_prompts_responses"]["actor"] = {
                "system_message": "You are a professional time series anomaly detection expert.",
                "prompt": tool_call_prompt,
                "response": None
            }

            response = bound_llm.invoke(messages)
            messages.append(response)

            if response.tool_calls:
                print(f"\n[Actor] Detected {len(response.tool_calls)} tool call(s)")
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    if tool_name in tools_dict:
                        try:
                            tool_result = tools_dict[tool_name].invoke(tool_args)
                            print(f"[Actor] Tool call: {tool_name}({tool_args})")
                            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))
                        except Exception as e:
                            error_msg = f"Error executing tool {tool_name}: {str(e)}"
                            print(f"[Actor] {error_msg}")
                            messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                    else:
                        error_msg = f"Unknown tool: {tool_name}"
                        messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
            else:
                print(f"\n[Actor] No tool calls detected")

            state["actor_conversation"] = messages
            response_content = response.content if hasattr(response, 'content') else str(response)
            state["agent_prompts_responses"]["actor"]["response"] = response_content
            print(f"\n[Actor] State set: actor_conversation length={len(messages)}")

        except Exception as e:
            error_str = str(e)
            print(f"\n[Actor] Error: {type(e).__name__}: {error_str}")
            import traceback
            print(f"[Actor] Traceback:\n{traceback.format_exc()}")
            state["has_error"] = True
            state["error_message"] = f"Actor tool call failed: {error_str}"
            if messages is None:
                messages = [
                    SystemMessage(content="You are a professional time series anomaly detection expert."),
                    HumanMessage(content=f"Tool calling phase failed: {error_str}")
                ]
            state["actor_conversation"] = messages

        return state
