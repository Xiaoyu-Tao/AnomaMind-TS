# -*- coding: utf-8 -*-
"""
Locator: Creates anomaly detection plan based on Localization results.
"""

from knowledge_base import KnowledgeBase
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from utils import get_column_names, format_time_series_value
from workflowstate import WorkflowState
from analysis_tools import get_available_tools_description
import re


class Locator:
    def __init__(self, llm=None, knowledge_base: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.knowledge_base = knowledge_base or KnowledgeBase()

    def locate(self, state: WorkflowState) -> WorkflowState:
        print("\n[Locator] generating plan...")

        data = state["data"]
        timestamp_col, value_col = get_column_names(data)
        intervals = state["localization_anomaly_intervals"]
        types = state["localization_anomaly_types"]

        value_series = data[value_col]
        all_values = value_series.tolist()

        if timestamp_col in data.columns:
            all_indices = data[timestamp_col].tolist()
        else:
            all_indices = list(range(len(data)))

        values_str = ""
        for j in range(0, len(all_values), 10):
            chunk_values = all_values[j:j+10]
            chunk_indices = all_indices[j:j+10] if j < len(all_indices) else []
            chunk_str = ", ".join([format_time_series_value(v) for v in chunk_values])
            if chunk_indices and len(chunk_indices) > 0:
                start_idx = chunk_indices[0]
                end_idx = chunk_indices[-1]
            else:
                start_idx = j
                end_idx = min(j+9, len(all_values)-1)
            values_str += f"  [{start_idx}:{end_idx}]: {chunk_str}\n"

        intervals_text = ""
        if intervals:
            for i, (start, end) in enumerate(intervals):
                anomaly_type = types[i] if i < len(types) else "Unknown"
                intervals_text += f"- Anomaly candidate interval #{i+1}: indices [{start}, {end}], type: {anomaly_type}\n"
        else:
            intervals_text = "- No anomaly candidate intervals found \n"

        data_range_info = ""
        if all_indices:
            min_idx = min(all_indices)
            max_idx = max(all_indices)
            data_range_info = f"\n**Important**: The actual data index range in this dataset is [{min_idx}, {max_idx}]. All index references should use values within this range.\n"

        time_series_text = f"""
## Complete Time Series Values (index: value)
{values_str}
{data_range_info}
## Possible anomaly intervals
{intervals_text}
"""

        tools_description = get_available_tools_description()
        local_view_path = state.get("local_view_path", "")
        dataset_type = self.knowledge_base.detect_dataset_type(folder_path=local_view_path)
        locator_knowledge = self.knowledge_base.get_agent_knowledge("locator", dataset_type)

        if dataset_type != "UNKNOWN":
            print(f"[Locator] Detect dataset type: {dataset_type}")

        needs_refinement = state.get("needs_refinement", False)
        refinement_count = state.get("refinement_count", 0)

        if needs_refinement:
            refinement_count = refinement_count + 1
            state["refinement_count"] = refinement_count
            state["needs_refinement"] = False
            print(f"[Locator] Refinement needed")

        check_result = state.get("check_result", {})
        previous_feedback = ""

        if refinement_count > 0 and check_result:
            issues = check_result.get("issues", [])
            suggestions = check_result.get("suggestions", [])
            planning_quality = check_result.get("planning_quality", "")
            tool_usage_quality = check_result.get("tool_usage_quality", "")
            reasoning_quality = check_result.get("reasoning_quality", "")

            if issues or suggestions or planning_quality or tool_usage_quality or reasoning_quality:
                previous_feedback = f"""
## Previous Analysis Feedback (Refinement Round {refinement_count})
The previous analysis execution was reviewed and the following feedback was provided:
"""
                if planning_quality:
                    previous_feedback += f"- Planning Quality: {planning_quality}\n"
                if tool_usage_quality:
                    previous_feedback += f"- Tool Usage Quality: {tool_usage_quality}\n"
                if reasoning_quality:
                    previous_feedback += f"- Reasoning Quality: {reasoning_quality}\n"

                if issues:
                    previous_feedback += f"\n**Issues Found:**\n"
                    for issue in issues:
                        previous_feedback += f"- {issue}\n"

                if suggestions:
                    previous_feedback += f"\n**Suggestions for Improvement:**\n"
                    for suggestion in suggestions:
                        previous_feedback += f"- {suggestion}\n"

                previous_feedback += "\nPlease create a NEW and IMPROVED plan that addresses these issues and incorporates the suggestions.\n"

                print(f"[Locator] Feedback length: {len(previous_feedback)}")
            else:
                print(f"[Locator] Epoch {refinement_count} no feedback")
        elif refinement_count > 0:
            print(f"[Locator] check_result is empty")
        else:
            print(f"[Locator] Generating plan...")

        print(f"[Locator] ==================================\n")
        visual_description = state.get("visual_description", "")
        if visual_description:
            visual_info_section = f"""
## Visual Analysis Description (from Localization)
The visual model has analyzed the time series images and provided the following description:
{visual_description}

Note: This visual description is for reference only. It may be inaccurate, especially when flagging the whole period as anomaly.
"""
        else:
            visual_info_section = ""

        prompt = f"""
You are an anomaly detection planning expert. Please create a detailed anomaly detection plan.

{previous_feedback}

{visual_info_section}

## Complete Time Series Numerical Information
Below is the complete time series data (all data points).

{time_series_text}

{tools_description}
**You need to particular focus on Domain Knowledge to arrange your plan**
{locator_knowledge}
## Rule
- The executing Agent should only be responsible for invoking existing tools and performing simple analytical reasoning.
It should not employ complex methods that cannot be directly implemented via available tool calls.
- You should plan the most suitable toolcall based on timeseries itself, instead of meaningless one. For each toolcall, be specific.
- Your plan should be informative, with the combination of Domain Knowledge.
- For toolcall, set a threshold if need, and **tell action agent do not flag those under threshold as anomaly**.
- Only analyze the interval you can accessible.
- Try to complete all necessary tool calls in a single invocation to save tokens and time.
- If you need to think, keep it short.

## Output Format
Please provide your response in the following format:
You can first conduct think (this thinking process will not be used as the plan).
** Then, wrap your final plan in <Plan> tags. **
"""

        messages = [
            SystemMessage(content="You are an anomaly detection planning expert"),
            HumanMessage(content=prompt)
        ]

        if "agent_prompts_responses" not in state:
            state["agent_prompts_responses"] = {}
        state["agent_prompts_responses"]["locator"] = {
            "system_message": "You are an anomaly detection planning expert",
            "prompt": prompt,
            "response": None
        }

        try:
            response = self.llm.invoke(messages)
            full_response = response.content

            state["agent_prompts_responses"]["locator"]["response"] = full_response

            plan_match = re.search(r'<Plan>(.*?)</Plan>', full_response, re.DOTALL)

            if plan_match:
                plan_content = plan_match.group(1).strip()
                state["plan"] = plan_content
            else:
                print(f"[Locator] Warning: no <Plan> tag found")
                state["plan"] = full_response
                state["current_interval_index"] = -1
            state["current_interval_index"] = 0
            print(f"[Locator] Plan generated")
        except Exception as e:
            error_str = str(e)
            print(f"[Locator] error: {error_str}")

            if "402" in error_str or "Insufficient Balance" in error_str or "401" in error_str or "403" in error_str:
                state["has_error"] = True
                state["error_message"] = f"API_ERROR: {error_str}"
                print(f"[Locator] API ERROR")
                state["current_interval_index"] = -1
                return state

            state["plan"] = "Use default analysis strategy"
            state["current_interval_index"] = 0

        return state
