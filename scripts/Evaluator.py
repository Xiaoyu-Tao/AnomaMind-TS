# -*- coding: utf-8 -*-
"""
Evaluator: Evaluates detection quality and decides if refinement is needed.
"""

from langchain_core.messages import HumanMessage, SystemMessage
import json
from utils import safe_json_loads
from workflowstate import WorkflowState


class Evaluator:
    def __init__(self, llm=None):
        self.llm = llm

    def evaluate(self, state: WorkflowState) -> WorkflowState:
        if state.get("has_error", False):
            state["check_result"] = {"needs_refinement": False}
            state["needs_refinement"] = False
            return state

        plan = state.get("plan", "")
        fine_grained_conversation = state.get("detector_conversation_history", []) or state.get("fine_grained_conversation_history", [])
        fine_intervals = state.get("detector_anomaly_intervals", []) or state.get("fine_anomaly_intervals", [])
        fine_types = state.get("detector_anomaly_types", []) or state.get("fine_anomaly_types", [])
        explanations = state.get("explanations", [])
        confidences = state.get("confidences", [])

        conversation_history_text = ""
        if fine_grained_conversation:
            conversation_parts = []
            for msg in fine_grained_conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    if isinstance(content, str):
                        conversation_parts.append(f"[System]: {content}")

                elif role == "user":
                    if isinstance(content, str):
                        if len(content) > 2500:
                            conversation_parts.append(f"[User]: {content[:2000]}...\n[User (continued)]: ...{content[-500:]}")
                        else:
                            conversation_parts.append(f"[User]: {content}")

                elif role == "assistant":
                    if isinstance(content, dict):
                        tool_calls = content.get("tool_calls", [])
                        if tool_calls:
                            tool_calls_str = []
                            for tc in tool_calls:
                                tool_name = tc.get('name', '')
                                tool_args = tc.get('args', {})
                                tool_calls_str.append(f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
                            conversation_parts.append(f"[Assistant - Tool Calls]: {', '.join(tool_calls_str)}")
                        text = content.get("text", "")
                        if text:
                            if len(text) > 4000:
                                conversation_parts.append(f"[Assistant - Text]: {text[:3000]}...\n[Assistant - Text (continued)]: ...{text[-1000:]}")
                            else:
                                conversation_parts.append(f"[Assistant - Text]: {text}")
                    elif isinstance(content, str):
                        if len(content) > 4000:
                            conversation_parts.append(f"[Assistant]: {content[:3000]}...\n[Assistant (continued)]: ...{content[-1000:]}")
                        else:
                            conversation_parts.append(f"[Assistant]: {content}")

                elif role == "tool":
                    if isinstance(content, str):
                        conversation_parts.append(f"[Tool Response]: {content}")

            conversation_history_text = "\n\n".join(conversation_parts)

        prompt = f"""Review anomaly detection execution:

## Analysis Plan
{plan}

## Conversation History
{conversation_history_text}

## Review Task
Evaluate leniently. Only flag super serious issues.

Guidelines:
- Set needs_refinement=false if analysis is generally reasonable
- Only true for major errors: critical logic errors, no reasoning
- Quality ratings: "good"/"acceptable"/"poor" (use "poor" only for serious problems)

JSON:
{{
    "issues": ["issue1"],
    "suggestions": ["suggestion1"],
    "needs_refinement": true/false,
    "planning_quality": "good/acceptable/poor",
    "tool_usage_quality": "good/acceptable/poor",
    "reasoning_quality": "good/acceptable/poor"
}}"""

        system_message_content = "Quality reviewer. Be lenient. Only flag serious issues."

        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=prompt)
        ]

        if "agent_prompts_responses" not in state:
            state["agent_prompts_responses"] = {}
        state["agent_prompts_responses"]["evaluator"] = {
            "system_message": system_message_content,
            "prompt": prompt,
            "response": None
        }

        try:
            response = self.llm.invoke(messages)
            content = response.content

            state["agent_prompts_responses"]["evaluator"]["response"] = content

            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    result = safe_json_loads(json_str, default={"needs_refinement": False})
                except Exception as e:
                    print(f"[Evaluator] JSON parsing error: {e}")
                    result = {"needs_refinement": False}
                state["check_result"] = result
                state["needs_refinement"] = result.get("needs_refinement", False)
            else:
                state["check_result"] = {"needs_refinement": False}
                state["needs_refinement"] = False

            print(f"[Evaluator] needs_refinement: {state['needs_refinement']}")

        except Exception as e:
            print(f"[Evaluator] error: {e}")
            state["check_result"] = {"needs_refinement": False}
            state["needs_refinement"] = False

        return state
