from knowledge_base import KnowledgeBase
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
import base64
from utils import get_column_names,safe_json_loads
from workflowstate import WorkflowState

class Localization:
    
    def __init__(self, llm=None,knowledge_base: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.knowledge_base = knowledge_base or KnowledgeBase()
    
    def analyze(self, state: WorkflowState) -> WorkflowState:
        with open(state["local_view_path"], "rb") as f:
            local_image = base64.b64encode(f.read()).decode('utf-8')
        
        data = state["data"]
        timestamp_col, _ = get_column_names(data)
        if timestamp_col in data.columns:
            min_idx = int(data[timestamp_col].iloc[0])
            max_idx = int(data[timestamp_col].iloc[-1])
            data_range_info = f"\n**Important**: The time series data you need to analyze covers index range [{min_idx}, {max_idx}]. When identifying anomalies, use Index values within this range.\n"
        else:
            data_range_info = ""
        
        local_view_path = state.get("local_view_path", "")
        dataset_type = self.knowledge_base.detect_dataset_type(folder_path=local_view_path)
        localization_knowledge = self.knowledge_base.get_agent_knowledge("localization_agent", dataset_type)
        
        if dataset_type != "UNKNOWN":
            print(f"[Localization] dataset type: {dataset_type}")
        
        prompt = f"""
You are a time series anomaly detection expert. Please analyze the following time series image (local view showing the segment to analyze).

{data_range_info}

You need to analyze those 100 time points, using the values from the horizontal and vertical axes. Put them inside the <description> tag. Consider whether it is noisy or periodic.

## Output format:
<description>
...
</description>
{{
    "anomaly_intervals": [[start1, end1], [start2, end2],...],
    "anomaly_types": ["anomaly_type1", "anomaly_type2", ...],
    "anomaly_reasons": ["reason1", "reason2", ...]
}}

## Here is the information you should particular focus
{localization_knowledge}

## IMPORTANT: 
- When you identify anomaly intervals, use the Index values shown in the image (read from the horizontal axis). Make sure the Index values you return are within the data range mentioned above.
- Anomalies must be significant deviations FROM the established pattern, **not the pattern itself**.
- Only return obvious anomalies, avoid over-detection.
"""
        
        messages = [
            SystemMessage(content="You are a professional time series anomaly detection expert."),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{local_image}"}
                }
            ])
        ]
        
        if "agent_prompts_responses" not in state:
            state["agent_prompts_responses"] = {}
        state["agent_prompts_responses"]["localization"] = {
            "system_message": "You are a professional time series anomaly detection expert.",
            "prompt": prompt,
            "response": None
        }
        try:
            response = self.llm.invoke(messages)
            import re
            content = response.content
            
            state["agent_prompts_responses"]["localization"]["response"] = content
            
            description_match = re.search(r'<description>(.*?)</description>', content, re.DOTALL)
            if description_match:
                visual_description = description_match.group(1).strip()
                state["visual_description"] = visual_description
            else:
                state["visual_description"] = ""
                print(f"[Localization] Warning: no <description> tags")
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    result = safe_json_loads(json_str, default={"anomaly_intervals": [], "anomaly_types": [], "anomaly_reasons": []})
                except Exception as e:
                    print(f"[Localization] JSON parsing error: {e}")
                    result = {"anomaly_intervals": [], "anomaly_types": [], "anomaly_reasons": []}
                
                state["localization_anomaly_intervals"] = result.get("anomaly_intervals", [])
                state["localization_anomaly_types"] = result.get("anomaly_types", [])
                state["localization_anomaly_reasons"] = result.get("anomaly_reasons", [])
                
                intervals_count = len(state["localization_anomaly_intervals"])
                reasons_count = len(state["localization_anomaly_reasons"])
                if reasons_count < intervals_count:
                    state["localization_anomaly_reasons"].extend([""] * (intervals_count - reasons_count))
                elif reasons_count > intervals_count:
                    state["localization_anomaly_reasons"] = state["localization_anomaly_reasons"][:intervals_count]
            else:
                state["localization_anomaly_intervals"] = []
                state["localization_anomaly_types"] = []
                state["localization_anomaly_reasons"] = []
        except Exception as e:
            error_str = str(e)
            print(f"[Localization] error: {error_str}")
            
            if "402" in error_str or "Insufficient Balance" in error_str or "401" in error_str or "403" in error_str:
                state["has_error"] = True
                state["error_message"] = f"API error: {error_str}"
                print(f"[Localization] API error")
            
            state["localization_anomaly_intervals"] = []
            state["localization_anomaly_types"] = []
            state["visual_description"] = ""
        
        return state
