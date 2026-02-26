# -*- coding: utf-8 -*-
"""
生成评估用 Prompts 脚本
使用 TrainerAgent 执行到 tool_call_phase，然后提取 fine_grained_reasoning 的 prompt
与 batch_process.py 100% 等价
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 如果没有 tqdm，创建一个简单的占位符
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if iterable is None:
            class FakeTqdm:
                def __init__(self, *args, **kwargs):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def update(self, n=1):
                    pass
                def set_postfix(self, **kwargs):
                    pass
            return FakeTqdm()
        return iterable

from utils import load_segment_data
from llm_config import LLMConfig, create_llm
from TrainerAgent import TrainerAgent
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage, HumanMessage as LangChainHumanMessage, AIMessage, ToolMessage


def convert_messages_to_chatml_string(messages: List) -> str:
    """
    将 LangChain 消息列表转换为 ChatML 格式的字符串
    与 ActionAgent.fine_grained_reasoning() 中使用的格式完全一致
    """
    chatml_str = ""

    for msg in messages:
        role = ""
        content = ""

        if isinstance(msg, SystemMessage):
            role = "system"
            content = msg.content
        elif isinstance(msg, LangChainHumanMessage):
            role = "user"
            content = str(msg.content)
        elif isinstance(msg, AIMessage):
            role = "assistant"
            # 处理工具调用 (Tool Calls)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls_list = [
                    {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                    for tc in msg.tool_calls
                ]
                content = json.dumps({"text": msg.content, "tool_calls": tool_calls_list}, ensure_ascii=False)
            else:
                content = msg.content
        elif isinstance(msg, ToolMessage):
            role = "tool"
            # ToolMessage 需要保存 tool_call_id
            tool_call_id = getattr(msg, 'tool_call_id', None) or getattr(msg, 'name', 'default_tool_call_id')
            tool_data = {
                "tool_call_id": tool_call_id,
                "content": msg.content
            }
            content = json.dumps(tool_data, ensure_ascii=False)

        # 拼接
        if role and content:
            chatml_str += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

    # 添加 Assistant 开头（引导模型生成）
    chatml_str += "<|im_start|>assistant\n"

    return chatml_str


def extract_fine_grained_prompt_from_state(state: Dict[str, Any], action_agent) -> str:
    """
    从 state 中提取 fine_grained_reasoning 的 prompt
    使用 ActionAgent.fine_grained_reasoning() 的完全相同逻辑
    """
    messages = state.get("tool_call_conversation", [])
    if not messages:
        raise ValueError("tool_call_conversation 为空")

    # 使用 ActionAgent 的完全相同逻辑构建 prompt
    # 这里我们调用 ActionAgent.fine_grained_reasoning() 的逻辑，但不执行 LLM 调用
    # 我们需要手动构建 final_answer_prompt
    
    # 1. 分析使用的工具（与 ActionAgent.fine_grained_reasoning() 完全一致）
    used_tool_names = set()
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                used_tool_names.add(tc['name'])
    
    # 2. 动态构建工具描述（与 ActionAgent.fine_grained_reasoning() 完全一致）
    from analysis_tools import get_available_tools_description
    full_desc = get_available_tools_description()
    filtered_tool_desc = "### Logic of Tools Used in Previous Steps:\n"
    
    tool_blocks = full_desc.split("\n\n")
    for block in tool_blocks:
        for name in used_tool_names:
            if f"**{name}" in block or f"{name}(" in block:
                filtered_tool_desc += block + "\n\n"
                break
    
    # 3. 获取领域知识（与 ActionAgent.fine_grained_reasoning() 完全一致）
    dataset_type = None
    global_view_path = state.get("global_view_path", "")
    if global_view_path:
        detected_type = action_agent.knowledge_base.detect_dataset_type(folder_path=global_view_path)
        if detected_type != "UNKNOWN":
            dataset_type = detected_type
    
    fine_grained_knowledge = action_agent.knowledge_base.get_agent_knowledge(
        agent_type="fine_grained_agent",
        dataset_type=dataset_type
    )
    
    # 4. 构建 final_answer_prompt（与 ActionAgent.fine_grained_reasoning() 完全一致）
    final_answer_prompt = f"""
You have received the tool call results in the conversation history above. 
(DO NOT CALL TOOL AGAIN)
Now provide your concise reasoning and final answer based on the analysis plan, the historical tool results, and the raw time series data.
{filtered_tool_desc}

{fine_grained_knowledge if fine_grained_knowledge else "Follow standard time series anomaly detection rules."}

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
    
    # 5. 将指令追加到消息列表（与 ActionAgent.fine_grained_reasoning() 完全一致）
    messages_with_prompt = messages + [HumanMessage(content=final_answer_prompt)]
    
    # 6. 转换为 ChatML 格式
    chatml_prompt = convert_messages_to_chatml_string(messages_with_prompt)
    
    return chatml_prompt, dataset_type


def find_all_samples(data_folder: str) -> List[str]:
    """查找所有样本文件夹"""
    data_path = Path(data_folder)
    samples = []
    
    for item in data_path.iterdir():
        if item.is_dir():
            csv_file = item / "segment_data.csv"
            segment_img = item / "segment_clean.jpg"
            extended_img = item / "extended_clean.jpg"
            
            if csv_file.exists() and segment_img.exists() and extended_img.exists():
                samples.append(str(item))
    
    samples.sort()
    return samples


def process_single_sample(
    segment_folder: str,
    trainer_agent: TrainerAgent,
) -> Optional[Dict[str, Any]]:
    """处理单个样本，生成评估用的 prompt（与 batch_process.py 100% 等价）"""
    try:
        segment_folder_path = Path(segment_folder).resolve()
        sample_name = segment_folder_path.name
        
        # 加载数据（与 batch_process.py 完全一致）
        from TrainerAgent import load_segment_data
        segment_data = load_segment_data(segment_folder_path)
        ground_truth = segment_data["ground_truth"]
        
        # 初始化状态（与 batch_process.py 完全一致）
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
            "fine_anomaly_intervals": [],
            "fine_anomaly_types": [],
            "explanations": [],
            "confidences": [],
            "fine_grained_conversation_history": [],
            "tool_call_conversation": [],
            "check_result": {},
            "needs_refinement": False,
            "refinement_count": 0,
            "agent_prompts_responses": {},
            "has_error": False,
            "error_message": "",
            "final_output": {}
        }
        
        # 构建工作流（与 batch_process.py 完全一致）
        workflow = trainer_agent.build_workflow()
        
        # 执行到 tool_call_phase（与 batch_process.py 完全一致，但不执行 fine_grained_reasoning）
        # 我们需要手动执行各个阶段
        state = trainer_agent.load_images(initial_state)
        state = trainer_agent.localization(state)
        if state.get("has_error"):
            print(f"✗ Localization 阶段失败: {sample_name}")
            return None
        
        state = trainer_agent.planning(state)
        if state.get("has_error"):
            print(f"✗ Planning 阶段失败: {sample_name}")
            return None
        
        state = trainer_agent.tool_call_phase(state)
        if state.get("has_error"):
            print(f"✗ Tool Call 阶段失败: {sample_name}")
            return None
        
        # 提取 fine_grained_reasoning 的 prompt（使用 ActionAgent 的完全相同逻辑）
        chatml_prompt, dataset_type = extract_fine_grained_prompt_from_state(state, trainer_agent.action_agent)
        
        # 处理 Ground Truth（确保是列表格式）
        import numpy as np
        
        def convert_ground_truth_to_list(gt):
            """递归转换 ground_truth 为列表格式"""
            if isinstance(gt, np.ndarray):
                if gt.size == 0:
                    return []
                gt = gt.tolist()
            
            if isinstance(gt, (list, tuple)):
                result = []
                for item in gt:
                    if isinstance(item, np.ndarray):
                        item = item.tolist()
                    
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in item[:2]):
                            result.append([int(item[0]), int(item[1])])
                        else:
                            converted = convert_ground_truth_to_list(item)
                            if isinstance(converted, list):
                                result.extend(converted)
                    elif isinstance(item, (int, float, np.integer, np.floating)):
                        result.append([int(item), int(item)])
                    elif isinstance(item, (list, tuple)):
                        converted = convert_ground_truth_to_list(item)
                        if isinstance(converted, list):
                            result.extend(converted)
                return result
            elif isinstance(gt, (int, float, np.integer, np.floating)):
                return [int(gt), int(gt)]
            else:
                return []
        
        ground_truth = convert_ground_truth_to_list(ground_truth)
        
        # 保存 segment_data.csv 的路径
        segment_data_file = segment_folder_path / "segment_data.csv"
        
        # 返回结果
        return {
            "sample_name": sample_name,
            "segment_folder": str(segment_folder),
            "segment_data_file": str(segment_data_file),
            "prompt": chatml_prompt,
            "dataset_type": dataset_type,
            "ground_truth": ground_truth,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"✗ 处理失败 {segment_folder}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _process_worker(segment_folder: str, trainer_agent: TrainerAgent) -> Optional[Dict[str, Any]]:
    """并行处理的 worker 函数"""
    try:
        return process_single_sample(segment_folder, trainer_agent)
    except Exception as e:
        print(f"✗ Worker 处理失败 {segment_folder}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_eval_prompts(
    data_folder: str,
    output_parquet: str,
    llm_config: LLMConfig,
    localization_config: Optional[LLMConfig] = None,
    planning_config: Optional[LLMConfig] = None,
    toolcall_config: Optional[LLMConfig] = None,
    max_samples: Optional[int] = None,
    start_from: Optional[int] = None,
    skip_existing: bool = True,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """
    生成评估用 Prompts（与 batch_process.py 100% 等价）
    """
    print("=" * 80)
    print("生成评估用 Prompts（与 batch_process.py 100% 等价）")
    print("=" * 80)
    
    localization_config = localization_config or llm_config
    planning_config = planning_config or llm_config
    toolcall_config = toolcall_config or llm_config
    
    print(f"\n配置信息:")
    print(f"  Localization Config: {localization_config.model_name}")
    print(f"  Planning Config: {planning_config.model_name}")
    print(f"  Toolcall Config: {toolcall_config.model_name}")
    
    # 创建 TrainerAgent（与 batch_process.py 完全一致）
    # 注意：即使 enable_checking=False，TrainerAgent 仍需要 checking_config（create_llm 不接受 None）
    # 我们提供一个默认的 checking_config
    checking_config = llm_config  # 使用默认配置，虽然不会使用
    # 同样，fine_grained_config 也需要提供（虽然我们不会执行推理，但 TrainerAgent 需要创建 LLM）
    fine_grained_config = llm_config  # 使用默认配置，虽然不会使用
    
    trainer_agent = TrainerAgent(
        max_turns=3,
        localization_config=localization_config,
        planning_config=planning_config,
        checking_config=checking_config,
        toolcall_config=toolcall_config,
        fine_grained_config=fine_grained_config,  # 提供默认配置，虽然不会使用
        training_mode=False,
        enable_checking=False
    )
    
    # 查找所有样本
    print(f"\n扫描文件夹: {data_folder}")
    all_samples = find_all_samples(data_folder)
    total_samples = len(all_samples)
    print(f"找到 {total_samples} 个样本")
    
    if start_from:
        all_samples = all_samples[start_from:]
        print(f"从第 {start_from + 1} 个样本开始处理")
    
    if max_samples:
        all_samples = all_samples[:max_samples]
        print(f"限制处理 {max_samples} 个样本")
    
    # 检查已处理的样本
    output_path = Path(output_parquet)
    existing_samples = set()
    
    if skip_existing and output_path.exists():
        try:
            existing_df = pd.read_parquet(output_parquet)
            if "sample_name" in existing_df.columns:
                existing_samples = set(existing_df["sample_name"].tolist())
                print(f"检测到已存在的输出文件: {output_parquet}")
                print(f"已处理的样本数量: {len(existing_samples)}")
        except Exception as e:
            print(f"警告: 读取已存在文件失败: {e}，将重新生成")
            existing_samples = set()
    
    # 过滤出未处理的样本
    samples_to_process = []
    skipped_count = 0
    
    for sample_folder in all_samples:
        sample_name = Path(sample_folder).name
        if skip_existing and sample_name in existing_samples:
            skipped_count += 1
            continue
        samples_to_process.append(sample_folder)
    
    if skipped_count > 0:
        print(f"跳过已处理的样本: {skipped_count} 个")
    print(f"需要处理的样本: {len(samples_to_process)} 个")
    
    if not samples_to_process:
        print("没有需要处理的新样本")
        return {
            "total_samples": total_samples,
            "processed_samples": 0,
            "skipped_count": skipped_count,
            "success_count": 0,
            "error_count": 0
        }
    
    # 加载已有结果
    existing_results = []
    if output_path.exists():
        try:
            existing_df = pd.read_parquet(output_parquet)
            existing_results = existing_df.to_dict('records')
        except:
            existing_results = []
    
    # 并行处理
    new_results = []
    results_lock = Lock()
    success_count = 0
    error_count = 0
    completed_count = 0
    
    print(f"\n开始处理 {len(samples_to_process)} 个样本，使用 {max_workers} 个线程...")
    
    # 创建进度条
    pbar = tqdm(total=len(samples_to_process), desc="生成 Prompts", unit="样本")
    
    if max_workers > 1 and len(samples_to_process) > 1:
        # 注意：TrainerAgent 可能不是线程安全的，每个线程需要创建自己的实例
        def create_worker_agent():
            return TrainerAgent(
                max_turns=3,
                localization_config=localization_config,
                planning_config=planning_config,
                checking_config=checking_config,
                toolcall_config=toolcall_config,
                fine_grained_config=fine_grained_config,
                training_mode=False,
                enable_checking=False
            )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {}
            for sample_folder in samples_to_process:
                worker_agent = create_worker_agent()
                future = executor.submit(_process_worker, sample_folder, worker_agent)
                future_to_sample[future] = sample_folder
            
            for future in as_completed(future_to_sample):
                sample_folder = future_to_sample[future]
                sample_name = Path(sample_folder).name
                try:
                    res = future.result()
                    with results_lock:
                        completed_count += 1
                        if res:
                            new_results.append(res)
                            success_count += 1
                        else:
                            error_count += 1
                        # 更新进度条
                        pbar.update(1)
                        pbar.set_postfix({
                            "成功": success_count,
                            "失败": error_count,
                            "当前": sample_name[:30] + "..." if len(sample_name) > 30 else sample_name
                        })
                except Exception as e:
                    with results_lock:
                        completed_count += 1
                        error_count += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "成功": success_count,
                        "失败": error_count,
                        "错误": str(e)[:30]
                    })
    else:
        # 单线程处理
        for i, sample_folder in enumerate(samples_to_process, 1):
            sample_name = Path(sample_folder).name
            res = _process_worker(sample_folder, trainer_agent)
            completed_count = i
            if res:
                new_results.append(res)
                success_count += 1
            else:
                error_count += 1
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "成功": success_count,
                "失败": error_count,
                "当前": sample_name[:30] + "..." if len(sample_name) > 30 else sample_name
            })
    
    # 关闭进度条
    pbar.close()
    
    # 保存结果
    all_results = existing_results + new_results
    
    if all_results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df = pd.DataFrame(all_results)
        
        # 保存 Parquet
        result_df.to_parquet(output_parquet, index=False, engine='pyarrow')
        
        # 同时保存 CSV
        output_csv = str(output_path.with_suffix('.csv'))
        result_df.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"\n✓ 结果已保存到:")
        print(f"  Parquet: {output_parquet}")
        print(f"  CSV: {output_csv}")
        print(f"  总数据量: {len(all_results)} 条 (已有: {len(existing_results)}, 新增: {len(new_results)})")
    
    # 返回摘要
    summary = {
        "total_samples": total_samples,
        "processed_samples": len(samples_to_process),
        "skipped_count": skipped_count,
        "success_count": success_count,
        "error_count": error_count,
        "output_file": str(output_parquet),
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n处理完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    if skipped_count > 0:
        print(f"  跳过: {skipped_count}")
    
    return summary


if __name__ == "__main__":
    # 配置
    data_folder = "./TestData/TODS"
    output_parquet = "./eval_prompts/TODS_PROMPT.parquet"
    
    # LLM配置
    grok_config = LLMConfig(
        model_name="grok-4-1-fast-non-reasoning",
        api_key="sk-WRkycVK2fsKY2z4zZlaDslWY4ZG37XLIlWg07i5LMfGy8f4b",
        base_url="https://api2.aigcbest.top/v1",
        temperature=0.3,
    )
    
    gemini_config = LLMConfig(
        model_name="gemini-2.5-flash-lite-preview-09-2025",
        api_key="sk-WRkycVK2fsKY2z4zZlaDslWY4ZG37XLIlWg07i5LMfGy8f4b",
        base_url="https://api2.aigcbest.top/v1",
        temperature=0.3,
    )
    
    # 生成 prompts
    summary = generate_eval_prompts(
        data_folder=data_folder,
        output_parquet=output_parquet,
        llm_config=grok_config,
        localization_config=grok_config,
        planning_config=grok_config,
        toolcall_config=grok_config,
        max_samples=800,
        start_from=None,
        skip_existing=True,
        max_workers=100,
    )
