# -*- coding: utf-8 -*-
"""
预处理脚本：生成用于VERL训练的 CSV 文件 (ChatML 格式)
包含: prompt (ChatML string), dataset_type, ground_truth
"""

import pandas as pd
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# 假设这些 utils 和 agent 都在当前目录下
from utils import load_segment_data, get_column_names
from llm_config import LLMConfig, create_llm
from Localization import Localization
from PlanningAgent import PlanningAgent
from ActionAgent import ActionAgent
from knowledge_base import KnowledgeBase
from analysis_tools import get_available_tools_description
from workflowstate import WorkflowState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 1. 核心格式转换函数：ChatML String
# -------------------------------------------------------------------------

def convert_messages_to_chatml_string(messages: List) -> str:
    """
    将 LangChain 消息列表转换为 ChatML 格式的字符串（用于训练数据保存）。
    格式:
    <|im_start|>system
    ...<|im_end|>
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>assistant
    ...<|im_end|>
    ...
    <|im_start|>assistant
    """
    chatml_str = ""

    # 处理所有消息
    for msg in messages:
        role = ""
        content = ""

        if isinstance(msg, SystemMessage):
            role = "system"
            content = msg.content
        elif isinstance(msg, HumanMessage):
            role = "user"
            content = str(msg.content)
        elif isinstance(msg, AIMessage):
            role = "assistant"
            # 处理工具调用 (Tool Calls)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # 将工具调用序列化为 JSON 字符串，以便模型理解
                tool_calls_list = [
                    {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                    for tc in msg.tool_calls
                ]
                # 这里构造一个包含 text 和 tool_calls 的 JSON 结构
                content = json.dumps({"text": msg.content, "tool_calls": tool_calls_list}, ensure_ascii=False)
            else:
                content = msg.content
        elif isinstance(msg, ToolMessage):
            role = "tool"
            # 工具返回的结果
            content = msg.content

        # 拼接
        if role and content:
            chatml_str += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

    # 添加 Assistant 开头 (引导模型生成，用于训练)
    chatml_str += "<|im_start|>assistant\n"

    return chatml_str

# -------------------------------------------------------------------------
# 2. Prompt 构建逻辑 (保持不变，用于生成 final_user_content)
# -------------------------------------------------------------------------

def build_fine_grained_prompt(state: WorkflowState, action_agent: ActionAgent) -> str:
    """构建 ActionAgent 最后一步用于推理的 Prompt 内容"""
    messages = state.get("tool_call_conversation", [])
    
    # 1. 分析使用的工具
    used_tool_names = set()
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                used_tool_names.add(tc['name'])
    
    # 2. 过滤工具描述
    full_desc = get_available_tools_description()
    filtered_tool_desc = "### Logic of Tools Used in Previous Steps:\n"
    tool_blocks = full_desc.split("\n\n")
    for block in tool_blocks:
        for name in used_tool_names:
            if f"**{name}" in block or f"{name}(" in block:
                filtered_tool_desc += block + "\n\n"
                break
    
    # 3. 获取领域知识
    dataset_type = "UNKNOWN"
    global_view_path = state.get("global_view_path", "")
    if global_view_path:
        detected_type = action_agent.knowledge_base.detect_dataset_type(folder_path=global_view_path)
        if detected_type != "UNKNOWN":
            dataset_type = detected_type
            
    fine_grained_knowledge = action_agent.knowledge_base.get_agent_knowledge(
        agent_type="fine_grained_agent",
        dataset_type=dataset_type
    )

    # 4. 构建 Prompt 文本
    final_answer_prompt = f"""
You have received the tool call results in the conversation history above. 
Now provide your concise reasoning and final answer based on the analysis plan, the historical tool results, and the raw time series data.

{filtered_tool_desc}

{fine_grained_knowledge if fine_grained_knowledge else "Follow standard time series anomaly detection rules."}

## Output format:
<think>
Your concise reasoning process here.
</think>
[
    {{
        "interval": [start, end],
        "type": "anomaly_type",
        "explanation": "reason",
        "confidence": 3
    }}
]

## IMPORTANT:
- If no anomalies detected, output exactly: []
- Do not wrap the output in markdown code blocks.
"""
    return final_answer_prompt, dataset_type

# -------------------------------------------------------------------------
# 3. Ground Truth 转换
# -------------------------------------------------------------------------

def convert_intervals_to_dict(intervals: List[List[int]], data_indices: List) -> Dict[str, int]:
    """将异常区间转换为 {index_str: 0/1} 字典"""
    gt_dict = {str(idx): 0 for idx in data_indices}
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) == 0:
            continue
        start = end = interval[0]
        if len(interval) >= 2:
            start, end = interval[0], interval[1]
        if start > end:
            start, end = end, start
        
        for idx in data_indices:
            if start <= idx <= end:
                gt_dict[str(idx)] = 1
    return gt_dict

# -------------------------------------------------------------------------
# 4. 单样本处理流程
# -------------------------------------------------------------------------

def _create_agents(llm_config: LLMConfig) -> tuple:
    """创建 Agent 实例（用于并行处理时每个线程创建自己的实例）"""
    kb = KnowledgeBase()
    localization_agent = Localization(create_llm(llm_config), kb)
    planning_agent = PlanningAgent(create_llm(llm_config), kb)
    action_agent = ActionAgent(kb, create_llm(llm_config), None)
    return localization_agent, planning_agent, action_agent

def process_single_sample(
    segment_folder: str,
    localization_agent: Localization,
    planning_agent: PlanningAgent,
    action_agent: ActionAgent
) -> Optional[Dict[str, Any]]:
    
    try:
        segment_folder_path = Path(segment_folder).resolve()
        logger.info(f"正在处理: {segment_folder_path.name}")
        
        # 加载数据
        segment_data = load_segment_data(segment_folder_path)
        ground_truth = segment_data["ground_truth"]
        
        # 初始化状态
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
            "tool_call_conversation": [],  # 保存工具调用阶段的对话历史（用于推理阶段）
            "check_result": {},
            "needs_refinement": False,
            "refinement_count": 0,
            "agent_prompts_responses": {},  # Agent的prompt和response记录（用于调试和评估）
            "has_error": False,
            "error_message": "",
            "final_output": {},
            "agent_timings": {}  # ✅ 新增：记录每个 Agent 的处理时间
        }
        
        # --- 执行 Pipeline ---
        # 1. Vision
        state = localization_agent.analyze(initial_state)
        if state.get("has_error"): return None
        
        # 2. Planning
        state = planning_agent.plan(state)
        if state.get("has_error"): return None
        
        # 3. Action (Tools)
        state = action_agent.tool_call_phase(state)
        if state.get("has_error"): return None
        
        # --- 构建输出数据 ---
        
        # A. 生成最终指令内容 和 Dataset Type
        final_prompt_content, dataset_type = build_fine_grained_prompt(state, action_agent)
        
        # B. 将 final_prompt_content 追加到消息列表（类似于 ActionAgent.py 第 267 行的处理方式）
        messages = state.get("tool_call_conversation", [])
        # 将 final_prompt_content 作为 HumanMessage 追加到消息列表
        messages.append(HumanMessage(content=final_prompt_content))
        # 将完整的消息列表转换为 ChatML 格式字符串（用于保存到 CSV）
        chatml_prompt = convert_messages_to_chatml_string(messages)
        
        # C. 处理 Ground Truth
        data = state["data"]
        timestamp_col, _ = get_column_names(data)
        data_indices = data[timestamp_col].tolist() if timestamp_col in data.columns else list(range(len(data)))
        gt_dict = convert_intervals_to_dict(ground_truth, data_indices)
        
        # 返回扁平化的一行数据
        return {
            "prompt": chatml_prompt,  # 使用完整的 ChatML 格式字符串
            "dataset_type": dataset_type,
            "ground_truth": json.dumps(gt_dict),  # 存为 JSON 字符串
            "segment_folder": segment_folder  # 添加 segment_folder 字段（用于日志标识）
        }
        
    except Exception as e:
        logger.error(f"处理失败 {segment_folder}: {e}", exc_info=True)
        return None

def _process_worker(segment_folder: str, llm_config: LLMConfig) -> Optional[Dict[str, Any]]:
    """并行处理的 worker 函数，每个线程创建自己的 Agent 实例"""
    try:
        localization_agent, planning_agent, action_agent = _create_agents(llm_config)
        return process_single_sample(segment_folder, localization_agent, planning_agent, action_agent)
    except Exception as e:
        logger.error(f"Worker 处理失败 {segment_folder}: {e}", exc_info=True)
        return None

def _save_results(
    all_results: List[Dict[str, Any]], 
    output_parquet: str, 
    output_csv: Optional[str] = None
):
    """保存结果到 Parquet 和 CSV 文件"""
    if not all_results:
        return
    
    result_df = pd.DataFrame(all_results)
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 Parquet
    result_df.to_parquet(output_parquet, index=False, engine='pyarrow')
    
    # 同步保存 CSV
    if output_csv is None:
        output_csv = str(output_path.with_suffix('.csv'))
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    logger.info(f"已保存结果: Parquet={output_parquet}, CSV={output_csv}, 共 {len(result_df)} 条")

# -------------------------------------------------------------------------
# 5. 主程序
# -------------------------------------------------------------------------

def main(
    input_parquet: str, 
    output_parquet: str, 
    max_samples: int = 5,
    num_workers: int = 4,
    save_interval: int = 5
):
    """
    主函数：并行处理样本并同步生成 Parquet 和 CSV 文件
    
    Args:
        input_parquet: 输入任务列表 Parquet 文件
        output_parquet: 输出 Parquet 文件路径
        max_samples: 最大处理样本数（None 表示处理所有）
        num_workers: 并行处理的线程数
        save_interval: 每处理多少个样本后保存一次（0 表示只在最后保存）
    """
    # 配置
    llm_config = LLMConfig(
        model_name="grok-4-1-fast-non-reasoning",
        api_key="sk-WRkycVK2fsKY2z4zZlaDslWY4ZG37XLIlWg07i5LMfGy8f4b",
        base_url="https://api2.aigcbest.top/v1",
        temperature=0.3,
    )

    # 读取任务列表
    df = pd.read_parquet(input_parquet)
    if max_samples:
        df = df.head(max_samples)
    
    # 检查输出文件是否存在，如果存在则加载已处理的样本
    processed_segments = set()
    existing_results = []
    output_path = Path(output_parquet)
    output_csv = str(output_path.with_suffix('.csv'))
    
    if output_path.exists():
        try:
            existing_df = pd.read_parquet(output_parquet)
            if "segment_folder" in existing_df.columns:
                processed_segments = set(existing_df["segment_folder"].tolist())
                existing_results = existing_df.to_dict('records')
                logger.info(f"检测到已存在的输出文件: {output_parquet}")
                logger.info(f"已处理的样本数量: {len(processed_segments)}")
        except Exception as e:
            logger.warning(f"读取已存在文件失败: {e}，将重新生成")
            processed_segments = set()
            existing_results = []
    
    # 过滤出未处理的样本
    pending_segments = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        segment_folder = row.get("segment_folder")
        if not segment_folder: 
            continue
        
        # 检查是否已处理
        if segment_folder in processed_segments:
            logger.info(f"跳过已处理的样本: {segment_folder}")
            skipped_count += 1
            continue
        
        pending_segments.append(segment_folder)
    
    if not pending_segments:
        logger.info("没有需要处理的新样本")
        return
    
    logger.info(f"开始并行处理 {len(pending_segments)} 个样本，使用 {num_workers} 个线程")
    
    # 并行处理
    new_results = []
    results_lock = Lock()
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_segment = {
            executor.submit(_process_worker, seg, llm_config): seg 
            for seg in pending_segments
        }
        
        # 收集结果
        for future in as_completed(future_to_segment):
            segment_folder = future_to_segment[future]
            try:
                res = future.result()
                if res:
                    should_save = False
                    with results_lock:
                        new_results.append(res)
                        completed_count += 1
                        logger.info(f"[{completed_count}/{len(pending_segments)}] 成功处理: {segment_folder}")
                        
                        # 检查是否需要定期保存
                        if save_interval > 0 and completed_count % save_interval == 0:
                            should_save = True
                            current_results = existing_results + new_results.copy()
                    
                    # 在锁外进行保存操作（避免阻塞其他线程）
                    if should_save:
                        _save_results(current_results, output_parquet, output_csv)
                else:
                    with results_lock:
                        completed_count += 1
                        current_count = completed_count
                    logger.warning(f"[{current_count}/{len(pending_segments)}] 处理失败: {segment_folder}")
            except Exception as e:
                with results_lock:
                    completed_count += 1
                    current_count = completed_count
                logger.error(f"[{current_count}/{len(pending_segments)}] 处理异常 {segment_folder}: {e}", exc_info=True)
    
    # 最终保存所有结果
    all_results = existing_results + new_results
    
    if all_results:
        _save_results(all_results, output_parquet, output_csv)
        logger.info(f"处理完成！总数据量: {len(all_results)} 条 (已有: {len(existing_results)}, 新增: {len(new_results)}, 跳过: {skipped_count})")
    else:
        logger.warning("未生成任何有效数据")

if __name__ == "__main__":
    main(
        input_parquet="WSD_PRO.parquet",
        output_parquet="WSD_PRO_GLOBAL.parquet",
        max_samples=1200,
        num_workers=50,  # 并行线程数
        save_interval=5  # 每处理 5 个样本保存一次（0 表示只在最后保存）
    )