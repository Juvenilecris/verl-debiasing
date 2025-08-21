import json
from typing import Optional, Dict, Any
import re

def computer_bbox_score(solution_str,ground_truth)->float:
    box_pattern = re.compile(r"<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>")
    match = box_pattern.search(solution_str)
    return 1.0 if match else 0.0
    

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    
    
    ground_truth = ground_truth.strip().lower() if isinstance(ground_truth, str) else ""
    if ground_truth not in ["waterbird", "landbird"]:
        # ground_truth 无法进行有效比较
        return 0.0

    try:
        # 步骤1: 尝试将VLM的输出字符串解析为Python字典
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', solution_str, re.DOTALL)
        json_str = json_match.group(1) if json_match else solution_str
        response = json.loads(json_str)
                
        
        if isinstance(response, dict):
            vlm_answer = response.get("classification")
        else:
            # 如果JSON解析出来不是一个字典（例如，是一个列表），则格式错误
            return 0.0
            

        vlm_answer = vlm_answer.strip().lower() if isinstance(vlm_answer, str) else None
        if isinstance(vlm_answer, str) and vlm_answer == ground_truth:
            return 1.0
        else:
            return 0.0
            
    except (json.JSONDecodeError, TypeError):
        return 0.0