# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Advanced logit-based reward function for image classification task.
This reward function computes rewards based on the probability of ground truth tokens
in the VLM's predicted logits at specific positions.
"""

import json
import re
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple


class TokenMappingManager:
    """Manages mapping between class names and token IDs."""
    
    def __init__(self, tokenizer, class_names: List[str]):
        """
        Initialize token mapping manager.
        
        Args:
            tokenizer: HuggingFace tokenizer
            class_names: List of class names
        """
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.class_to_tokens = {}
        self._build_mapping()
    
    def _build_mapping(self):
        """Build mapping from class names to token IDs."""
        for class_name in self.class_names:
            # Tokenize the class name
            tokens = self.tokenizer.encode(class_name, add_special_tokens=False)
            self.class_to_tokens[class_name] = tokens
    
    def get_class_tokens(self, class_name: str) -> List[int]:
        """Get token IDs for a given class name."""
        return self.class_to_tokens.get(class_name, [])
    
    def get_class_probability(self, logits: torch.Tensor, class_name: str, 
                            position: int = -1) -> torch.Tensor:
        """
        Get probability of a class name at a specific position.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            class_name: Class name to compute probability for
            position: Position in sequence to compute probability at
            
        Returns:
            Probability tensor [batch_size]
        """
        if class_name not in self.class_to_tokens:
            return torch.zeros(logits.size(0), device=logits.device)
        
        class_tokens = self.class_to_tokens[class_name]
        if not class_tokens:
            return torch.zeros(logits.size(0), device=logits.device)
        
        # Get logits at the specified position
        pos_logits = logits[:, position, :]  # [batch_size, vocab_size]
        
        # Compute probability for the first token of the class name
        first_token_id = class_tokens[0]
        probs = F.softmax(pos_logits, dim=-1)
        token_probs = probs[:, first_token_id]  # [batch_size]
        
        return token_probs


def extract_json_from_response(response_str: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from the response string."""
    try:
        # Find JSON pattern in the response
        json_pattern = r'\{[^}]*\}'
        json_match = re.search(json_pattern, response_str)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        return None
    except (json.JSONDecodeError, AttributeError):
        return None


def compute_advanced_logit_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    """
    Compute advanced logit-based reward for image classification task.
    
    Args:
        data_source: Dataset source identifier
        solution_str: Model's response string
        ground_truth: Ground truth class label
        extra_info: Additional information including logits and tokenizer
        **kwargs: Additional keyword arguments
    
    Returns:
        float: Reward value between 0 and 1
    """
    # Extract predicted class from response
    predicted_json = extract_json_from_response(solution_str)
    if predicted_json is None:
        return 0.0  # No valid JSON found
    
    predicted_class = predicted_json.get('class', '')
    if not predicted_class:
        return 0.0  # No class field found
    
    # Check if we have logits information
    if extra_info and 'logits' in extra_info and 'tokenizer' in extra_info:
        return compute_token_level_reward(
            extra_info['logits'],
            extra_info['tokenizer'],
            ground_truth,
            predicted_class,
            extra_info.get('class_names', [ground_truth, predicted_class]),
            **kwargs
        )
    else:
        # Fallback to simple reward
        if ground_truth == predicted_class:
            return 1.0
        else:
            return 0.0


def compute_token_level_reward(
    logits: torch.Tensor,
    tokenizer,
    ground_truth: str,
    predicted_class: str,
    class_names: List[str],
    temperature: float = 1.0,
    position_weight: float = 0.8,
    **kwargs
) -> float:
    """
    Compute token-level reward based on logits.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        tokenizer: HuggingFace tokenizer
        ground_truth: Ground truth class label
        predicted_class: Predicted class label
        class_names: List of all possible class names
        temperature: Temperature for softmax computation
        position_weight: Weight for position-specific rewards
        **kwargs: Additional keyword arguments
    
    Returns:
        float: Reward value between 0 and 1
    """
    try:
        # Initialize token mapping manager
        token_manager = TokenMappingManager(tokenizer, class_names)
        
        # Convert logits to probabilities
        probs = F.softmax(logits / temperature, dim=-1)
        
        # Compute reward based on ground truth class probability
        gt_prob = token_manager.get_class_probability(logits, ground_truth, position=-1)
        
        # Compute reward based on predicted class probability
        pred_prob = token_manager.get_class_probability(logits, predicted_class, position=-1)
        
        # Base reward: 1.0 if prediction is correct, 0.0 otherwise
        base_reward = 1.0 if ground_truth == predicted_class else 0.0
        
        # Logit-based reward: higher probability of ground truth class gets higher reward
        logit_reward = gt_prob.mean().item()
        
        # Combine base reward with logit-based reward
        final_reward = (1 - position_weight) * base_reward + position_weight * logit_reward
        
        return max(0.0, min(1.0, final_reward))  # Clamp to [0, 1]
        
    except Exception as e:
        print(f"Error computing token-level reward: {e}")
        # Fallback to simple reward
        if ground_truth == predicted_class:
            return 1.0
        else:
            return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    """
    Main reward function interface for VERL.
    
    Args:
        data_source: Dataset source identifier
        solution_str: Model's response string
        ground_truth: Ground truth class label
        extra_info: Additional information including logits if available
        **kwargs: Additional keyword arguments
    
    Returns:
        float: Reward value between 0 and 1
    """
    return compute_advanced_logit_reward(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs
    )


def compute_score_with_position_analysis(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    """
    Alternative reward function that analyzes multiple positions.
    
    Args:
        data_source: Dataset source identifier
        solution_str: Model's response string
        ground_truth: Ground truth class label
        extra_info: Additional information including logits if available
        **kwargs: Additional keyword arguments
    
    Returns:
        float: Reward value between 0 and 1
    """
    if not extra_info or 'logits' not in extra_info:
        return compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    
    try:
        logits = extra_info['logits']
        tokenizer = extra_info.get('tokenizer')
        class_names = extra_info.get('class_names', [ground_truth])
        
        if tokenizer is None:
            return compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
        
        # Analyze multiple positions
        positions = [-1, -2, -3]  # Last few positions
        position_rewards = []
        
        for pos in positions:
            if abs(pos) <= logits.size(1):  # Check if position is valid
                pos_logits = logits[:, pos, :]
                token_manager = TokenMappingManager(tokenizer, class_names)
                gt_prob = token_manager.get_class_probability(logits, ground_truth, position=pos)
                position_rewards.append(gt_prob.mean().item())
        
        # Use average of position rewards
        if position_rewards:
            avg_position_reward = sum(position_rewards) / len(position_rewards)
        else:
            avg_position_reward = 0.0
        
        # Combine with base reward
        predicted_json = extract_json_from_response(solution_str)
        predicted_class = predicted_json.get('class', '') if predicted_json else ''
        base_reward = 1.0 if ground_truth == predicted_class else 0.0
        
        final_reward = 0.7 * base_reward + 0.3 * avg_position_reward
        return max(0.0, min(1.0, final_reward))
        
    except Exception as e:
        print(f"Error in position analysis reward: {e}")
        return compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
