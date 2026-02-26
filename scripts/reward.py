# -*- coding: utf-8 -*-
from typing import List, Set

def intervals_to_set(intervals: List[List[int]]) -> Set[int]:
    point_set = set()
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) == 0:
            continue
        
        if len(interval) == 1:
            point = int(interval[0])
            point_set.add(point)
        elif len(interval) >= 2:
            start = int(interval[0])
            end = int(interval[1])
            if start > end:
                start, end = end, start
            point_set.update(range(start, end + 1))
    
    return point_set


def reward(anomaly_intervals: List[List[int]], 
           ground_truth: List[List[int]]) -> float:
    if not isinstance(anomaly_intervals, list):
        anomaly_intervals = []
    if not isinstance(ground_truth, list):
        ground_truth = []
    
    predicted_points = intervals_to_set(anomaly_intervals)
    true_points = intervals_to_set(ground_truth)
    
    if len(true_points) == 0:
        if len(predicted_points) == 0:
            return 1.0
        reward = 1.0 / (len(predicted_points) + 1.0)
        return float(reward)
    
    tp = len(predicted_points & true_points)
    fp = len(predicted_points - true_points)
    fn = len(true_points - predicted_points)
    
    if tp + fp == 0:
        precision = 0.0  
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        return 0.0  
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(f1)