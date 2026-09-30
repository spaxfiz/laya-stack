"""Ranking metrics for DecisionBench."""

import math
from typing import Any, List, Sequence


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain at position k."""
    dcg = 0.0
    for i in range(min(len(relevance_scores), k)):
        rel = relevance_scores[i]
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
    return dcg


def ndcg_at_k(relevance_scores: Sequence[float], k: int = 3) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    if not relevance_scores or k <= 0:
        return 0.0

    actual_dcg = dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0.0:
        return 0.0
    return round(actual_dcg / ideal_dcg, 4)


def mean_reciprocal_rank(ranked_items_list: List[List[Any]], relevant_items: List[Any]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    if not ranked_items_list or len(ranked_items_list) != len(relevant_items):
        return 0.0

    reciprocal_ranks = []
    for ranked, rel in zip(ranked_items_list, relevant_items):
        try:
            rank = ranked.index(rel) + 1  # 1-based
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def hit_at_k(ranked_items_list: List[List[Any]], relevant_items: List[Any], k: int = 1) -> float:
    """Calculate Hit Rate @ K."""
    if not ranked_items_list or len(ranked_items_list) != len(relevant_items):
        return 0.0

    hits = 0
    for ranked, rel in zip(ranked_items_list, relevant_items):
        if rel in ranked[:k]:
            hits += 1
    return round(hits / len(ranked_items_list), 4)
