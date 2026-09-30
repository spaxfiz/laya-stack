"""Candidate Ranking Task."""

import random
from typing import Any, Dict, List, Optional
from decision_bench.dataset.schema import BenchmarkSample, TaskType
from decision_bench.models.base import BaseDecisionModel
from decision_bench.metrics.ranking import hit_at_k, mean_reciprocal_rank, ndcg_at_k
from .base import BaseTask, SamplePrediction


# Standard distractor repository across domains
DISTRACTORS_BY_TRACK = {
    "科技数码": ["华为Mate70深度性能拆解与影像实测", "极客桌面打造指南：双屏生产力配置"],
    "萌宠动物": ["金毛犬幼犬科学换粮与防拆家心得", "猫咪日常驱虫避坑指南"],
    "美妆护肤": ["干敏皮换季维稳面霜深度成分测评", "早C晚A抗初老实操七天记录"],
    "穿搭时尚": ["小个子入秋显高胶囊衣橱穿搭公式", "灰色系针织衫搭配指南"],
    "美食探店": ["藏在胡同里的老字号爆肚排队实测", "免烤箱巴斯克芝士蛋糕新手必做"],
    "职场成长": ["述职答辩如何用STAR法则震撼全场", "体制内职场社交边界感法则"],
    "家居生活": ["精装房无主灯改造极简避坑经验", "阳台打造家庭绿植微森林清单"],
    "运动健康": ["减脂期低GI主食碳水循环食谱", "普拉提大器械核心肌群唤醒训练"],
    "母婴育儿": ["宝宝断奶过渡期辅食制作实操手册", "0-1岁婴儿玩具早教黑红榜"],
    "旅游出行": ["川西小环线四天三晚自驾路线与防高反攻略", "京都红叶季冷门机位徒步推荐"],
    "情感心理": ["高敏感人群如何摆脱精神内耗与情绪反刍", "亲密关系中非暴力沟通技巧"],
    "摄影剪辑": ["手机电影感调色参数：富士胶片色彩美学", "索尼微单慢门光影人像拍摄指南"]
}


class CandidateRankingTask(BaseTask):
    """
    Evaluates candidate semantic ranking against negative distractors.
    Tests model's discriminative ordering ability (NDCG@3, MRR, Top-1 Hit Rate).
    """

    def __init__(self, seed: int = 42):
        super().__init__(
            name="candidate_ranking",
            task_type=TaskType.RANK,
            description="4-Way Candidate Semantic Relevance Ranking"
        )
        self.rng = random.Random(seed)

    def _get_distractors(self, true_track: str, count: int = 3) -> List[str]:
        other_tracks = [t for t in DISTRACTORS_BY_TRACK if t != true_track]
        chosen_tracks = self.rng.sample(other_tracks, min(count, len(other_tracks)))
        return [self.rng.choice(DISTRACTORS_BY_TRACK[t]) for t in chosen_tracks]

    def evaluate_sample(
        self,
        model: BaseDecisionModel,
        sample: BenchmarkSample
    ) -> SamplePrediction:
        target_title = sample.title.strip()
        distractors = self._get_distractors(sample.ground_truth_track, count=3)
        candidates = [target_title] + distractors
        self.rng.shuffle(candidates)

        query = f"寻找符合【{sample.ground_truth_track}】领域的最佳内容笔记"
        out = model.rank(
            prompt=query,
            candidates=candidates,
            instruction="请根据内容与目标赛道的相关性对候选标题进行排序"
        )

        ranked_list = out.value if isinstance(out.value, list) else candidates
        top1 = ranked_list[0] if ranked_list else ""
        is_correct = (top1 == target_title)

        # Relevance vector for NDCG calculation (1 for true title, 0 for distractors)
        relevance_scores = [1.0 if c == target_title else 0.0 for c in ranked_list]
        ndcg3 = ndcg_at_k(relevance_scores, k=3)

        return SamplePrediction(
            sample_id=sample.id,
            context=f"Query: {query}",
            ground_truth=target_title,
            prediction=ranked_list,
            confidence=out.confidence,
            is_correct=is_correct,
            latency_ms=out.latency_ms,
            error=out.error,
            metadata={"ndcg3": ndcg3, "candidates": candidates}
        )

    def compute_metrics(
        self,
        predictions: List[SamplePrediction]
    ) -> Dict[str, Any]:
        ranked_lists = [p.prediction for p in predictions]
        targets = [p.ground_truth for p in predictions]

        top1_hit = hit_at_k(ranked_lists, targets, k=1)
        top2_hit = hit_at_k(ranked_lists, targets, k=2)
        mrr = mean_reciprocal_rank(ranked_lists, targets)

        ndcg_list = [p.metadata.get("ndcg3", 0.0) for p in predictions]
        avg_ndcg = round(sum(ndcg_list) / len(ndcg_list), 4) if ndcg_list else 0.0

        return {
            "top1_hit_rate": top1_hit,
            "top2_hit_rate": top2_hit,
            "mrr": mrr,
            "ndcg_at_3": avg_ndcg
        }
