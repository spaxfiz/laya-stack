import urllib.request
import json
import numpy as np
import re
from laya_stack.server import predict_track_bayesian, get_agent, MORPHOLOGICAL_PATTERNS, XHS_TRACK_LEXICON

agent = get_agent()
categories = list(XHS_TRACK_LEXICON.keys())
OMLX_URL = "http://localhost:8090/v1/embeddings"
API_KEY = "123456"

def get_embeddings(texts):
    req = urllib.request.Request(
        OMLX_URL,
        data=json.dumps({"model": "laya-multilingual-mlx", "input": texts}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return np.array([item["embedding"] for item in data["data"]])

cat_embs = get_embeddings(categories)

def predict_omlx(title, snippet=None):
    text_lower = (title + " " + (snippet or "")).lower()
    priors = {c: 0.0 for c in categories}
    for cat, regexes in MORPHOLOGICAL_PATTERNS.items():
        for r in regexes:
            if re.search(r, text_lower):
                priors[cat] += 3.5
    if any(w in text_lower for w in ["chatgpt", "gpt", "claude", "openai", "deepseek", "astra", "tibo", "antigravity", "codex", "sol6.1", "sol", "mac", "大模型", "人工智能", "llm"]):
        priors["科技数码"] += 6.0
        priors["萌宠动物"] = -5.0
    if re.search(r"苹果.*?(跳号|发布|系统|手机|官网|耳机|手表|芯片|平板|开发者|发布会|更新)", text_lower):
        priors["科技数码"] += 6.0
        priors["美食探店"] = -5.0
    if any(w in text_lower for w in ["机械", "试驾", "手工耿", "油耗子", "被封", "订阅", "四驱车", "三级头", "航天飞机", "地效"]):
        priors["科技数码"] += 6.0
        priors["萌宠动物"] = -5.0
    if "菌子" in text_lower or "吃吃喝喝" in text_lower or "胃根本装不下" in text_lower:
        priors["美食探店"] += 5.0
        priors["萌宠动物"] = -5.0
        priors["旅游出行"] = max(0.0, priors["旅游出行"] - 3.0)
    if any(w in text_lower for w in ["写真", "旅拍", "陪拍", "约拍", "跟拍", "互拍", "拍摄技巧"]):
        priors["摄影剪辑"] += 5.0

    state = f"小红书笔记标题: {title.strip()}\n正文片段: {snippet.strip()}" if snippet and snippet.strip() else f"小红书笔记标题: {title.strip()}"
    t_emb = get_embeddings([state])[0]
    sims = np.dot(cat_embs, t_emb)
    emb_logits = sims * 10.0
    fused_logits = emb_logits + np.array([priors[c] for c in categories]) * 1.5
    fused_p = np.exp(fused_logits - np.max(fused_logits))
    fused_p /= np.sum(fused_p)
    return categories[int(np.argmax(fused_p))], float(np.max(fused_p))

test_cases = [
    ("富士山🗻初雪", None),
    ("开高速全程90码会被骂吗？", None),
    ("今日大模型迷思", None),
    ("苹果再一次跳号了！", None),
    ("麦理浩径全段徒步避坑攻略，千万别轻敌！", "风景绝美，但难度真的很大，需要带足水"),
    ("买个这玩意花了1.4w，被家里人骂惨了……", "为了去川西自驾徒步买的高原重装帐篷和全套户外装备")
]

print("=" * 80)
print(f"{'TITLE':<32} | {'LAYA NATIVE':<18} | {'oMLX EMBEDDING API':<18} | MATCH")
print("=" * 80)
for title, snippet in test_cases:
    l_pred, l_conf, _ = predict_track_bayesian(agent, title, snippet)
    o_pred, o_conf = predict_omlx(title, snippet)
    match = "MATCH" if l_pred == o_pred else "DIFF"
    print(f"{title[:28]:<32} | {l_pred} ({l_conf*100:4.1f}%) | {o_pred} ({o_conf*100:4.1f}%) | {match}")
