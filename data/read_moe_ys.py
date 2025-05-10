from config.configs import ExperimentConfig
from typing import Dict, List, Union
from data.tag_prepocessing import load_character_tags
import json

cfg = ExperimentConfig()

def read_tags_from_moe_ys_al(moe_ys_al_state_path: str = cfg.dataset.moe_ys_al_state_path) -> Dict[str, Union[List[str], float]]:
    """
    从 Moe YS AL 状态文件读取角色和分数
    """
    with open(moe_ys_al_state_path, "r", encoding="utf-8") as f:
        data_raw = json.load(f)
    data = json.loads(data_raw["ratingHistory"])

    character_tags = load_character_tags(cfg.dataset.characters_tags_dataset_path)

    # data 格式：[{"name":..., "score":...}]
    # 将 name 映射到 tags

    result = []
    for item in data:
        name = item["name"]
        score = item["score"]

        if score == None:
            continue

        tags = character_tags.get(name, {})
        if not tags:
            continue

        result.append({
            "tags": list(tags),
            "score": score
        })

    # 转换分数为非零 Z-score
    scores = [item["score"] for item in result]
    mean_score = sum(scores) / len(scores)
    std_score = (sum((x - mean_score) ** 2 for x in scores) / len(scores)) ** 0.5
    for item in result:
        item["score"] = (item["score"] - mean_score) / std_score
    print(result[0:5])

    return result