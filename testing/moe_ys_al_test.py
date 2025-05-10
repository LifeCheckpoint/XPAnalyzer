import pickle
import torch
from config.configs import ExperimentConfig
from marking.mark_tag_mapping import TagMarker
from data.read_moe_ys import read_tags_from_moe_ys_al

cfg = ExperimentConfig()

def test_moe_ys_al_xpmark():
    mark_data = read_tags_from_moe_ys_al(moe_ys_al_state_path=cfg.dataset.moe_ys_al_state_path)

    full_embedding = torch.load(cfg.dataset.embedding_saving_path, map_location="cpu")
    with open(cfg.dataset.tags_embedding_distance_pickle_path, "rb") as f:
        distance = pickle.load(f)

    marker = TagMarker(distance, full_embedding, weights_path=cfg.dataset.tags_weights_dataset_path)

    marks = marker.evaluate(mark_data)

    # 排序并输出
    sorted_marks = sorted(marks.items(), key=lambda x: x[1], reverse=True)
    i = 0
    for tag, score in sorted_marks[:50]:
        i += 1
        print(f"{i}. Tag: {tag}, Score: {score:.4f}")