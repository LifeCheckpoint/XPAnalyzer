from config.configs import ExperimentConfig
import torch
import pickle
import matplotlib.pyplot as plt
from model.tag_transformer import TagTransformer
from utils.tag_transformer_predictor import TagPredictor
from utils.tag_transformer_occlusion import TagTransformerOcclusionAnalyzer
from utils.tag_transformer_emb_dist import TagTransformerEmbeddingDistance

cfg = ExperimentConfig()

def test_tag_transformer_mask_predict():
    # tag MASK 预测测试

    model_datas = torch.load(cfg.test.model_tag_transformer_path)
    vocab = model_datas['vocab']
    model = TagTransformer(len(vocab))
    model.load_state_dict(model_datas['model_state_dict'])
    model.eval()

    predictor = TagPredictor(model, vocab)

    # 定义用于测试可视化的输入标签
    test_input_tags = [
        ["傲娇", "金瞳", "冷娇", "银发", "猫娘", "萝莉", "裸足", "[MASK]"],
        ["银发", "萝莉", "光环", "步枪", "[MASK]"],
        ["巨乳", "御姐", "分离袖子", "渐变色发", "过膝靴", "手套", "紫瞳", "金发", "[MASK]"],
        ["金发", "御姐", "红发", "腹黑", "颜艺", "褐色皮肤", "蓝瞳", "[MASK]"],
        ["扶她", "[MASK]"]
    ]

    # 获取预测结果（结构为List[List[List[Dict]]]）
    predict_results = predictor.predict(test_input_tags, top_k=cfg.test.top_k)
    
    # 遍历每个测试样本
    for sample_idx, (input_tags, preds_per_seq) in enumerate(zip(test_input_tags, predict_results)):
        print(f"\nSample {sample_idx + 1}:")
        print(f"Input: {input_tags}")
        
        # 跳过[CLS]位置（preds_per_seq[0]），从第一个实际标签开始
        for tag, pos_preds in zip(input_tags, preds_per_seq[1:]):
            # 提取top_k的token列表
            top_tokens = [pred['token'] for pred in pos_preds[:cfg.test.top_k]]
            print(f"({tag}): {top_tokens}")


def test_tag_transformer_occlusion():
    # 遮挡分析测试

    model_datas = torch.load(cfg.test.model_tag_transformer_path)
    vocab = model_datas['vocab']
    model = TagTransformer(len(vocab))
    model.load_state_dict(model_datas['model_state_dict'])
    model.eval()

    occlusion = TagTransformerOcclusionAnalyzer(model, vocab)

    # 遮挡分析测试
    input_for_occlusion = ["傲娇", "金瞳", "冷娇", "银发", "猫娘", "萝莉", "裸足", "[MASK]"]
    target_tag_for_analysis = "兽娘"

    occlusion_impact = occlusion.analyze_impact(
        input_for_occlusion,
        target_tag_for_analysis,
        max_combo_size=3
    )

    print(f"Analyzing impact on predicting '{target_tag_for_analysis}' probability for input: {input_for_occlusion}")

    # 找到要分析的位置，默认为对原序列 [MASK] 位置的预测
    mask_indices = [i for i, tag in enumerate(["[CLS]"] + input_for_occlusion) if tag == "[MASK]"]
    position_to_report = mask_indices[0] if mask_indices else None # 如果有 [MASK]，报告第一个 [MASK] 的影响

    if position_to_report is not None:
        print(f"\nImpact on probability at position {position_to_report} ('{(['[CLS]'] + input_for_occlusion)[position_to_report]}'):")
        # 按影响大小排序并打印结果 (只打印对 [MASK] 位置的影响)
        sorted_impact = sorted(
            [(combo, impact[position_to_report]) for combo, impact in occlusion_impact.items() if position_to_report in impact],
            key=lambda item: item[1], # 按影响排序
            reverse=True
        )

        for combo, impact_value in sorted_impact:
            print(f"  Combination {combo}: Impact = {impact_value:.6f}")
    else:
        print("\nImpact on probability for all analyzed positions:")
        # 如果没有 [MASK]，打印所有分析位置的影响 (可能比较多)
        for combo, impact_at_positions in occlusion_impact.items():
            print(f"  Combination {combo}:")
            for pos, impact_value in impact_at_positions.items():
                print(f"    Position {pos} ('{(['[CLS]'] + input_for_occlusion)[pos]}'): Impact = {impact_value:.4f}")


def test_tag_transformer_embedding_distance():
    """标签嵌入距离测试"""

    model_datas = torch.load(cfg.test.model_tag_transformer_path)
    vocab = model_datas['vocab']
    model = TagTransformer(len(vocab))
    model.load_state_dict(model_datas['model_state_dict'])
    model.eval()

    emb_disor = TagTransformerEmbeddingDistance(model=model, vocab=vocab)

    tags = list(vocab.keys())

    # 计算标签嵌入距离
    embedding_distance, embedding_dict = emb_disor.compute_embedding_distance(tags)
    torch.save(embedding_dict, f"{cfg.test.emb_dist_output_dir}/tag_embedding_dict.pt")

    # 保存距离信息
    with open(f"{cfg.test.emb_dist_output_dir}/tag_embedding_distance.pkl", "wb") as f:
        pickle.dump(embedding_distance, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 输出距离信息
    eps = 1e-1
    print("N:", len(embedding_distance))
    print("AVG:", sum(embedding_distance.values()) / len(embedding_distance))
    print("AVG (without zero):", sum([v for v in embedding_distance.values() if v > eps]) / len([v for v in embedding_distance.values() if v > eps]))
    print("MAX:", max(embedding_distance.values()))
    print("MIN (without zero):", min([v for v in embedding_distance.values() if v > eps]))
    print("Distance info saved to:", f"{cfg.test.emb_dist_output_dir}/tag_embedding_distance.pkl")

    # 直方图
    plt.figure(figsize=(10, 6))
    plt.hist(embedding_distance.values(), bins=20, alpha=0.7)
    plt.title('Tag Embedding Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(f"{cfg.test.emb_dist_output_dir}/tag_embedding_distance_distribution.png")
    plt.show()
