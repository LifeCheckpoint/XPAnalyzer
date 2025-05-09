from config.configs import TestConfig
import torch
from model.tag_transformer import TagTransformer
from utils.tag_transformer_predictor import TagPredictor
from utils.tag_transformer_occlusion import TagTransformerOcclusionAnalyzer

testCFG = TestConfig()

def test_tag_transformer_mask_predict():
    # tag MASK 预测测试

    model_datas = torch.load(testCFG.model_tag_transformer_path)
    vocab = model_datas['vocab']
    model = TagTransformer(len(vocab))
    model.load_state_dict(model_datas['model_state_dict'])
    model.eval()

    predictor = TagPredictor(model, vocab)

    # 定义用于测试可视化的输入标签
    test_input_tags = [
        ["傲娇", "金瞳", "冷娇", "银发", "猫娘", "萝莉", "裸足", "[MASK]"],
        ["银发", "萝莉", "光环", "步枪", "[MASK]"]
    ]

    # 获取预测结果（结构为List[List[List[Dict]]]）
    predict_results = predictor.predict(test_input_tags, top_k=testCFG.top_k)
    
    # 遍历每个测试样本
    for sample_idx, (input_tags, preds_per_seq) in enumerate(zip(test_input_tags, predict_results)):
        print(f"\nSample {sample_idx + 1}:")
        print(f"Input: {input_tags}")
        
        # 跳过[CLS]位置（preds_per_seq[0]），从第一个实际标签开始
        for tag, pos_preds in zip(input_tags, preds_per_seq[1:]):
            # 提取top_k的token列表
            top_tokens = [pred['token'] for pred in pos_preds[:testCFG.top_k]]
            print(f"({tag}): {top_tokens}")


def test_tag_transformer_occlusion():
    # 遮挡分析测试

    model_datas = torch.load(testCFG.model_tag_transformer_path)
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