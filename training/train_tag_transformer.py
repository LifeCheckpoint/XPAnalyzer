from config.configs import ExperimentConfig
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
from data.tag_prepocessing import load_character_tags
from data.tag_dataset import TagDataset, collate_fn
from model.tag_transformer import TagTransformer
from training.tag_transformer_trainer import TagTransformerTrainer
from training.tag_transformer_eval import TagTransformerEvaluator
from utils.tag_transformer_predictor import TagPredictor

cfg = ExperimentConfig()

def train_tag_transformer():
    # 读取数据 [character_name: {tag1, tag2, ...}]
    raw_data = load_character_tags(cfg.dataset.characters_tags_dataset_path)

    # 构建特殊词与词汇表
    vocab = defaultdict(lambda: len(vocab))
    special_tokens = ['[PAD]', '[CLS]', '[MASK]', '[UNK]', '[SEP]']
    for token in special_tokens:
        vocab[token]

    all_tags = [tags for tags in raw_data.values()]

    for tags_list in all_tags:
        for tag in tags_list:
            vocab[tag]

    print(f"Vocabulary size: {len(vocab)}")

    # 生成训练和验证数据
    train_data, val_data = train_test_split(all_tags, test_size=cfg.train.eval_size, random_state=cfg.seed)

    # 创建数据加载器
    train_dataset = TagDataset(train_data, vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, vocab['[PAD]'])
    )

    val_dataset = TagDataset(val_data, vocab)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluate.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab['[PAD]'])
    )

    # 初始化
    model = TagTransformer(len(vocab))
    trainer = TagTransformerTrainer(model, vocab)
    evaluator = TagTransformerEvaluator(model, vocab)
    predictor = TagPredictor(model, vocab)

    # 定义用于测试可视化的输入标签
    test_input_tags = [
        ["傲娇", "金瞳", "冷娇", "银发", "猫娘", "萝莉", "裸足", "[MASK]"],
        ["银发", "萝莉", "光环", "步枪", "[MASK]"]
    ]

    # 训练循环
    for epoch in range(cfg.train.epochs):
        # 训练阶段
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {loss:.4f}")

        # 验证阶段
        evaluate_data = evaluator.evaluate(val_loader)
        val_loss = evaluate_data["loss"]
        top1_acc = evaluate_data["top1_acc"]
        top5_acc = evaluate_data["top5_acc"]
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Top 1 Acc: {top1_acc:.4f}, Top 5 Acc: {top5_acc:.4f}")

        # 测试预测可视化
        print(f"\n--- Epoch {epoch+1} Specific Tag Prediction Test ---")
        
        # 获取预测结果（结构为List[List[List[Dict]]]）
        predict_results = predictor.predict(test_input_tags, top_k=cfg.evaluate.top_k)
        
        # 遍历每个测试样本
        for sample_idx, (input_tags, preds_per_seq) in enumerate(zip(test_input_tags, predict_results)):
            print(f"\nSample {sample_idx + 1}:")
            print(f"Input: {input_tags}")
            
            # 跳过[CLS]位置（preds_per_seq[0]），从第一个实际标签开始
            for tag, pos_preds in zip(input_tags, preds_per_seq[1:]):
                # 提取top_k的token列表
                top_tokens = [pred['token'] for pred in pos_preds[:cfg.evaluate.top_k]]
                print(f"({tag}): {top_tokens}")
        
        print("-" * 30)

        if (epoch + 1) % cfg.train.save_interval == 0:
            # 保存模型
            path = Path(cfg.train.save_dir) / f"{cfg.model.model_name}_epoch_{epoch+1}.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': dict(vocab)
            }, str(path))

            print(f"Model saved to {path}")
