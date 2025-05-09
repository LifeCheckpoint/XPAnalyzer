from collections import Counter
from config.configs import ExperimentConfig
from data.tag_prepocessing import load_character_tags
from model.tag_transformer import TagTransformer
from pathlib import Path
from sklearn.manifold import TSNE
from utils.tag_transformer_predictor import TagPredictor
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch

cfg = ExperimentConfig()

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def split_list(lst, n):
    """将列表分割成大小为 n 的子列表"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def test_tag_transformer_tsne():
    """词嵌入可视化"""

    # 加载原始标签映射表统计标签频率
    tags_dict = load_character_tags(cfg.dataset.characters_tags_dataset_path)
    tag_counts = Counter()
    for tags in tags_dict.values():
        tag_counts.update(tags)

    # 初始化模型等
    model_datas = torch.load(cfg.test.model_tag_transformer_path)
    vocab = model_datas["vocab"]
    model = TagTransformer(len(vocab))
    model.load_state_dict(model_datas["model_state_dict"])
    model.eval()

    predictor = TagPredictor(model, vocab)

    # 生成所有词汇 embedding
    print("Extracting word embeddings...")
    
    # 打包成 batches
    vocab_list = list(vocab.keys())
    batches = prepare_batch(predictor, [[tag] for tag in vocab_list])
    betches_gen = split_list(batches, cfg.test.batch_size)
    collated_batches = []
    for batch in betches_gen:
        collated_batches.append(predictor.collate_batch(batch))

    with torch.no_grad():
        embeddings = []
        for batch in collated_batches:
            embedding = predictor.model(batch['input_ids'], batch['attention_mask'])["encoded"]
            embeddings.append(embedding.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = torch.reshape(embeddings, (-1, cfg.model.embed_dim))

    print(f"Embeddings shape: {embeddings.shape}")

    # 选取频率最高的前 N 个词汇
    print(f"Selecting top {cfg.test.tsne_visual_num_words} most frequent words")
    sorted_tags_by_freq = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)

    # 选取前 N 个词汇及其频率
    top_tags_freq = sorted_tags_by_freq[:cfg.test.tsne_visual_num_words]
    top_tags = [tag for tag, freq in top_tags_freq]
    top_freqs = [freq for tag, freq in top_tags_freq]

    # 获取这些词汇对应的嵌入和索引
    top_tag_indices = [vocab[tag] for tag in top_tags if tag in vocab] # 确保词汇在构建的词汇表中
    top_tag_embeddings = embeddings[top_tag_indices]

    print(f"Selected {len(embeddings)} words for visualization")

    # t-SNE
    print("Performing t-SNE dimensionality reduction")
    tsne = TSNE(
        n_components=2,
        random_state=cfg.seed,
        perplexity=cfg.test.tsne_perplexity,
        n_iter=cfg.test.tsne_n_iter,
        init="random",
        learning_rate="auto"
    )
    all_embeddings_2d = tsne.fit_transform(embeddings.numpy())
    # 前 N 个词汇
    top_tag_embeddings_2d = all_embeddings_2d[top_tag_indices]

    # 可视化
    df = pd.DataFrame(top_tag_embeddings_2d, columns=["x", "y"])
    df["word"] = top_tags
    # 词汇到频率映射
    tag_to_freq = dict(top_tags_freq)
    df["frequency"] = df["word"].apply(lambda w: tag_to_freq.get(w, 0)) # 获取对应词汇的频率

    print("Generating visualization")
    plt.figure(figsize=(12, 12))

    # 使用 scatterplot 绘制点
    scatterplot = sns.scatterplot(
        x="x",
        y="y",
        hue="frequency", # 用频率作为颜色
        size="frequency", # 可选：用频率作为点的大小
        sizes=(50, 2000), # 点大小的范围
        palette="plasma", # 颜色方案
        data=df,
        alpha=0.7
    )

    # 添加词汇标签
    min_freq = min(top_freqs)
    max_freq = max(top_freqs)
    
    for i in range(len(df)):
        normalized_freq = (df.iloc[i]["frequency"] - min_freq) / (max_freq - min_freq)
        # 频率映射到 8-12 的字体大小
        fontsize = 8 + normalized_freq * 4
        
        plt.text(
            df.iloc[i]["x"],
            df.iloc[i]["y"],
            df.iloc[i]["word"],
            fontsize=fontsize,
            fontfamily="Microsoft YaHei",
            ha="center",
            va="bottom"
        )

    plt.title(f"t-SNE Visualization of Top {len(top_tags)} Word Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 1])

    output_filename = Path(cfg.test.tsne_image_output_dir) / f"word_embeddings_tsne_top{len(top_tags)}_freq.png"
    plt.savefig(str(output_filename), dpi=300)
    print(f"Visualization saved to {output_filename}")

def prepare_batch(predictor, input_tags_batch):
    """将原始输入转换为token IDs"""
    return [
        {
            "input_ids": torch.tensor([predictor.vocab.get(tag, predictor.vocab['[UNK]']) for tag in tags],
                dtype=torch.long
            ),
            "target_ids": torch.tensor([predictor.vocab.get(tag, predictor.vocab['[UNK]']) for tag in tags],
                dtype=torch.long
            ),
            "attention_mask": torch.ones(len(tags), dtype=torch.long)
        }
        for tags in input_tags_batch
    ]