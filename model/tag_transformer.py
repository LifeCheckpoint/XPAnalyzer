from config.configs import ExperimentConfig
import torch.nn.functional as F
import torch.nn as nn
import torch

cfg = ExperimentConfig()

class TagTransformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dim_feedforward=cfg.model.dim_feedforward
        ):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 主任务
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, vocab_size)
        )

        # 对比任务
        self.contrastive_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.pooler = AttentionPooling(embed_dim)

    def forward(self, x, attention_mask):
        x = self.embedding(x)
        src_key_padding_mask = (attention_mask == 0)
        
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        contrastive_output = self.contrastive_layer(encoded)
        sentence_emb = self.pooler(contrastive_output, attention_mask)
        logits = self.output_layer(encoded)
        
        return {
            "logits": logits,
            "contrastive": contrastive_output,
            "sentence_emb": sentence_emb,
            "encoded": encoded
        }


class AttentionPooling(nn.Module):
    """注意力池化"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, embeddings, mask):
        # embeddings: (batch_size, seq_len, embed_dim)
        # mask: (batch_size, seq_len)
        attn_weights = self.attention(embeddings).squeeze(-1)  # (batch_size, seq_len)
        attn_weights = attn_weights.masked_fill(~mask.bool(), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)
        return torch.sum(embeddings * attn_weights.unsqueeze(-1), dim=1)