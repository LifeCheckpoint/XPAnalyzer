from config.configs import ModelConfig
import torch.nn as nn

modelCFG = ModelConfig()

class TagTransformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=modelCFG.embed_dim,
            num_layers=modelCFG.num_layers,
            num_heads=modelCFG.num_heads,
            dim_feedforward=modelCFG.dim_feedforward
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
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, vocab_size)
        )


    def forward(self, x, attention_mask):
        x = self.embedding(x)
        src_key_padding_mask = (attention_mask == 0)
        
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_layer(encoded)
        return logits, encoded  # 直接返回整个编码器的输出作为嵌入
