from configs import TrainConfig
from torch import nn, optim

TrainCFG = TrainConfig()

class TagTransformerTrainer:
    def __init__(self, model, vocab, device=TrainCFG.device):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])
        self.optimizer = optim.AdamW(model.parameters(), lr=TrainCFG.learning_rate)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            inputs = batch['input_ids'].to(self.device)
            targets = batch['target_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)

            self.optimizer.zero_grad()

            logits, embeddings = self.model(inputs, mask)

            # embeddings 将会用到后续对比损失

            # 直接输出的交叉熵
            loss = self.criterion(
                logits.view(-1, len(self.vocab)),
                targets.view(-1)
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
    
