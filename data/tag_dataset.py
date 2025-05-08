from configs import TrainConfig
import torch
from torch.utils.data import Dataset
import numpy as np

trainCFG = TrainConfig()

class TagDataset(Dataset):
    def __init__(self, data, vocab, mask_prob=trainCFG.mask_prob):
        self.vocab = vocab
        self.unk_token_id = vocab['[UNK]']
        self.mask_token_id = vocab['[MASK]']
        self.cls_token_id = vocab['[CLS]']
        self.mask_prob = mask_prob
        
        # 生成序列
        self.samples = [
            [self.cls_token_id] + [vocab.get(tag, self.unk_token_id) for tag in tags]
            for tags in data
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        mask_positions = np.where(np.random.rand(len(tokens)) < self.mask_prob)[0]
        
        # 至少 mask 一个非 [CLS] 位置
        if len(mask_positions) == 0:
            mask_positions = np.array([np.random.randint(1, len(tokens))])
            
        masked_tokens = np.array(tokens)
        masked_tokens[mask_positions] = self.mask_token_id
        
        return {
            'input_ids': torch.LongTensor(masked_tokens),
            'target_ids': torch.LongTensor(tokens),
            'attention_mask': torch.ones(len(tokens))
        }

def collate_fn(batch, pad_token_id):
    """
    将不同长度的序列填充到相同长度并补充对应 attention mask
    """

    max_len = max(len(item['input_ids']) for item in batch)
    
    # 使用 value 填充序列到 max_len
    def pad_sequence(sequence, value, max_length):
        padding = torch.full((max_length - len(sequence),), value, dtype=torch.long)
        return torch.cat([sequence, padding])
    
    batch_dict = {
        'input_ids': [],
        'target_ids': [], 
        'attention_mask': []
    }
    
    # 输入
    batch_dict['input_ids'] = torch.stack([
        pad_sequence(item['input_ids'], pad_token_id, max_len) 
        for item in batch
    ])
    
    # 输出
    batch_dict['target_ids'] = torch.stack([
        pad_sequence(item['target_ids'], pad_token_id, max_len)
        for item in batch
    ])
    
    # 注意力掩码
    batch_dict['attention_mask'] = torch.stack([
        pad_sequence(item['attention_mask'], 0, max_len)
        for item in batch
    ])
    
    return batch_dict