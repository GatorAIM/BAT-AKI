import torch
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    def __init__(self, dataframe, token2id, max_len=500):
        self.input_data = [self.encode(seq, token2id, max_len) for seq in dataframe['TOKEN_SEQUENCE']]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.input_data[idx]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
        }

    def encode(self, tokens, token2id, max_len):
        PAD_TOKEN = '[PAD]'
        UNK_TOKEN = '[UNK]'
        CLS_TOKEN = '[CLS]'
        tokens = tokens.split()
        demo_tokens = tokens[:3]
        event_tokens = tokens[3:]
        event_tokens = event_tokens[-(max_len - 3):]
        tokens = demo_tokens + event_tokens
        tokens = [PAD_TOKEN] * (max_len - len(tokens)) + tokens

        input_ids = [token2id.get(tok, token2id[UNK_TOKEN]) for tok in tokens]
        attention_mask = [1 if tok != PAD_TOKEN else 0 for tok in tokens]
        return input_ids, attention_mask

def encode_sequence(row, token2id, max_len=500):
    tokens = row['TOKEN_SEQUENCE'].split()
    delta_t = row['DELTA_T_SEQUENCE']
    segment_ids = row['TYPE_SEQUENCE']

    demo_tokens = tokens[:3]
    event_tokens = tokens[3:]
    delta_demo = delta_t[:3]
    delta_event = delta_t[3:]
    seg_demo = segment_ids[:3]
    seg_event = segment_ids[3:]

    event_tokens = event_tokens[-(max_len - 4):]
    delta_event = delta_event[-(max_len - 4):]
    seg_event = seg_event[-(max_len - 4):]
    tokens = ['[CLS]'] + demo_tokens + event_tokens
    delta_full = [0] + delta_demo + delta_event
    seg_full = [0] + seg_demo + seg_event
    pad_len = max_len - len(tokens)
    tokens += ['[PAD]'] * pad_len
    delta_full += [0] * pad_len
    seg_full += [0] * pad_len

    input_ids = [token2id.get(tok, token2id['[UNK]']) for tok in tokens]
    attention_mask = [1 if tok != '[PAD]' else 0 for tok in tokens]
    return input_ids, attention_mask, delta_full, seg_full