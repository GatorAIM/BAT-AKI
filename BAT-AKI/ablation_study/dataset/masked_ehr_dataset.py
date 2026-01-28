import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
MASK_TOKEN = '[MASK]'
mask_prob = 0.15

def mask_tokens(tokens, token2id):
    masked_tokens = []
    labels = []

    for tok in tokens:
        prob = random.random()
        if tok in [PAD_TOKEN, UNK_TOKEN] or tok.startswith('SEX') or tok.startswith('AGE') or tok.startswith('RACE'):
            masked_tokens.append(tok)
            labels.append('[PAD]')
        elif prob < mask_prob:
            masked_tokens.append(MASK_TOKEN)
            labels.append(tok)
        else:
            masked_tokens.append(tok)
            labels.append('[PAD]')
    return masked_tokens, labels


class MaskedEHRDataset(Dataset):
    def __init__(self, dataframe, token2id, max_len=500, use_cls=True):
        self.sequences = dataframe['TOKEN_SEQUENCE'].tolist()
        self.deltaT_seqs = dataframe['DELTA_T_SEQUENCE'].tolist()
        self.seg_seqs = dataframe['TYPE_SEQUENCE'].tolist()
        self.module_seqs = dataframe['MODULE_ID_SEQUENCE'].tolist()
        self.flags = dataframe['FLAG'].astype(float).tolist() 
        self.flags2 = dataframe['FLAG_CDX'].astype(float).tolist()
        self.token2id = token2id
        self.use_cls = use_cls
        self.max_len = max_len

        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx].split()
        deltaT_values = self.deltaT_seqs[idx] 
        segment_ids_all = self.seg_seqs[idx] 
        module_values = self.module_seqs[idx]
        
        demo_tokens = tokens[:3]         
        event_tokens = tokens[3:]        
        deltaT_demo = deltaT_values[:3]
        deltaT_events = deltaT_values[3:]
        segment_demo = segment_ids_all[:3]
        segment_events = segment_ids_all[3:]
        module_demo = module_values[:3]
        module_events = module_values[3:]
        
        reserved = 4 if self.use_cls else 3
        event_tokens = event_tokens[-(self.max_len - reserved):]
        deltaT_events = deltaT_events[-(self.max_len - reserved):]
        segment_events = segment_events[-(self.max_len - reserved):]
        module_events = module_events[-(self.max_len - reserved):]

        if self.use_cls:
            tokens = ['[CLS]'] + demo_tokens + event_tokens
            deltaT = [0] + deltaT_demo + deltaT_events
            segment_ids = [0] + segment_demo + segment_events
            module_ids = [0] + module_demo + module_events
        else:
            tokens = demo_tokens + event_tokens
            deltaT = deltaT_demo + deltaT_events
            segment_ids = segment_demo + segment_events
            module_ids = module_demo + module_events


        pad_len = self.max_len - len(tokens)
        tokens += [PAD_TOKEN] * pad_len
        deltaT += [0] * pad_len
        segment_ids += [0] * pad_len  
        module_ids += [0] * pad_len 
        
        input_ids = [self.token2id.get(tok, self.token2id['[UNK]']) for tok in tokens]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'delta_t': torch.tensor(deltaT, dtype=torch.float),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'module_ids': torch.tensor(module_ids, dtype=torch.long),
            'label': torch.tensor(self.flags[idx], dtype=torch.float),  
            'flag2': torch.tensor(self.flags2[idx], dtype=torch.float)
        }
    

def get_mask_candidates(input_ids, protected_ids, mask_prob):
    return {
        tid for tid in input_ids
        if tid not in protected_ids and random.random() < mask_prob
    }

def get_mask_candidates_with_aki(input_ids, protected_ids, mask_prob, aki_token_ids=None, aki_mask_prob=None):
    candidates = set()
    for tid in input_ids:
        if tid in protected_ids:
            continue
        if aki_token_ids and tid in aki_token_ids:
            prob = aki_mask_prob if aki_mask_prob is not None else mask_prob
        else:
            prob = mask_prob
        if random.random() < prob:
            candidates.add(tid)
    return candidates

def expand_ranges(candidates, range_groups):
    expanded = set(candidates)
    for tid in candidates:
        for group in range_groups:
            if tid in group:
                expanded.update(group)
    return expanded

def apply_masking(input_ids, mask_id, candidates, protected_ids, pad_id):
    masked = []
    labels = []
    for tid in input_ids:
        if tid in protected_ids:
            masked.append(tid)
            labels.append(pad_id)
        elif tid in candidates:
            masked.append(mask_id)
            labels.append(tid)
        else:
            masked.append(tid)
            labels.append(pad_id)
    return masked, labels


def load_range_id_groups(range_csv_path):
    df = pd.read_csv(range_csv_path)
    return [set(range(row["start_id"], row["end_id"] + 1)) for _, row in df.iterrows()]

def get_mask_candidates(input_ids, protected_ids, mask_prob):
    return {
        tid for tid in input_ids
        if tid not in protected_ids and random.random() < mask_prob
    }


def expand_with_ranges(candidates, range_id_groups):
    expanded = set(candidates)
    for tid in candidates:
        for group in range_id_groups:
            if tid in group:
                expanded.update(group)
    return expanded



def apply_masking(input_ids, mask_id, pad_id, candidates, protected_ids):
    masked = []
    labels = []
    for tid in input_ids:
        if tid in protected_ids:
            masked.append(tid)
            labels.append(pad_id)
        elif tid in candidates:
            masked.append(mask_id)
            labels.append(tid)
        else:
            masked.append(tid)
            labels.append(pad_id)
    return masked, labels


class MaskedInput(Dataset):
    def __init__(self, base_dataset, token2id, mask_prob=0.15, do_mask=True, protect_special=True, protect_demo=True, expand_ranges=False, range_csv_path=None, token2ontoid=None, token2mrontoid=None, aki_token_ids=None, aki_mask_prob=None):
        self.base_dataset = base_dataset
        self.token2id = token2id
        self.token2ontoid = token2ontoid or {}
        self.token2mrontoid = token2mrontoid or {}
        self.max_len = base_dataset[0]['input_ids'].shape[0]
        
        self.do_mask = do_mask        
        self.mask_prob = mask_prob        
        
        self.pad_id = token2id['[PAD]']
        self.unk_id = token2id['[UNK]']
        self.cls_id = token2id['[CLS]']
        self.mask_id = token2id['[MASK]']

        self.protect_special = protect_special
        self.protect_demo = protect_demo
        self.expand_ranges = expand_ranges

        self.demo_ids = set([
            tid for tok, tid in token2id.items()
            if tok.startswith('SEX') or tok.startswith('AGE') or tok.startswith('RACE')
        ])
        
        if range_csv_path:
            self.range_id_groups = load_range_id_groups(range_csv_path)
        else:
            self.range_id_groups = []
            
        self.aki_token_ids = set(aki_token_ids) if aki_token_ids else set()
        self.aki_mask_prob = aki_mask_prob  
        
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        input_ids_raw = item['input_ids'].tolist()
        delta_t = item['delta_t']
        segment_ids = item['segment_ids']
        module_ids = item['module_ids']
        flag = item['label']
        flag2 = item['flag2']
        if self.do_mask:
            protected_ids = {self.pad_id}  
            if self.protect_special:
                protected_ids |= {self.unk_id, self.cls_id}
            protected_ids |= self.demo_ids
            
            mask_candidates = get_mask_candidates_with_aki(
                input_ids_raw,
                protected_ids,
                self.mask_prob,
                aki_token_ids=self.aki_token_ids,
                aki_mask_prob=self.aki_mask_prob
            )

            if self.expand_ranges and self.range_id_groups:
                mask_candidates = expand_with_ranges(mask_candidates, self.range_id_groups)

                
            masked_ids, labels = apply_masking(
                input_ids=input_ids_raw,
                mask_id=self.mask_id,
                pad_id=self.pad_id,
                candidates=mask_candidates,
                protected_ids=protected_ids
            )
        
        else:
            masked_ids = input_ids_raw
            labels = [self.pad_id] * self.max_len

        attention_mask = [1 if token_id != self.pad_id else 0 for token_id in input_ids_raw]
        
        id2token = {v: k for k, v in self.token2id.items()}  #       
        
        ontology_ids = []
        mrontology_ids = []
        for tid in input_ids_raw:
            token = id2token.get(tid, None)
            assert token is not None, f"  token_id {tid}  token"
            onto_id = self.token2ontoid.get(token, 0) 
            mronto_id = self.token2mrontoid.get(token, 0) 
            ontology_ids.append(onto_id)
            mrontology_ids.append(mronto_id)
        
        return {
            'input_ids': torch.tensor(masked_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'position_ids': torch.arange(self.max_len),
            'delta_t': delta_t.clone().detach().float() if isinstance(delta_t, torch.Tensor) else torch.tensor(delta_t, dtype=torch.float),
            'segment_ids': segment_ids.clone().detach().long() if isinstance(segment_ids, torch.Tensor) else torch.tensor(segment_ids, dtype=torch.long),
            'module_ids': module_ids.clone().detach().long() if isinstance(module_ids, torch.Tensor) else torch.tensor(module_ids, dtype=torch.long),
            'ontology_ids': torch.tensor(ontology_ids, dtype=torch.long),  
            'mrontology_ids': torch.tensor(mrontology_ids, dtype=torch.long),
            'flags': flag if isinstance(flag, torch.Tensor) else torch.tensor(flag, dtype=torch.float),
            'flag2': flag2
        }