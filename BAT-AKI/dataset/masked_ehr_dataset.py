import torch
from torch.utils.data import Dataset
import random
import pandas as pd

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
MASK_TOKEN = '[MASK]'

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
        deltaT_values = self.deltaT_seqs[idx]  # List of floats
        segment_ids_all = self.seg_seqs[idx] 
        module_values = self.module_seqs[idx]
        
        demo_tokens = tokens[:3]         # DEMO
        event_tokens = tokens[3:]        # EVENTS
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
    def __init__(
        self,
        base_dataset,
        token2id,
        mask_prob=0.15,
        do_mask=True,
        protect_special=True,
        protect_demo=True,
        expand_ranges=False,
        range_csv_path=None,
        token2ontoid=None,
        token2mrontoid=None,
        selected_tokens_df_map=None,
        inject_prob=0.0,
    ):
        self.base_dataset = base_dataset
        self.token2id = token2id
        self.token2ontoid = token2ontoid or {}
        self.token2mrontoid = token2mrontoid or {}
        self.max_len = base_dataset[0]["input_ids"].shape[0]

        self.do_mask = do_mask
        self.mask_prob = mask_prob

        self.pad_id = token2id["[PAD]"]
        self.unk_id = token2id["[UNK]"]
        self.cls_id = token2id["[CLS]"]
        self.mask_id = token2id["[MASK]"]

        self.protect_special = protect_special
        self.protect_demo = protect_demo
        self.expand_ranges = expand_ranges

        self.demo_ids = {
            tid
            for tok, tid in token2id.items()
            if tok.startswith("SEX") or tok.startswith("AGE") or tok.startswith("RACE")
        }

        if range_csv_path:
            self.range_id_groups = load_range_id_groups(range_csv_path)
        else:
            self.range_id_groups = []

        self.selected_map = selected_tokens_df_map or {}
        self.inject_prob = inject_prob

    def __len__(self):
        return len(self.base_dataset)

    def _random_fallback(self, idx):
        if 1300 <= idx <= 1586:
            return random.randint(1300, 1586)
        if 1587 <= idx <= 1996:
            return random.randint(1587, 1996)
        if 553 <= idx <= 1299 or 2021 <= idx <= 2403:
            choices = list(range(553, 1300)) + list(range(2021, 2404))
            return random.choice(choices)
        if 30 <= idx <= 546:
            return random.randint(30, 546)
        if 20 <= idx <= 29 or 2000 <= idx <= 2020:
            choices = list(range(20, 30)) + list(range(2000, 2021))
            return random.choice(choices)
        raise ValueError(f"token id {idx}")

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        input_ids_raw = item["input_ids"].tolist()
        delta_t = item["delta_t"]
        segment_ids = item["segment_ids"]
        module_ids = item["module_ids"]
        flag = item["label"]
        flag2 = item["flag2"]

        abnormal_flags = [0] * self.max_len

        if self.do_mask:
            protected_ids = {self.pad_id}
            if self.protect_special:
                protected_ids |= {self.unk_id, self.cls_id}
            protected_ids |= self.demo_ids

            mask_candidates = get_mask_candidates(
                input_ids_raw, protected_ids, self.mask_prob
            )

            if self.expand_ranges and self.range_id_groups:
                mask_candidates = expand_with_ranges(
                    mask_candidates, self.range_id_groups
                )

            masked_ids = []
            labels = []
            for tid in input_ids_raw:
                if tid in protected_ids:
                    masked_ids.append(tid)
                    labels.append(self.pad_id)
                elif tid in mask_candidates:
                    masked_ids.append(self.mask_id)
                    labels.append(tid)
                else:
                    masked_ids.append(tid)
                    labels.append(self.pad_id)

            injectable_positions = [
                i
                for i, tid in enumerate(input_ids_raw)
                if tid not in protected_ids
                and tid not in mask_candidates
                and tid in self.selected_map
            ]
            n_inject = int(len(injectable_positions) * self.inject_prob)
            inject_targets = (
                set(random.sample(injectable_positions, n_inject))
                if n_inject > 0
                else set()
            )

            for pos in inject_targets:
                tid = input_ids_raw[pos]
                opp_id = self.selected_map.get(tid)
                if opp_id is not None:
                    masked_ids[pos] = opp_id
                else:
                    masked_ids[pos] = self._random_fallback(tid)
                abnormal_flags[pos] = 1
        else:
            masked_ids = input_ids_raw
            labels = [self.pad_id] * self.max_len

        attention_mask = [
            1 if token_id != self.pad_id else 0 for token_id in input_ids_raw
        ]

        id2token = {v: k for k, v in self.token2id.items()}

        ontology_ids = []
        mrontology_ids = []
        for tid in input_ids_raw:
            token = id2token.get(tid)
            onto_id = self.token2ontoid.get(token, 0)
            mronto_id = self.token2mrontoid.get(token, 0)
            ontology_ids.append(onto_id)
            mrontology_ids.append(mronto_id)

        return {
            "input_ids": torch.tensor(masked_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "position_ids": torch.arange(self.max_len),
            "delta_t": delta_t.clone().detach().float()
            if isinstance(delta_t, torch.Tensor)
            else torch.tensor(delta_t, dtype=torch.float),
            "segment_ids": segment_ids.clone().detach().long()
            if isinstance(segment_ids, torch.Tensor)
            else torch.tensor(segment_ids, dtype=torch.long),
            "module_ids": module_ids.clone().detach().long()
            if isinstance(module_ids, torch.Tensor)
            else torch.tensor(module_ids, dtype=torch.long),
            "ontology_ids": torch.tensor(ontology_ids, dtype=torch.long),
            "mrontology_ids": torch.tensor(mrontology_ids, dtype=torch.long),
            "flags": flag
            if isinstance(flag, torch.Tensor)
            else torch.tensor(flag, dtype=torch.float),
            "flag2": flag2,
            "abnormal_flags": torch.tensor(abnormal_flags, dtype=torch.float),
        }
    
class MaskedInputabj(Dataset):
    def __init__(
        self,
        base_dataset,
        token2id,
        mask_prob=0.15,
        do_mask=True,
        protect_special=True,
        protect_demo=True,
        expand_ranges=False,
        range_csv_path=None,
        token2ontoid=None,
        token2mrontoid=None,
        selected_tokens_df_map=None,
        inject_prob=0.0,
    ):
        self.base_dataset = base_dataset
        self.token2id = token2id
        self.token2ontoid = token2ontoid or {}
        self.token2mrontoid = token2mrontoid or {}
        self.max_len = base_dataset[0]["input_ids"].shape[0]

        self.do_mask = do_mask
        self.mask_prob = mask_prob

        self.pad_id = token2id["[PAD]"]
        self.unk_id = token2id["[UNK]"]
        self.cls_id = token2id["[CLS]"]
        self.mask_id = token2id["[MASK]"]

        self.protect_special = protect_special
        self.protect_demo = protect_demo
        self.expand_ranges = expand_ranges

        self.demo_ids = {
            tid
            for tok, tid in token2id.items()
            if tok.startswith("SEX") or tok.startswith("AGE") or tok.startswith("RACE")
        }

        if range_csv_path:
            self.range_id_groups = load_range_id_groups(range_csv_path)
        else:
            self.range_id_groups = []

        self.selected_map = selected_tokens_df_map or {}
        self.inject_prob = inject_prob

    def __len__(self):
        return len(self.base_dataset)

    def _random_same_prefix(self, tid, id2token, token2id):
        ignore_prefixes = {
            "LAB::48643-1(mL/min)_Q",
            "LAB::48643-1(NI)_Q",
            "LAB::LG13584-4(OT)_Q",
            "LAB::LG37574-7(OT)_Q",
            "LAB::LG42103-8(OT)_Q",
            "LAB::LG4652-6(OT)_Q",
            "LAB::LG16225-1(mg/dL)_Q",
            "LAB::LG46646-2(mg/dL)_Q",
        }

        tok = id2token.get(tid)
        if tok is None:
            return None

        if tok.startswith("SYSTOLIC_") or tok.startswith("DIASTOLIC_"):
            prefix = tok.split("_")[0] + "_"
            candidates = [
                tid2 for tok2, tid2 in token2id.items() if tok2.startswith(prefix)
            ]
            other_choices = [c for c in candidates if c != tid]
            if not other_choices:
                return None
            return random.choice(other_choices)

        if "_Q" not in tok:
            return None

        prefix = tok.split("_Q")[0] + "_Q"
        candidates = [
            tid2 for tok2, tid2 in token2id.items() if tok2.startswith(prefix)
        ]

        if len(candidates) <= 1:
            return None

        other_choices = [c for c in candidates if c != tid]
        if not other_choices:
            return None

        return random.choice(other_choices)

    def _random_fallback(self, idx):
        if 1300 <= idx <= 1586:
            return random.randint(1300, 1586)
        if 1587 <= idx <= 1996:
            return random.randint(1587, 1996)
        if 553 <= idx <= 1299 or 2021 <= idx <= 2403:
            choices = list(range(553, 1300)) + list(range(2021, 2404))
            return random.choice(choices)
        if 30 <= idx <= 546:
            return random.randint(30, 546)
        if 20 <= idx <= 29 or 2000 <= idx <= 2020:
            choices = list(range(20, 30)) + list(range(2000, 2021))
            return random.choice(choices)
        raise ValueError(f"token id {idx}")

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        input_ids_raw = item["input_ids"].tolist()
        delta_t = item["delta_t"]
        segment_ids = item["segment_ids"]
        module_ids = item["module_ids"]
        flag = item["label"]
        flag2 = item["flag2"]

        abnormal_flags = [0] * self.max_len

        if self.do_mask:
            protected_ids = {self.pad_id}
            if self.protect_special:
                protected_ids |= {self.unk_id, self.cls_id}
            protected_ids |= self.demo_ids

            mask_candidates = get_mask_candidates(
                input_ids_raw, protected_ids, self.mask_prob
            )

            if self.expand_ranges and self.range_id_groups:
                mask_candidates = expand_with_ranges(
                    mask_candidates, self.range_id_groups
                )

            masked_ids = []
            labels = []
            for tid in input_ids_raw:
                if tid in protected_ids:
                    masked_ids.append(tid)
                    labels.append(self.pad_id)
                elif tid in mask_candidates:
                    masked_ids.append(self.mask_id)
                    labels.append(tid)
                else:
                    masked_ids.append(tid)
                    labels.append(self.pad_id)

            injectable_positions = [
                i
                for i, tid in enumerate(input_ids_raw)
                if tid not in protected_ids
                and tid not in mask_candidates
                and tid in self.selected_map
            ]
            n_inject = int(len(injectable_positions) * self.inject_prob)
            inject_targets = (
                set(random.sample(injectable_positions, n_inject))
                if n_inject > 0
                else set()
            )

            id2token = {v: k for k, v in self.token2id.items()}

            for pos in inject_targets:
                tid = input_ids_raw[pos]
                opp_id = self._random_same_prefix(tid, id2token, self.token2id)
                if opp_id is not None:
                    masked_ids[pos] = opp_id
                else:
                    masked_ids[pos] = self._random_fallback(tid)
                abnormal_flags[pos] = 1
        else:
            masked_ids = input_ids_raw
            labels = [self.pad_id] * self.max_len

        attention_mask = [
            1 if token_id != self.pad_id else 0 for token_id in input_ids_raw
        ]

        id2token = {v: k for k, v in self.token2id.items()}

        ontology_ids = []
        mrontology_ids = []
        for tid in input_ids_raw:
            token = id2token.get(tid)
            onto_id = self.token2ontoid.get(token, 0)
            mronto_id = self.token2mrontoid.get(token, 0)
            ontology_ids.append(onto_id)
            mrontology_ids.append(mronto_id)

        return {
            "input_ids": torch.tensor(masked_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "position_ids": torch.arange(self.max_len),
            "delta_t": delta_t.clone().detach().float()
            if isinstance(delta_t, torch.Tensor)
            else torch.tensor(delta_t, dtype=torch.float),
            "segment_ids": segment_ids.clone().detach().long()
            if isinstance(segment_ids, torch.Tensor)
            else torch.tensor(segment_ids, dtype=torch.long),
            "module_ids": module_ids.clone().detach().long()
            if isinstance(module_ids, torch.Tensor)
            else torch.tensor(module_ids, dtype=torch.long),
            "ontology_ids": torch.tensor(ontology_ids, dtype=torch.long),
            "mrontology_ids": torch.tensor(mrontology_ids, dtype=torch.long),
            "flags": flag
            if isinstance(flag, torch.Tensor)
            else torch.tensor(flag, dtype=torch.float),
            "flag2": flag2,
            "abnormal_flags": torch.tensor(abnormal_flags, dtype=torch.float),
        }
    
    

class FluctuationInput(Dataset):
    def __init__(
        self,
        base_dataset,
        token2id,
        mask_prob=0.15,
        do_mask=True,
        protect_special=True,
        protect_demo=True,
        expand_ranges=False,
        range_csv_path=None,
        token2ontoid=None,
        token2mrontoid=None,
        inject_prob=0.3,
        fluctuating_prefixes=None,
    ):
        self.base_dataset = base_dataset
        self.token2id = token2id
        self.token2ontoid = token2ontoid or {}
        self.token2mrontoid = token2mrontoid or {}
        self.max_len = base_dataset[0]["input_ids"].shape[0]

        self.do_mask = do_mask
        self.mask_prob = mask_prob

        self.pad_id = token2id["[PAD]"]
        self.unk_id = token2id["[UNK]"]
        self.cls_id = token2id["[CLS]"]
        self.mask_id = token2id["[MASK]"]

        self.protect_special = protect_special
        self.protect_demo = protect_demo
        self.expand_ranges = expand_ranges
        self.inject_prob = inject_prob

        self.fluctuating_prefixes = (
            set(fluctuating_prefixes) if fluctuating_prefixes else set()
        )

        self.demo_ids = {
            tid
            for tok, tid in token2id.items()
            if tok.startswith("SEX") or tok.startswith("AGE") or tok.startswith("RACE")
        }

        if range_csv_path:
            self.range_id_groups = load_range_id_groups(range_csv_path)
        else:
            self.range_id_groups = []

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        input_ids_raw = item["input_ids"].tolist()
        delta_t = item["delta_t"]
        segment_ids = item["segment_ids"]
        module_ids = item["module_ids"]
        flag = item["label"]
        flag2 = item["flag2"]

        abnormal_flags = [0] * self.max_len

        if self.do_mask:
            protected_ids = {self.pad_id}
            if self.protect_special:
                protected_ids |= {self.unk_id, self.cls_id}
            protected_ids |= self.demo_ids

            mask_candidates = get_mask_candidates(
                input_ids_raw, protected_ids, self.mask_prob
            )
            if self.expand_ranges and self.range_id_groups:
                mask_candidates = expand_with_ranges(
                    mask_candidates, self.range_id_groups
                )

            masked_ids = []
            labels = []
            for tid in input_ids_raw:
                if tid in protected_ids:
                    masked_ids.append(tid)
                    labels.append(self.pad_id)
                elif tid in mask_candidates:
                    masked_ids.append(self.mask_id)
                    labels.append(tid)
                else:
                    masked_ids.append(tid)
                    labels.append(self.pad_id)
        else:
            masked_ids = input_ids_raw
            labels = [self.pad_id] * self.max_len

        if self.fluctuating_prefixes:
            id2token = {v: k for k, v in self.token2id.items()}
            for pos, tid in enumerate(input_ids_raw):
                tok = id2token.get(tid, "")
                for p in self.fluctuating_prefixes:
                    if tok.startswith(p):
                        if random.random() < self.inject_prob:
                            abnormal_flags[pos] = 1
                        break

        attention_mask = [
            1 if token_id != self.pad_id else 0 for token_id in input_ids_raw
        ]

        id2token = {v: k for k, v in self.token2id.items()}

        ontology_ids = []
        mrontology_ids = []
        for tid in input_ids_raw:
            token = id2token.get(tid)
            onto_id = self.token2ontoid.get(token, 0)
            mronto_id = self.token2mrontoid.get(token, 0)
            ontology_ids.append(onto_id)
            mrontology_ids.append(mronto_id)

        return {
            "input_ids": torch.tensor(masked_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "position_ids": torch.arange(self.max_len),
            "delta_t": delta_t.clone().detach().float()
            if isinstance(delta_t, torch.Tensor)
            else torch.tensor(delta_t, dtype=torch.float),
            "segment_ids": segment_ids.clone().detach().long()
            if isinstance(segment_ids, torch.Tensor)
            else torch.tensor(segment_ids, dtype=torch.long),
            "module_ids": module_ids.clone().detach().long()
            if isinstance(module_ids, torch.Tensor)
            else torch.tensor(module_ids, dtype=torch.long),
            "ontology_ids": torch.tensor(ontology_ids, dtype=torch.long),
            "mrontology_ids": torch.tensor(mrontology_ids, dtype=torch.long),
            "flags": flag
            if isinstance(flag, torch.Tensor)
            else torch.tensor(flag, dtype=torch.float),
            "flag2": flag2,
            "abnormal_flags": torch.tensor(abnormal_flags, dtype=torch.float),
        }


class PatientExtremeInput(Dataset):
    def __init__(
        self,
        base_dataset,
        token2id,
        record_ids,
        patient_map,
        mask_prob=0.15,
        do_mask=True,
        protect_special=True,
        protect_demo=True,
        expand_ranges=False,
        range_csv_path=None,
        token2ontoid=None,
        token2mrontoid=None,
        inject_prob=0.3,
    ):
        self.base_dataset = base_dataset
        self.token2id = token2id
        self.token2ontoid = token2ontoid or {}
        self.token2mrontoid = token2mrontoid or {}
        self.max_len = base_dataset[0]["input_ids"].shape[0]

        self.record_ids = list(record_ids)
        self.patient_map = patient_map

        self.do_mask = do_mask
        self.mask_prob = mask_prob

        self.pad_id = token2id["[PAD]"]
        self.unk_id = token2id["[UNK]"]
        self.cls_id = token2id["[CLS]"]
        self.mask_id = token2id["[MASK]"]

        self.protect_special = protect_special
        self.protect_demo = protect_demo
        self.expand_ranges = expand_ranges
        self.inject_prob = inject_prob

        self.demo_ids = {
            tid
            for tok, tid in token2id.items()
            if tok.startswith("SEX") or tok.startswith("AGE") or tok.startswith("RACE")
        }

        if range_csv_path:
            self.range_id_groups = load_range_id_groups(range_csv_path)
        else:
            self.range_id_groups = []

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        rid = self.record_ids[idx]

        input_ids_raw = item["input_ids"].tolist()
        delta_t = item["delta_t"]
        segment_ids = item["segment_ids"]
        module_ids = item["module_ids"]
        flag = item["label"]
        flag2 = item["flag2"]

        abnormal_flags = [0] * self.max_len

        if self.do_mask:
            protected_ids = {self.pad_id}
            if self.protect_special:
                protected_ids |= {self.unk_id, self.cls_id}
            protected_ids |= self.demo_ids

            mask_candidates = get_mask_candidates(
                input_ids_raw, protected_ids, self.mask_prob
            )
            if self.expand_ranges and self.range_id_groups:
                mask_candidates = expand_with_ranges(
                    mask_candidates, self.range_id_groups
                )

            masked_ids = []
            labels = []
            for tid in input_ids_raw:
                if tid in protected_ids:
                    masked_ids.append(tid)
                    labels.append(self.pad_id)
                elif tid in mask_candidates:
                    masked_ids.append(self.mask_id)
                    labels.append(tid)
                else:
                    masked_ids.append(tid)
                    labels.append(self.pad_id)
        else:
            masked_ids = input_ids_raw
            labels = [self.pad_id] * self.max_len

        id2token = {v: k for k, v in self.token2id.items()}
        patient_tokens = self.patient_map.get(rid, set())

        if patient_tokens:
            for pos, tid in enumerate(input_ids_raw):
                tok = id2token.get(tid, "")
                for p in patient_tokens:
                    if tok == p or tok.startswith(p):
                        if random.random() < self.inject_prob:
                            abnormal_flags[pos] = 1
                        break

        attention_mask = [
            1 if token_id != self.pad_id else 0 for token_id in input_ids_raw
        ]

        ontology_ids = []
        mrontology_ids = []
        for tid in input_ids_raw:
            token = id2token.get(tid)
            onto_id = self.token2ontoid.get(token, 0)
            mronto_id = self.token2mrontoid.get(token, 0)
            ontology_ids.append(onto_id)
            mrontology_ids.append(mronto_id)

        return {
            "input_ids": torch.tensor(masked_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "position_ids": torch.arange(self.max_len),
            "delta_t": delta_t.clone().detach().float()
            if isinstance(delta_t, torch.Tensor)
            else torch.tensor(delta_t, dtype=torch.float),
            "segment_ids": segment_ids.clone().detach().long()
            if isinstance(segment_ids, torch.Tensor)
            else torch.tensor(segment_ids, dtype=torch.long),
            "module_ids": module_ids.clone().detach().long()
            if isinstance(module_ids, torch.Tensor)
            else torch.tensor(module_ids, dtype=torch.long),
            "ontology_ids": torch.tensor(ontology_ids, dtype=torch.long),
            "mrontology_ids": torch.tensor(mrontology_ids, dtype=torch.long),
            "flags": flag
            if isinstance(flag, torch.Tensor)
            else torch.tensor(flag, dtype=torch.float),
            "flag2": flag2,
            "abnormal_flags": torch.tensor(abnormal_flags, dtype=torch.float),
            "record_id": self.record_ids[idx],
        }