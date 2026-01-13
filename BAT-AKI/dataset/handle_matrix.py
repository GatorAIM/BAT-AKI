import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

def build_token2ontoid(ontology_map_table):
    df_valid = ontology_map_table.dropna(subset=['ontology_id'])

    df_valid['ontology_id'] = df_valid['ontology_id'].astype(int)

    token2ontoid = dict(zip(df_valid['medcode_origin'], df_valid['ontology_id']))
    return token2ontoid

def build_token2mrontoid(ontology_map_table):
    df_valid = ontology_map_table.dropna(subset=['minor_ontology_id']).copy()
    if len(df_valid) == 0:
        return {}  
    df_valid['minor_ontology_id'] = df_valid['minor_ontology_id'].astype(int)
    return dict(zip(df_valid['medcode_origin'], df_valid['minor_ontology_id']))


def build_semantic_table(prompt_df, token2id):
    prompt_df['code'] = prompt_df['code'].astype(str)

    code2embedding = {
        row['code']: np.array(row['embedding']) 
        for _, row in prompt_df.iterrows()
    }
    code_list = list(code2embedding.keys())

    rows = []
    for token, idx in token2id.items():
        matched_code = next((code for code in code_list if code in token), None)
        embedding = code2embedding.get(matched_code) if matched_code else None
        rows.append({
            'token': token,
            'id': idx,
            'matched_code': matched_code,
            'embedding': embedding if embedding is not None else np.nan
        })

    return pd.DataFrame(rows)


def build_semantic_matrix_from_df(df: pd.DataFrame, target_dim=128, do_project=True):
    df_sorted = df.sort_values(by='id').reset_index(drop=True)

    raw_vectors = []
    valid_mask = []
    original_dim = None

    for x in df_sorted['embedding']:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            vec = np.array(x)
            if original_dim is None:
                original_dim = len(vec)
            if len(vec) == original_dim:
                raw_vectors.append(vec)
                valid_mask.append(True)
            else:
                raw_vectors.append(np.zeros(original_dim))
                valid_mask.append(False)
        else:
            if original_dim is None:
                original_dim = 384  
            raw_vectors.append(np.zeros(original_dim))
            valid_mask.append(False)

    embedding_array = np.stack(raw_vectors)  # [vocab_size, original_dim]

    if do_project and embedding_array.shape[1] != target_dim:
        print(f"📐 PCA : {embedding_array.shape[1]} → {target_dim}")
        valid_array = embedding_array[valid_mask]
        pca = PCA(n_components=target_dim)
        reduced_valid_array = pca.fit_transform(valid_array)  # shape: [N_valid, target_dim]

        embedding_array_reduced = np.zeros((len(embedding_array), target_dim), dtype=np.float32)
        embedding_array_reduced[np.where(valid_mask)[0]] = reduced_valid_array
    else:
        embedding_array_reduced = embedding_array.astype(np.float32)

    return torch.tensor(embedding_array_reduced, dtype=torch.float32)


def process_semantic_matrix(semantic_matrix: torch.Tensor, padding_idx: int = 0):
    embedding_dim = semantic_matrix.shape[1]
    mask_nan_rows = torch.isnan(semantic_matrix).any(dim=1)  
    for idx in range(semantic_matrix.size(0)):
        if idx == padding_idx:
            semantic_matrix[idx] = 0.0  
        elif mask_nan_rows[idx]:
            semantic_matrix[idx] = torch.empty(embedding_dim).uniform_(-0.1, 0.1)  
    return semantic_matrix