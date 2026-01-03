from dataset.handle_matrix import (
    build_token2ontoid,
    build_token2mrontoid,
    build_semantic_table,
    build_semantic_matrix_from_df,
    process_semantic_matrix,
)

def build_semantic_resources(ontology_map_table, prompt_df, token2id, pad_id):
    token2ontoid = build_token2ontoid(ontology_map_table)
    token2mrontoid = build_token2mrontoid(ontology_map_table)

    semantic_table = build_semantic_table(prompt_df, token2id)
    semantic_matrix = build_semantic_matrix_from_df(semantic_table, target_dim=128)
    semantic_matrix = process_semantic_matrix(semantic_matrix, padding_idx=pad_id)

    return token2ontoid, token2mrontoid, semantic_matrix
