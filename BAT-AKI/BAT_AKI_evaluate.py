import os
import pickle
from configs.pretrain_config import config
from pipelines.resources import load_all_resources
from pipelines.semantic import build_semantic_resources
from pipelines.load_pretrained import load_pretrained_mlm
from pipelines.finetune import evaluate_aki, evaluate_death, evaluate_rcvrvrt

def main():
    train_df, val_df, test_df, token2id, ontology_map_table, prompt_df = load_all_resources(config)

    token2ontoid, token2mrontoid, semantic_matrix = build_semantic_resources(
        ontology_map_table, prompt_df, token2id, token2id["[PAD]"]
    )

    model = load_pretrained_mlm(
        checkpoint_path="./outputs/best_mlm.pt",
        token2id=token2id,
        semantic_matrix=semantic_matrix,
        config=config,
        device=config["device"],
    )

    results = evaluate_aki(
        pretrained_model=model,
        token2id=token2id,
        token2ontoid=token2ontoid,
        token2mrontoid=token2mrontoid,
        device=config["device"],
        splits_dir="./splits",
        output_dir="./eval_outputs",
        config=config,
    )

    death_results = evaluate_death(
        use_module_embedding=config["use_module_embedding"],
        use_ontology=config["use_ontology"],
        pretrained_model=model,
        token2id=token2id,
        token2ontoid=token2ontoid,
        token2mrontoid=token2mrontoid,
        device=config["device"],
        sample_units=[...],
        config=config,
        model_name="Model_Proposed",
        splits_dir="./splits",
        output_prefix="./eval_outputs",
        label_source_df=None,
        label_merge_keys=('PATID', 'ENCOUNTERID'),
        label_source_col='death90',
    )

    rcvrvrt_results_ervrt = evaluate_rcvrvrt(
        use_module_embedding=config["use_module_embedding"],
        use_ontology=config["use_ontology"],
        pretrained_model=model,
        token2id=token2id,
        token2ontoid=token2ontoid,
        token2mrontoid=token2mrontoid,
        device=config["device"],
        sample_units=[...],
        config=config,
        model_name="Model_Proposed",
        splits_dir="./splits",
        output_prefix="./eval_outputs",
        label_source_df=None,
        label_merge_keys=('PATID', 'ENCOUNTERID'),
        label_source_col='AKI_ERVRT',
    )

    rcvrvrt_results_rcv = evaluate_rcvrvrt(
        use_module_embedding=config["use_module_embedding"],
        use_ontology=config["use_ontology"],
        pretrained_model=model,
        token2id=token2id,
        token2ontoid=token2ontoid,
        token2mrontoid=token2mrontoid,
        device=config["device"],
        sample_units=[...],
        config=config,
        model_name="Model_Proposed",
        splits_dir="./splits",
        output_prefix="./eval_outputs",
        label_source_df=None,
        label_merge_keys=('PATID', 'ENCOUNTERID'),
        label_source_col='AKI_RCV',
    )

    results_dir = "./main_results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "evaluate_aki_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    with open(os.path.join(results_dir, "evaluate_death_results.pkl"), "wb") as f:
        pickle.dump(death_results, f)
    
    with open(os.path.join(results_dir, "evaluate_rcvrvrt_ervrt_results.pkl"), "wb") as f:
        pickle.dump(rcvrvrt_results_ervrt, f)
    
    with open(os.path.join(results_dir, "evaluate_rcvrvrt_rcv_results.pkl"), "wb") as f:
        pickle.dump(rcvrvrt_results_rcv, f)
    
    print(f"All results saved to {results_dir}")

if __name__ == "__main__":
    main()
