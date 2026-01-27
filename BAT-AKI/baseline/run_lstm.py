import argparse
import os

from function.lstm_func import (
    build_tokenizer_from_id2token,
    filter_retain_ready_data_by_keys,
    flatten_retain_ready_data,
    load_retain_ready_data,
    load_sampled_keys_df,
    load_token_dicts,
    process_visit_ids,
    train_lstm_earlystop,
)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", default="./", help="")
    parser.add_argument("--retain_file", default="./", help="")
    parser.add_argument("--sampled_keys_file", default="./", help="")
    parser.add_argument("--label_col", default="FLAG", help="")
    parser.add_argument("--use_tokenizer", action="store_true", help="")
    parser.add_argument("--token_dict_dir", default="./", help="")
    parser.add_argument("--split_load_path", default="./", help="")
    parser.add_argument("--split_seed", type=int, default=1, help="")
    parser.add_argument("--unit_list", default=".", help="")
    parser.add_argument("--reruns", type=int, default=., help="")
    parser.add_argument("--base_seed", type=int, default=., help="")
    parser.add_argument("--batch_size", type=int, default=., help="")
    parser.add_argument("--embedding_dim", type=int, default=., help="")
    parser.add_argument("--hidden_dim", type=int, default=., help="")
    parser.add_argument("--num_layers", type=int, default=., help="")
    parser.add_argument("--bidirectional", action="store_true", help="")
    parser.add_argument("--dropout", type=float, default=., help="")
    parser.add_argument("--max_epochs", type=int, default=., help="")
    parser.add_argument("--patience", type=int, default=., help="")
    parser.add_argument("--lr", type=float, default=., help="")
    parser.add_argument("--save_dir", default="./", help="")
    parser.add_argument("--ckpt_name_fmt", default="lstm_best_seed{seed}.pt", help="")
    parser.add_argument("--output_csv", default="", help="")
    return parser.parse_args()


def main():
    args = parse_args()

    retain_ready_data = load_retain_ready_data(args.data_dir, args.retain_file)
    sampled_keys_df = load_sampled_keys_df(args.data_dir, args.sampled_keys_file)
    retain_ready_data = filter_retain_ready_data_by_keys(retain_ready_data, sampled_keys_df)

    tokenizer = None
    if args.use_tokenizer:
        _, id2token = load_token_dicts(args.token_dict_dir)
        tokenizer = build_tokenizer_from_id2token(id2token)

    retain_samples = flatten_retain_ready_data(
        retain_ready_data,
        sampled_keys_df,
        label_col=args.label_col,
        tokenizer=tokenizer,
    )

    df_rs = process_visit_ids(retain_samples)
    unit_list = [int(x) for x in args.unit_list.split(",") if x.strip()]

    df = train_lstm_earlystop(
        retain_samples=retain_samples,
        df_rs=df_rs,
        unit_list=unit_list,
        split_load_path=args.split_load_path,
        split_seed=args.split_seed,
        base_seed=args.base_seed,
        reruns=args.reruns,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        save_dir=args.save_dir,
        ckpt_name_fmt=args.ckpt_name_fmt,
    )

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
