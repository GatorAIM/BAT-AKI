import argparse
import os

from function.transformer_func import (
    train_transformer_earlystop,
    filter_retain_ready_data_by_keys,
    flatten_retain_ready_data,
    load_pickle,
    load_retain_ready_data,
    load_sampled_keys_df,
    process_visit_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", default="./", help="")
    parser.add_argument("--retain_file", default="./", help="")
    parser.add_argument("--sampled_keys_file", default="./", help="")
    parser.add_argument("--label_col", default="FLAG", help="")
    parser.add_argument("--label_filter", action=".", help="")
    parser.add_argument("--parse_mode", default=".", choices=[".", "."], help="")
    parser.add_argument("--model_samples_path", default="./", help="")
    parser.add_argument("--checkpoint_dir", default="./", help="")
    parser.add_argument("--checkpoint_name_fmt", default="./", help="")
    parser.add_argument("--unit_list", default=".", help="")
    parser.add_argument("--seed_list", default=".", help="")
    parser.add_argument("--output_csv", default="./", help="")
    return parser.parse_args()


def main():
    args = parse_args()

    retain_ready_data = load_retain_ready_data(args.data_dir, args.retain_file)
    sampled_keys_df = load_sampled_keys_df(args.data_dir, args.sampled_keys_file)
    retain_ready_data = filter_retain_ready_data_by_keys(retain_ready_data, sampled_keys_df)

    retain_samples = flatten_retain_ready_data(
        retain_ready_data,
        sampled_keys_df,
        label_col=args.label_col,
        label_filter=args.label_filter,
    )

    model_samples = load_pickle(args.model_samples_path)

    unit_list = [int(x) for x in args.unit_list.split(",") if x.strip()]
    seed_list = [int(x) for x in args.seed_list.split(",") if x.strip()]

    df = train_transformer_earlystop(
        retain_samples=retain_samples,
        model_samples=model_samples,
        unit_list=unit_list,
        seeds=seed_list,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name_fmt=args.checkpoint_name_fmt,
    )
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
