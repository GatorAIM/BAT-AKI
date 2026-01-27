import argparse
import os

from function.xgboost_func import (
    check_columns_exist,
    drop_sparse_feature_cols,
    filter_by_test_keys,
    load_and_merge_labels,
    load_cohort_csv,
    run_xgboost_bootstrap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cohort_csv", default="./", help="")
    parser.add_argument("--test_csv", default="", help="")
    parser.add_argument("--check_cols", default="", help="")
    parser.add_argument("--label_col", default="FLAG", help="")
    parser.add_argument("--label_datafolder", default="./", help="")
    parser.add_argument("--label_site", default="", help="")
    parser.add_argument("--drop_sparse", action="store_true", help="")
    parser.add_argument("--zero_ratio_threshold", type=float, default=., help="")
    parser.add_argument("--split_load_path", default="./", help="")
    parser.add_argument("--split_seed", type=int, default=., help="")
    parser.add_argument("--unit_list", default=".", help="")
    parser.add_argument("--n_runs", type=int, default=., help="")
    parser.add_argument("--save_dir", default="./", help="")
    parser.add_argument("--prefix", default=".", help="")
    parser.add_argument("--output_csv", default="", help="")
    return parser.parse_args()


def main():
    args = parse_args()

    cohort = load_cohort_csv(args.cohort_csv)

    if args.check_cols:
        cols = [c.strip() for c in args.check_cols.split(",") if c.strip()]
        check_columns_exist(cohort, cols)

    if args.test_csv:
        cohort = filter_by_test_keys(cohort, args.test_csv)

    if args.label_site:
        cohort = load_and_merge_labels(args.label_datafolder, args.label_site, cohort)

    if args.drop_sparse:
        cohort = drop_sparse_feature_cols(cohort, zero_ratio_threshold=args.zero_ratio_threshold)

    unit_list = [int(x) for x in args.unit_list.split(",") if x.strip()]
    df = run_xgboost_bootstrap(
        cohort_df=cohort,
        label_col=args.label_col,
        n_runs=args.n_runs,
        unit_list=unit_list,
        split_load_path=args.split_load_path,
        split_seed=args.split_seed,
        save_dir=args.save_dir,
        prefix=args.prefix,
    )

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
