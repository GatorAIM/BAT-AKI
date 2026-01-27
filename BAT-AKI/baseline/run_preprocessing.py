import argparse
import os
import pickle

import pandas as pd

from function.preprocessing import (
    attach_module_id,
    convert_to_retain_format,
    filter_dx_latest_per_code,
    load_cohort_addlabel,
    load_medcode_description,
    load_parameters,
    load_tables,
    sample_loaded_tables_by_cohort,
)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--datafolder2",
        default="./",
        help="",
    )
    parser.add_argument(
        "--cohort_path",
        default="./",
        help="",
    )
    parser.add_argument(
        "--cohort_datafolder",
        default="./",
        help="",
    )
    parser.add_argument(
        "--site",
        default="MASKED_VALUE",
        help="",
    )
    parser.add_argument(
        "--retain_output",
        default=None,
        help="",
    )
    parser.add_argument(
        "--sampled_output_dir",
        default=None,
        help="",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=int,
        help="",
    )
    parser.add_argument(
        "--parameters_path",
        default="./",
        help="",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datafolder2 = args.datafolder2
    cohort_path = args.cohort_path

    load_map, table_config, exclude_dx_codes = load_parameters(args.parameters_path)
    loaded_tables = load_tables(datafolder2, load_map)
    medcode_df = load_medcode_description(datafolder2)
    loaded_tables = attach_module_id(loaded_tables, medcode_df)

    sampled_keys_df, loaded_tables_sampled = sample_loaded_tables_by_cohort(
        cohort_path=cohort_path,
        loaded_tables=loaded_tables,
    )

    if "dx" in loaded_tables_sampled:
        loaded_tables_sampled["dx"] = filter_dx_latest_per_code(
            loaded_tables_sampled["dx"],
            exclude_dx_codes,
        )

    demo_df = loaded_tables.get("demo", pd.DataFrame())
    retain_ready_data = convert_to_retain_format(
        loaded_tables_sampled,
        table_config,
        demo_df,
        maxlen=args.maxlen,
    )

    retain_output = args.retain_output or os.path.join(datafolder2, "./")
    with open(retain_output, "wb") as f:
        pickle.dump(retain_ready_data, f)
    print(f"RETAIN input saved at: {retain_output}")

    df_csv = pd.read_csv(cohort_path)
    print(f"Loaded CSV: {cohort_path}, shape={df_csv.shape}")
    print(df_csv[["PATID", "ENCOUNTERID"]].head())

    cohort = load_cohort_addlabel(
        datafolder=args.cohort_datafolder,
        site=args.site,
    )
    cohort = cohort.merge(
        df_csv[["PATID", "ENCOUNTERID"]].drop_duplicates(),
        on=["PATID", "ENCOUNTERID"],
        how="inner",
    )

    flag_death_df = cohort[["PATID", "ENCOUNTERID", "FLAG", "death90"]]
    sampled_keys_df = sampled_keys_df.merge(flag_death_df, on=["PATID", "ENCOUNTERID"], how="left")

    output_dir = args.sampled_output_dir or datafolder2
    sampled_keys_df.to_csv(os.path.join(output_dir, "."), index=False)
    sampled_keys_df.to_pickle(os.path.join(output_dir, "."))
    print(f"sampled_keys_df saved with shape: {sampled_keys_df.shape}")


if __name__ == "__main__":
    main()
