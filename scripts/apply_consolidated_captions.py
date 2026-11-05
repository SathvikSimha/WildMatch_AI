"""
Consolidated caption update script for WildMatch AI.
Updates final_caption for 40 Lion, 46 Asian Elephant, and 45 Chital records
in dataset/captions/pilot_captions.csv from dataset/captions/pending_captions.json.
"""

import json
import os
import shutil
import time
import pandas as pd

CSV_PATH = "dataset/captions/pilot_captions.csv"
PENDING_JSON_PATH = "dataset/captions/pending_captions.json"
BASELINE_BACKUP_PATH = "dataset/captions/pilot_captions.csv.bak_expansion_20261105_221531"


def main():
    print("=== STEP 1: LOAD AND VALIDATE INPUTS ===")
    with open(PENDING_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    lion_captions = {int(k): v for k, v in data["lion"].items()}
    elephant_captions = {int(k): v for k, v in data["elephant"].items()}
    chital_captions = {int(k): v for k, v in data["chital"].items()}

    print(f"Loaded {len(lion_captions)} Lion captions.")
    print(f"Loaded {len(elephant_captions)} Asian Elephant captions.")
    print(f"Loaded {len(chital_captions)} Chital captions.")

    assert len(lion_captions) == 40, f"Expected 40 Lion captions, got {len(lion_captions)}"
    assert len(elephant_captions) == 46, f"Expected 46 Elephant captions, got {len(elephant_captions)}"
    assert len(chital_captions) == 45, f"Expected 45 Chital captions, got {len(chital_captions)}"
    total_to_update = len(lion_captions) + len(elephant_captions) + len(chital_captions)
    assert total_to_update == 131, f"Expected 131 updates, got {total_to_update}"

    # Confirm the 3 Chital corrected IDs are present
    assert 676538804 in chital_captions, "Corrected Chital ID 676538804 missing!"
    assert 679611988 in chital_captions, "Corrected Chital ID 679611988 missing!"
    assert 720589857 in chital_captions, "Corrected Chital ID 720589857 missing!"
    print("Confirmed all three corrected Chital IDs (676538804, 679611988, 720589857) are present.")

    all_update_dict = {**lion_captions, **elephant_captions, **chital_captions}
    target_pids = set(all_update_dict.keys())

    df = pd.read_csv(CSV_PATH)
    assert len(df) == 200, f"Expected 200 rows, got {len(df)}"
    assert df["photo_id"].nunique() == 200, "Photo IDs not unique"
    assert (df["species"].value_counts() == 50).all(), "Species counts not 50 each"
    print(f"Current {CSV_PATH} has 200 rows, 50 per species, 200 unique photo IDs.")

    # Check baseline records
    baseline_df = pd.read_csv(BASELINE_BACKUP_PATH)
    assert len(baseline_df) == 36, f"Expected 36 baseline records, got {len(baseline_df)}"
    baseline_pids = set(baseline_df["photo_id"])
    merged_base = pd.merge(baseline_df, df, on="photo_id", suffixes=("_orig", "_curr"))
    assert len(merged_base) == 36, "Not all baseline photo IDs present in current CSV"
    assert (merged_base["final_caption_orig"] == merged_base["final_caption_curr"]).all(), "Baseline captions mismatch!"
    print("Baseline 36 captions verified intact.")

    # Check Tiger expansion records
    tiger_exp_pids = set(df[(df["species"] == "Tiger") & (~df["photo_id"].isin(baseline_pids))]["photo_id"])
    assert len(tiger_exp_pids) == 33, f"Expected 33 Tiger expansion records, got {len(tiger_exp_pids)}"
    assert df[df["photo_id"].isin(tiger_exp_pids)]["final_caption"].notna().all(), "Some Tiger expansion captions missing!"
    print("Tiger 33 expansion records verified intact with non-empty final captions.")

    # Verify target IDs in CSV
    missing_pids = target_pids - set(df["photo_id"])
    assert len(missing_pids) == 0, f"Missing target photo IDs in CSV: {missing_pids}"
    target_rows = df[df["photo_id"].isin(target_pids)]
    assert len(target_rows) == 131, f"Expected 131 target rows, got {len(target_rows)}"
    assert target_rows["final_caption"].isna().all(), "Some target rows already have non-null final_caption!"
    print("All 131 target records found and verified to have blank final_caption.")

    print("\n=== STEP 2: CREATE TIMESTAMPED BACKUP ===")
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{CSV_PATH}.bak_{ts}"
    shutil.copy2(CSV_PATH, backup_path)
    assert os.path.exists(backup_path), "Backup failed!"
    print(f"Created pre-update backup: {backup_path}")

    print("\n=== STEP 3: APPLY CAPTION UPDATES ===")
    df_updated = df.copy()
    for pid, caption in all_update_dict.items():
        idx = df_updated[df_updated["photo_id"] == pid].index[0]
        df_updated.at[idx, "final_caption"] = caption

    df_updated.to_csv(CSV_PATH, index=False)
    print(f"Successfully wrote updated CSV to {CSV_PATH}")

    print("\n=== STEP 4: MANDATORY POST-UPDATE VALIDATION ===")
    df_reloaded = pd.read_csv(CSV_PATH)
    backup_reloaded = pd.read_csv(backup_path)

    # 1. Dataset structure
    assert len(df_reloaded) == 200, f"Expected 200 rows, got {len(df_reloaded)}"
    assert df_reloaded["photo_id"].nunique() == 200, "Photo IDs not unique"
    assert (df_reloaded["species"].value_counts() == 50).all(), "Species counts not 50 each"
    assert list(df_reloaded.columns) == list(backup_reloaded.columns), "Columns / schema changed!"

    # 2. Caption completeness
    assert df_reloaded["final_caption"].notna().all(), "Some final_caption values are null!"
    assert (df_reloaded["final_caption"].str.strip() != "").all(), "Some final_caption values are empty!"

    for species, count in [("Lion", 40), ("Asian Elephant", 46), ("Chital", 45)]:
        spec_expansion = df_reloaded[(df_reloaded["species"] == species) & (~df_reloaded["photo_id"].isin(baseline_pids))]
        assert len(spec_expansion) == count, f"Expected {count} {species} expansion records, got {len(spec_expansion)}"
        assert spec_expansion["final_caption"].notna().all(), f"Some {species} expansion captions missing!"
    print("Caption completeness verified: All 200 records have non-empty final captions.")

    # 3. Tiger records untouched
    tiger_curr = df_reloaded[df_reloaded["species"] == "Tiger"].sort_values("photo_id").reset_index(drop=True)
    tiger_bak = backup_reloaded[backup_reloaded["species"] == "Tiger"].sort_values("photo_id").reset_index(drop=True)
    assert tiger_curr.equals(tiger_bak), "Tiger records were modified!"
    print("Tiger verification: All 50 Tiger records (including 33 expansion records) are 100% identical to pre-update backup.")

    # 4. Baseline 36 records untouched
    base_curr = df_reloaded[df_reloaded["photo_id"].isin(baseline_pids)].sort_values("photo_id").reset_index(drop=True)
    base_bak = backup_reloaded[backup_reloaded["photo_id"].isin(baseline_pids)].sort_values("photo_id").reset_index(drop=True)
    assert base_curr.equals(base_bak), "Baseline records were modified!"
    print("Baseline preservation: All 36 baseline records are 100% identical to pre-update backup.")

    # 5. Protected fields preservation across all 200 rows
    protected_cols = ["image_id", "photo_id", "species", "image_path", "model_name", "blip_caption"]
    for col in protected_cols:
        assert (df_reloaded[col] == backup_reloaded[col]).all(), f"Protected column {col} changed!"
    print(f"Protected-field integrity: Columns {protected_cols} are 100% identical across all 200 rows.")

    # 6. Only the 131 final_caption cells differ
    diff_mask = df_reloaded["final_caption"] != backup_reloaded["final_caption"]
    assert diff_mask.sum() == 131, f"Expected exactly 131 changed cells, got {diff_mask.sum()}"
    changed_pids = set(df_reloaded.loc[diff_mask, "photo_id"])
    assert changed_pids == target_pids, "Changed photo IDs do not match target photo IDs!"
    print("Update precision verified: Exactly the 131 approved final_caption cells were updated.")

    print("\nALL POST-UPDATE VALIDATIONS PASSED SUCCESSFULLY!")
    print(f"BACKUP_FILE={backup_path}")


if __name__ == "__main__":
    main()
