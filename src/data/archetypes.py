import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.getcwd())
from src.config.settings import PROCESSED_DIR, SYNTHETIC_POP_FILE


def get_archetype_counts() -> None:
    print("--- EXTRACTING REAL ARCHETYPE COUNTS ---")

    # Load synthetic population
    pop_path = PROCESSED_DIR / SYNTHETIC_POP_FILE
    if not pop_path.exists():
        print(f"Error: {pop_path} not found.")
        return

    df = pd.read_csv(pop_path)

    # Group by property type and age to get the counts
    counts = df.groupby(["property_type", "property_age"]).size().reset_index(name="count")

    # Sort and get top 10
    top_10 = counts.sort_values("count", ascending=False).head(10)

    # Load the baseline physics data to get the HLC
    physics_path = Path("data/raw/physics/physics_archetypes_baseline.csv")
    physics_df = pd.read_csv(physics_path)

    # Map specific types to baseline 'House' where applicable
    type_mapping = {
        "Semi-detached": "House",
        "Terraced": "House",
        "Detached": "House",
        "Flat": "Flat",
        "Bungalow": "Bungalow",
        "Maisonette": "Maisonette",
    }

    top_10 = top_10.assign(baseline_type=top_10["property_type"].map(type_mapping))

    # Merge to get the baseline HLC
    merged = top_10.merge(
        physics_df[["property_type", "property_age", "hlc"]],
        left_on=["baseline_type", "property_age"],
        right_on=["property_type", "property_age"],
        how="left",
        suffixes=("", "_y"),
    )

    # Clean up columns
    merged = merged[["property_type", "property_age", "count", "hlc"]]

    print("\nTop 10 Archetypes:")
    print(merged.to_string(index=False))

    # Save to file
    out_path = PROCESSED_DIR / "top_10_archetypes.txt"
    merged.to_csv(out_path, index=False, sep="\t")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    get_archetype_counts()
