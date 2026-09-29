"""Filesystem layout for the project, selected by run mode."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class PathConfig:
    """Every data path the pipeline reads or writes.

    ``test_mode`` points at the tiny fixture data under ``tests/fixtures``;
    ``use_fake_city`` points at the synthetic city under ``data/*/fake``.
    Paths are derived on access and nothing touches the filesystem here, so
    creating the directories is the caller's job (see ``settings.py``).
    """

    test_mode: bool = False
    use_fake_city: bool = False
    base_dir: Path = field(default=BASE_DIR)

    @property
    def raw_dir(self) -> Path:
        if self.test_mode:
            return self.base_dir / "tests" / "fixtures" / "raw"
        if self.use_fake_city:
            return self.base_dir / "data" / "raw" / "fake"
        return self.base_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        if self.test_mode:
            return self.base_dir / "tests" / "fixtures" / "processed"
        if self.use_fake_city:
            return self.base_dir / "data" / "processed" / "fake"
        return self.base_dir / "data" / "processed"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def regional_traces_dir(self) -> Path:
        return self.processed_dir / "regional_traces"

    @property
    def lad_traces_dir(self) -> Path:
        return self.processed_dir / "lad_traces"

    @property
    def need_microdata_path(self) -> Path:
        if self.use_fake_city:
            return self.raw_dir / "fake_need_seed.csv"
        return self.raw_dir / "energy" / "need_2024_official_50k.csv"

    @property
    def census_constraints_path(self) -> Path:
        return self.raw_dir / "census" / "census_2021_msoa_housing.csv"

    @property
    def imd_path(self) -> Path:
        return self.processed_dir / "msoa_imd_official.csv"

    @property
    def physics_archetypes_path(self) -> Path:
        return self.raw_dir / "physics" / "physics_archetypes_baseline.csv"

    @property
    def census_housing_national(self) -> Path:
        if self.use_fake_city:
            return self.raw_dir / "fake_census_housing.csv"
        census = self.raw_dir / "census"
        candidates = (
            census / "ts044_bulk_extracted" / "census2021-ts044-msoa.csv",
            census / "ts044_extracted" / "census2021-ts044-msoa.csv",
        )
        for path in candidates:
            if path.exists():
                return path
        return census / "census2021-ts044-msoa.csv"

    @property
    def census_tenure_national(self) -> Path:
        if self.use_fake_city:
            return self.raw_dir / "fake_census_tenure.csv"
        return self.raw_dir / "census" / "TS054-2021-4-filtered-2026-02-27T03_51_51Z.csv"

    @property
    def msoa_confounders_national(self) -> Path:
        if self.use_fake_city:
            return self.processed_dir / "fake_msoa_confounders.csv"
        return self.processed_dir / "msoa_confounders_national.csv"

    @property
    def lookup_path(self) -> Path:
        if self.use_fake_city:
            return self.raw_dir / "fake_lookup.csv"
        primary = self.raw_dir / "spatial" / "lookup.csv"
        return primary if primary.exists() else self.raw_dir / "lookup.csv"

    @property
    def lad_lookup_path(self) -> Path:
        return self.processed_dir / "msoa_lad_lookup.csv"

    @property
    def msoa_region_lookup(self) -> Path:
        if self.use_fake_city:
            return self.processed_dir / "fake_msoa_region_lookup.csv"
        return self.processed_dir / "msoa_region_lookup.csv"

    @property
    def boundaries_path(self) -> Path:
        if self.use_fake_city:
            return self.raw_dir / "fake_msoa_boundaries.gpkg"
        return self.raw_dir / "spatial" / "msoa dec 2021 boundaries.gpkg"
