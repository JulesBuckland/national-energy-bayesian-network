"""CLI entry point for the ABS-UBEM inference pipeline."""

import sys
from pathlib import Path

from rich.console import Console

from src.config.settings import (
    BOUNDARIES_PATH,
    MSOA_CONFOUNDERS_NATIONAL,
    PROCESSED_DIR,
    RAW_DIR,
    SYNTHETIC_POP_FILE,
)

console = Console()

# Inputs run_national_unified_model() reads before it can sample. Checked up
# front so a clean checkout gets a list of what to obtain rather than a
# traceback from whichever loader happened to fail first.
#
# Each entry lists one or more candidate paths and is satisfied if ANY of them
# exists: the thermal-demand stage can be served either by the trained GP
# surrogate or by the analytical archetype baseline it falls back to.
REQUIRED_INPUTS = [
    (
        [PROCESSED_DIR / SYNTHETIC_POP_FILE],
        "Synthetic population",
        "python -m src.data.population  (or src.data.generate_synthetic_data)",
    ),
    (
        [MSOA_CONFOUNDERS_NATIONAL],
        "MSOA income-deprivation confounders",
        "python -m src.data.generate_synthetic_data",
    ),
    (
        [BOUNDARIES_PATH],
        "MSOA boundary geometries",
        "Download ONS MSOA (Dec 2021) boundaries - see README.md",
    ),
    (
        [
            PROCESSED_DIR / "gp_emulator.pkl",
            RAW_DIR / "physics" / "physics_archetypes_baseline.csv",
        ],
        "Thermal-demand model (GP surrogate, or archetype baseline CSV)",
        "python -m src.inference.gp_emulator  (needs EnergyPlus LHS results)",
    ),
]


def check_inputs() -> list[tuple[list[Path], str, str]]:
    """Return the entries whose candidate paths are all missing."""
    return [entry for entry in REQUIRED_INPUTS if not any(p.exists() for p in entry[0])]


def main() -> int:
    """Run the national unified model.

    Progress reporting is deliberately left to PyMC's own sampler, which
    reports real per-chain draw counts and divergences. A wrapper dashboard
    would have to invent numbers before sampling starts, which is worse than
    no dashboard.
    """
    missing = check_inputs()
    if missing:
        console.print("[bold red]Cannot start:[/bold red] required inputs are missing.\n")
        for paths, label, how in missing:
            console.print(f"  [yellow]{label}[/yellow]")
            for path in paths:
                console.print(f"    expected at: {path}")
            console.print(f"    obtain with: {how}\n")
        console.print(
            "The full national run needs licensed inputs that are not "
            "redistributable. See the Data section of README.md."
        )
        return 1

    console.print(
        "[bold cyan]ABS-UBEM: starting national unified Bayesian inference run.[/bold cyan]"
    )
    # Imported here, not at module scope, so --help-style preflight failures do
    # not pay the multi-second PyMC/PyTensor import cost.
    from src.inference.model_unified import run_national_unified_model

    run_national_unified_model()
    console.print("[bold green]Run complete.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
