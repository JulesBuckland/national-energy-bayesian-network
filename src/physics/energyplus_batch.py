"""
src/physics/energyplus_batch.py
==============================
Runs EnergyPlus for every point in the LHS design files produced by
src/inference/lhs_sampler.py.

Parallelised across available CPU cores via ProcessPoolExecutor.
Supports resume: already-completed runs are skipped.

Inputs:  data/raw/physics/lhs_designs/lhs_*.csv
Outputs: data/raw/physics/lhs_results/lhs_results_{archetype}.csv
         data/raw/physics/lhs_results_combined.csv
"""

import subprocess
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import concurrent.futures
import math
import argparse
import re
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("EnergyPlusLHSBatch")

from src.config.settings import RAW_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PHYSICS_DIR  = RAW_DIR / "physics"
LHS_DIR      = PHYSICS_DIR / "lhs_designs"
RESULTS_DIR  = PHYSICS_DIR / "lhs_results"
SIM_DIR      = PHYSICS_DIR / "lhs_energyplus_sims"
TEMPLATE_FILE = PHYSICS_DIR / "seed_template.idf"
EP_EXE        = Path(r"C:\EnergyPlusV25-2-0\energyplus.exe")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SIM_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT_PER_FLOOR = 3.0

# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------
def _bc(exposed_walls: int, wall_idx: int) -> Tuple[str, str, str]:
    """Return (boundary, sun, wind) for wall_idx given exposed_walls count."""
    if wall_idx < exposed_walls:
        return "Outdoors", "SunExposed", "WindExposed"
    return "Adiabatic", "NoSun", "NoWind"

def calculate_occupants(floor_area: float) -> float:
    """Calculate SAP 2012 Occupancy based on Floor Area."""
    if floor_area > 13.9:
        return 1 + 1.76 * (1 - math.exp(-0.000349 * (floor_area - 13.9)**2)) + 0.0013 * (floor_area - 13.9)
    return 1.0

def build_idf_string(template_str: str, run_id: str, floor_area: float, wall_u: float, ach: float,
                     wwr: float, floors: int, exposed_walls: int, north_axis: float,
                     occupants: float, lights_w: float, equip_w: float) -> str:
    """Pure function to build IDF string from a template string."""
    footprint = floor_area / floors
    side = math.sqrt(footprint)
    total_height = floors * HEIGHT_PER_FLOOR
    zone_vol = footprint * HEIGHT_PER_FLOOR

    wall_area_total = side * total_height * exposed_walls
    window_area     = wall_area_total * wwr
    opaque_wall     = wall_area_total - window_area

    win_side = math.sqrt(window_area)
    win_side = min(win_side, side * 0.8, total_height * 0.8)
    w_x_min  = (side - win_side) / 2
    w_x_max  = w_x_min + win_side
    w_z_min  = (total_height - win_side) / 2
    w_z_max  = w_z_min + win_side

    r_wall   = 1.0 / wall_u

    idf = template_str.replace("@@ARCHETYPE_NAME@@", run_id)
    idf = idf.replace("@@NORTH_AXIS@@",     f"{north_axis:.1f}")
    idf = idf.replace("@@PEOPLE_COUNT@@",   f"{occupants:.2f}")
    idf = idf.replace("@@LIGHTING_W@@",     f"{lights_w:.2f}")
    idf = idf.replace("@@EQUIPMENT_W@@",    f"{equip_w:.2f}")
    
    idf = idf.replace("@@WALL_R_VALUE@@",   f"{r_wall:.4f}")
    idf = idf.replace("@@ROOF_R_VALUE@@",   "1.0000")
    idf = idf.replace("@@FLOOR_R_VALUE@@",  "0.8333")
    idf = idf.replace("@@WINDOW_U_VALUE@@", f"{2.0:.2f}")
    idf = idf.replace("@@ZONE_VOLUME@@",    f"{zone_vol:.2f}")
    idf = idf.replace("@@SIDE_LENGTH@@",    f"{side:.2f}")
    idf = idf.replace("@@TOTAL_HEIGHT@@",   f"{total_height:.2f}")
    idf = idf.replace("@@W_X_MIN@@",        f"{w_x_min:.2f}")
    idf = idf.replace("@@W_X_MAX@@",        f"{w_x_max:.2f}")
    idf = idf.replace("@@W_Z_MIN@@",        f"{w_z_min:.2f}")
    idf = idf.replace("@@W_Z_MAX@@",        f"{w_z_max:.2f}")

    bc_e, sun_e, wind_e = _bc(exposed_walls, 1)
    bc_n, sun_n, wind_n = _bc(exposed_walls, 2)
    bc_w, sun_w, wind_w = _bc(exposed_walls, 3)
    idf = idf.replace("@@BC_EAST@@",   bc_e).replace("@@SUN_EAST@@",  sun_e).replace("@@WIND_EAST@@",  wind_e)
    idf = idf.replace("@@BC_NORTH@@",  bc_n).replace("@@SUN_NORTH@@", sun_n).replace("@@WIND_NORTH@@", wind_n)
    idf = idf.replace("@@BC_WEST@@",   bc_w).replace("@@SUN_WEST@@",  sun_w).replace("@@WIND_WEST@@",  wind_w)
    return idf

def generate_task_from_row(row: pd.Series, resume: bool) -> Dict[str, Any]:
    """Pure function to map a row to a task definition."""
    slug = (str(row["archetype"])
            .replace(" ", "_")
            .replace("/", "-")
            .replace("+", "plus"))
    run_id = f"{slug}_s{int(row['sample_id']):04d}"
    
    return {
        "run_id":        run_id,
        "archetype":     row["archetype"],
        "property_type": row["property_type"],
        "age_band":      row["age_band"],
        "floor_area":    float(row["floor_area"]),
        "wall_u":        float(row["wall_u"]),
        "ach":           float(row["ach"]),
        "wwr":           float(row["wwr"]),
        "form_code":     int(row["form_code"]),
        "floors":        int(row["floors"]),
        "exposed_walls": int(row["exposed_walls"]),
        "sample_id":     int(row["sample_id"]),
        "resume":        resume,
        **({"city": row["city"]} if "city" in row else {}),
        **({"hdd": float(row["hdd"])} if "hdd" in row else {}),
    }

def generate_tasks_from_df(df: pd.DataFrame, resume: bool) -> List[Dict[str, Any]]:
    """Pure function returning a new list of tasks."""
    return [generate_task_from_row(row, resume) for _, row in df.iterrows()]

# ---------------------------------------------------------------------------
# Side-effect Functions (I/O)
# ---------------------------------------------------------------------------
def _parse_heating_kwh(out_dir: Path) -> float:
    html_path = out_dir / "eplustbl.htm"
    if not html_path.exists():
        raise RuntimeError(f"FATAL: EnergyPlus failed! No output HTML found in {out_dir}")
    with open(html_path, "r", errors="ignore") as f:
        content = f.read()
    match = re.search(r"End Uses.*?<table.*?>(.*?)</table>", content, re.DOTALL | re.IGNORECASE)
    if match:
        heating_row = re.search(r"<tr>\s*<td[^>]*>\s*Heating\s*</td>(.*?)</tr>", match.group(1), re.DOTALL | re.IGNORECASE)
        if heating_row:
            cols = re.findall(r"<td[^>]*>(.*?)</td>", heating_row.group(1), re.DOTALL)
            total_gj = 0.0
            for c in cols:
                try: 
                    total_gj += float(c.strip())
                except ValueError as e: 
                    pass
            return total_gj * 277.778 # Convert GJ to kWh
        else:
            raise RuntimeError(f"Could not find Heating row in eplustbl.htm for {out_dir}")
    else:
        raise RuntimeError(f"Could not find End Uses table in eplustbl.htm for {out_dir}")

def read_template_file(template_path: Path) -> str:
    with open(template_path, "r") as f:
        return f.read()

def write_idf_file(idf_path: Path, idf_str: str) -> None:
    with open(idf_path, "w") as f:
        f.write(idf_str)

def _run_single_axis(axis: int, run_id: str, args: dict, template_str: str, weather_file: Path, out_dir: Path) -> float:
    run_axis_id = f"{run_id}_axis{axis}"
    axis_dir = out_dir / run_axis_id
    axis_dir.mkdir(parents=True, exist_ok=True)
    idf_path = axis_dir / f"{run_axis_id}.idf"
    
    occupants = calculate_occupants(args["floor_area"])
    lights_w = args["floor_area"] * 2.0
    equip_w = args["floor_area"] * 3.0
    
    idf_str = build_idf_string(
        template_str=template_str,
        run_id=run_axis_id,
        floor_area=args["floor_area"],
        wall_u=args["wall_u"],
        ach=args["ach"],
        wwr=args["wwr"],
        floors=args["floors"],
        exposed_walls=args["exposed_walls"],
        north_axis=axis,
        occupants=occupants,
        lights_w=lights_w,
        equip_w=equip_w
    )
    
    write_idf_file(idf_path, idf_str)

    try:
        subprocess.run(
            [str(EP_EXE), "-w", str(weather_file), "-d", str(axis_dir), str(idf_path)],
            capture_output=True, timeout=120
        )
        val = _parse_heating_kwh(axis_dir)
        import shutil
        shutil.rmtree(axis_dir, ignore_errors=True)
        return val
    except Exception as e:
        logger.warning(f"FAILED {run_axis_id}: {e}")
        import shutil
        shutil.rmtree(axis_dir, ignore_errors=True)
        return 0.0

def run_single(args: dict) -> dict:
    run_id   = args["run_id"]
    out_dir  = SIM_DIR / run_id
    
    result_file = out_dir / "result.json"
    if args.get("resume", False) and result_file.exists():
        with open(result_file) as f:
            return json.load(f)

    city = args.get("city", "Manchester")
    weather_file = PHYSICS_DIR / f"{city}_2030_ColdSnap.epw"
    template_str = read_template_file(TEMPLATE_FILE)

    axes = [0, 90, 180, 270]
    t_h_runs = [_run_single_axis(axis, run_id, args, template_str, weather_file, out_dir) for axis in axes]
    valid_runs = [v for v in t_h_runs if v > 0.0]

    T_h = sum(valid_runs) / max(1, len(valid_runs)) if valid_runs else 0.0

    result = {**args, "T_h": T_h, "status": "ok" if valid_runs else "error"}
    with open(result_file, "w") as f:
        json.dump(result, f)

    return result

def check_placeholder_weather(physics_dir: Path) -> bool:
    """Return whether a known placeholder EPW file is present."""
    return any(wf.stat().st_size == 1546562 for wf in physics_dir.glob("*.epw"))


def _validate_weather_files(physics_dir: Path, cities: set[str]) -> None:
    """Fail before scheduling simulations if any requested weather input is bad."""
    if check_placeholder_weather(physics_dir):
        raise ValueError(
            "Placeholder weather files (1,546,562 bytes) detected in "
            f"{physics_dir}; download real EPW files before running simulations."
        )

    missing = [
        city for city in sorted(cities)
        if not (physics_dir / f"{city}_2030_ColdSnap.epw").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing EPW weather files for: " + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_lhs_batch(max_workers: int = 6, check_completeness: bool = False, resume: bool = False):
    design_files = sorted(LHS_DIR.glob("lhs_*.csv"))
    if not design_files:
        logger.error(f"No LHS design files found in {LHS_DIR}. Run 'python -m src.inference.lhs_sampler' first.")
        return

    if not EP_EXE.exists():
        logger.error(f"EnergyPlus executable not found at {EP_EXE}")
        return

    logger.info(f"Found {len(design_files)} archetype design files.")

    all_tasks_nested = [generate_tasks_from_df(pd.read_csv(f), resume) for f in design_files]
    all_tasks = [task for sublist in all_tasks_nested for task in sublist]

    logger.info(f"Total runs: {len(all_tasks)}")

    if check_completeness:
        done  = sum(1 for t in all_tasks if (SIM_DIR / t["run_id"] / "result.json").exists())
        total = len(all_tasks)
        logger.info(f"Completeness: {done}/{total} ({100*done/total:.1f}%)")
        return

    # Validate all requested weather inputs before launching any workers. The
    # previous code only logged a placeholder warning and continued, allowing a
    # complete-looking result file to contain failed/zero-demand runs.
    requested_cities = {
        str(task["city"]).strip()
        for task in all_tasks
        if task.get("city") is not None and not pd.isna(task.get("city"))
    }
    requested_cities.discard("")
    if not requested_cities:
        requested_cities.add("Manchester")
    _validate_weather_files(PHYSICS_DIR, requested_cities)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single, task) for task in all_tasks]
        results = [f.result() for f in tqdm(concurrent.futures.as_completed(futures), total=len(all_tasks), desc="Simulating") if not f.exception()]

    df_all = pd.DataFrame(results)
    combined_path = PHYSICS_DIR / "lhs_results_combined.csv"
    df_all.to_csv(combined_path, index=False)
    logger.info(f"Saved combined results -> {combined_path}")

    for arch, grp in df_all.groupby("archetype"):
        slug = arch.replace(" ", "_").replace("/", "-").replace("+", "plus")
        out_path = RESULTS_DIR / f"lhs_results_{slug}.csv"
        grp.to_csv(out_path, index=False)

    ok_count = (df_all["status"] == "ok").sum()
    logger.info(f"Done. {ok_count}/{len(df_all)} runs succeeded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",             type=int, default=6,
                        help="Number of parallel EnergyPlus workers (default 6)")
    parser.add_argument("--check-completeness",  action="store_true",
                        help="Report completion status without running new simulations.")
    parser.add_argument("--resume",              action="store_true",
                        help="Skip simulations if result.json already exists.")
    args = parser.parse_args()
    run_lhs_batch(max_workers=args.workers, check_completeness=args.check_completeness, resume=args.resume)
