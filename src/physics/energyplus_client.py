import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.epw_parser import calculate_hdd_from_epw

# --- CONFIG ---
EP_EXE = Path(r"C:\EnergyPlusV25-2-0\energyplus.exe")
WEATHER_FILE = Path("data/raw/physics/Manchester_2030_ColdSnap.epw")
TEMPLATE_FILE = Path("data/raw/physics/seed_template.idf")
SIM_DIR = Path("data/raw/physics/energyplus_sims")
BASELINE_CSV = Path("data/raw/physics/physics_archetypes_baseline.csv")

# --- MATRICES ---
FORMS = {
    "Bungalow": {"floors": 1, "exposed_walls": 4, "wwr": 0.15},
    "Flat": {"floors": 1, "exposed_walls": 1, "wwr": 0.20},
    "House": {"floors": 2, "exposed_walls": 4, "wwr": 0.15},
    "Maisonette": {"floors": 2, "exposed_walls": 2, "wwr": 0.15},
}

AGES = {
    "Pre-1900": {"wall": 2.1, "roof": 2.0, "floor": 1.2, "window": 4.8, "ach": 1.5},
    "1900-1929": {"wall": 2.1, "roof": 1.5, "floor": 1.2, "window": 4.8, "ach": 1.5},
    "1930-1949": {"wall": 1.7, "roof": 1.0, "floor": 1.0, "window": 4.8, "ach": 1.2},
    "1950-1966": {"wall": 1.5, "roof": 0.7, "floor": 0.8, "window": 4.8, "ach": 1.0},
    "1967-1982": {"wall": 1.0, "roof": 0.4, "floor": 0.6, "window": 3.0, "ach": 0.8},
    "1983-1995": {"wall": 0.6, "roof": 0.3, "floor": 0.4, "window": 2.5, "ach": 0.6},
    "1996-2006": {"wall": 0.45, "roof": 0.2, "floor": 0.3, "window": 2.0, "ach": 0.5},
    "2007+": {"wall": 0.3, "roof": 0.15, "floor": 0.2, "window": 1.6, "ach": 0.5},
}


def filter_standard_types(df: pd.DataFrame, forms: dict) -> pd.DataFrame:
    """Pure function to filter dataframe."""
    return df[df["property_type"].isin(forms.keys())].copy()


def calculate_geometry(total_area: float, f_spec: dict) -> dict:
    """Pure function to calculate geometry properties."""
    footprint = total_area / f_spec["floors"]
    side = np.sqrt(footprint)
    height_per_floor = 3.0
    total_height = f_spec["floors"] * height_per_floor

    target_win_area = total_area * f_spec["wwr"]
    win_side = np.sqrt(target_win_area)
    win_side_clamped = min(win_side, side * 0.8, total_height * 0.8)

    w_x_min = (side - win_side_clamped) / 2
    w_x_max = w_x_min + win_side_clamped
    w_z_min = (total_height - win_side_clamped) / 2
    w_z_max = w_z_min + win_side_clamped

    return {
        "side": side,
        "total_height": total_height,
        "w_x_min": w_x_min,
        "w_x_max": w_x_max,
        "w_z_min": w_z_min,
        "w_z_max": w_z_max,
        "zone_volume": total_area * total_height,
    }


def fill_template(template: str, name: str, f_spec: dict, a_spec: dict, geo: dict) -> str:
    """Pure function to fill the IDF template."""
    idf = template.replace("@@ARCHETYPE_NAME@@", name)
    idf = idf.replace("@@NORTH_AXIS@@", "0.0")
    idf = idf.replace("@@PEOPLE_COUNT@@", "0.0")
    idf = idf.replace("@@LIGHTING_W@@", "0.0")
    idf = idf.replace("@@EQUIPMENT_W@@", "0.0")
    idf = idf.replace("@@WALL_R_VALUE@@", f"{1.0 / a_spec['wall']:.4f}")
    idf = idf.replace("@@ROOF_R_VALUE@@", f"{1.0 / a_spec['roof']:.4f}")
    idf = idf.replace("@@FLOOR_R_VALUE@@", f"{1.0 / a_spec['floor']:.4f}")
    idf = idf.replace("@@WINDOW_U_VALUE@@", f"{a_spec['window']:.2f}")
    idf = idf.replace("@@ZONE_VOLUME@@", f"{geo['zone_volume']:.2f}")
    idf = idf.replace("@@SIDE_LENGTH@@", f"{geo['side']:.2f}")
    idf = idf.replace("@@TOTAL_HEIGHT@@", f"{geo['total_height']:.2f}")
    idf = idf.replace("@@W_X_MIN@@", f"{geo['w_x_min']:.2f}")
    idf = idf.replace("@@W_X_MAX@@", f"{geo['w_x_max']:.2f}")
    idf = idf.replace("@@W_Z_MIN@@", f"{geo['w_z_min']:.2f}")
    idf = idf.replace("@@W_Z_MAX@@", f"{geo['w_z_max']:.2f}")

    idf = idf.replace("@@BC_EAST@@", "Outdoors" if f_spec["exposed_walls"] >= 2 else "Adiabatic")
    idf = idf.replace("@@SUN_EAST@@", "SunExposed" if f_spec["exposed_walls"] >= 2 else "NoSun")
    idf = idf.replace("@@WIND_EAST@@", "WindExposed" if f_spec["exposed_walls"] >= 2 else "NoWind")
    idf = idf.replace("@@BC_NORTH@@", "Outdoors" if f_spec["exposed_walls"] >= 3 else "Adiabatic")
    idf = idf.replace("@@SUN_NORTH@@", "SunExposed" if f_spec["exposed_walls"] >= 3 else "NoSun")
    idf = idf.replace("@@WIND_NORTH@@", "WindExposed" if f_spec["exposed_walls"] >= 3 else "NoWind")
    idf = idf.replace("@@BC_WEST@@", "Outdoors" if f_spec["exposed_walls"] >= 4 else "Adiabatic")
    idf = idf.replace("@@SUN_WEST@@", "SunExposed" if f_spec["exposed_walls"] >= 4 else "NoSun")
    idf = idf.replace("@@WIND_WEST@@", "WindExposed" if f_spec["exposed_walls"] >= 4 else "NoWind")
    return idf


def calculate_hlc(sim_df: pd.DataFrame, ach: float, zone_volume: float) -> float:
    """Pure function to calculate HLC from simulation dataframe."""
    heat_cols = [c for c in sim_df.columns if "Zone Air System Sensible Heating Rate" in c]
    temp_cols = [c for c in sim_df.columns if "Site Outdoor Air Drybulb Temperature" in c]

    if not heat_cols or not temp_cols:
        return 0.0

    heat_col = heat_cols[0]
    temp_col = temp_cols[0]

    active_sim = sim_df[sim_df[heat_col] > 0]
    if len(active_sim) == 0:
        active_sim = sim_df

    avg_heating = active_sim[heat_col].mean()
    avg_out_temp = active_sim[temp_col].mean()
    delta_t = 15.5 - avg_out_temp
    if delta_t == 0:
        delta_t = 1.0

    hlc_cond = avg_heating / delta_t
    hlc_inf = (ach * zone_volume * 1200) / 3600
    return hlc_cond + hlc_inf


def enrich_dataframe(df: pd.DataFrame, hlc_results: list[float], true_hdd: float) -> pd.DataFrame:
    """Pure function to enrich dataframe with calculated metrics without mutation."""
    df_hlc = df.assign(hlc=hlc_results)
    theoretical_gas_kwh = df_hlc["hlc"] * true_hdd * 24 * 1e-3
    return df_hlc.assign(theoretical_gas_kwh=theoretical_gas_kwh)


def read_file_content(path: Path) -> str:
    """I/O boundary."""
    with open(path) as f:
        return f.read()


def write_file_content(path: Path, content: str) -> None:
    """I/O boundary."""
    with open(path, "w") as f:
        f.write(content)


def run_energyplus(ep_exe: Path, weather_file: Path, out_dir: Path, idf_path: Path) -> None:
    """I/O boundary."""
    subprocess.run(
        [str(ep_exe), "-w", str(weather_file), "-d", str(out_dir), "-r", str(idf_path)],
        capture_output=True,
        check=True,
    )


def process_row(row: pd.Series, template: str) -> float:
    """Impure orchestrator for a single row processing."""
    name = row["Archetype"].replace(" ", "_").replace("+", "plus")
    form_name = row["property_type"]
    age_name = row["property_age"]

    f_spec = FORMS[form_name]
    a_spec = AGES[age_name]

    geo = calculate_geometry(row["Mean_Area"], f_spec)
    idf_str = fill_template(template, name, f_spec, a_spec, geo)

    idf_path = SIM_DIR / f"{name}.idf"
    write_file_content(idf_path, idf_str)

    print(f"Simulating {row['Archetype']}...")
    out_dir = SIM_DIR / name
    out_dir.mkdir(exist_ok=True, parents=True)
    run_energyplus(EP_EXE, WEATHER_FILE, out_dir, idf_path)

    csv_path = out_dir / "eplusout.csv"
    if not csv_path.exists():
        return 0.0
    sim_df = pd.read_csv(csv_path)

    total_hlc = calculate_hlc(sim_df, a_spec["ach"], geo["zone_volume"])
    print(f"  HLC: {total_hlc:.2f} W/K")

    import shutil

    shutil.rmtree(out_dir, ignore_errors=True)
    idf_path.unlink(missing_ok=True)

    return total_hlc


def process_all_rows(df: pd.DataFrame, template: str) -> list[float]:
    """Impure orchestrator."""
    return [process_row(row, template) for _, row in df.iterrows()]


def run_pipeline() -> None:
    """Main orchestrator."""
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(BASELINE_CSV)
    df = filter_standard_types(df_raw, FORMS)

    template = read_file_content(TEMPLATE_FILE)

    hlc_results = process_all_rows(df, template)

    true_hdd = calculate_hdd_from_epw(WEATHER_FILE)
    df_final = enrich_dataframe(df, hlc_results, true_hdd)

    df_final.to_csv(BASELINE_CSV, index=False)
    print("\nProgrammatic EnergyPlus Stage Complete.")


if __name__ == "__main__":
    run_pipeline()
