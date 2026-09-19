# Ground-Truth Datasets for GP Emulator & T* Validation

Last updated: 2026-09-19

## Overview

The GP emulator predicts household-level thermal demand from physical
parameters (floor_area, wall_u, ach, wwr, form_code, hdd). T* is the
Bayesian-decomposed demand metric that strips rationing from observed
consumption. Both need ground-truthing against independent data sources.

---

## Available Datasets

### 1. EFUS 2017 (Energy Follow-Up Survey)

| Field | Value |
|---|---|
| **Path** | `data/raw/efus_2017/ukda_9434_csv_r/csv/` |
| **On disk** | Yes |
| **Licence** | UK Data Service (SN 9434), registered use |
| **Current role** | Unused |
| **Records** | 2,632 interview responses; 147 dwellings with half-hourly gas meter data |
| **Key columns** | `CaseID`, `dwtype_efus` (dwelling type), `floor6x_efus` (floor area band), `dwage_efus` (dwelling age), `WallType2x_efus` (wall type), `InsulatedWalls_efus`, `EPceeb12e_efus` (EPC band), `MainFuelUsedNM_efus` (fuel type), `tenure_efus` |
| **Meter data** | `gasdata_csv/gas_{1,2,3}.csv` — half-hourly readings (ReadingDate, MeterReading, CaseID) |

**Ground-truth potential:**
- Compare GP emulator predictions against metered annual gas for the 147
  gas-metered dwellings (after mapping EFUS dwelling categories to GP features).
- Validates physics model at dwelling level — the strongest available test.
- Limitation: EFUS dwelling descriptors are categorical bands (e.g.
  `floor6x_efus` = 1–6 ordinal), not continuous values. Requires a mapping
  table from EFUS codes to midpoint estimates compatible with GP features.
- Limitation: 147 metered dwellings is a small sample; coverage skewed
  toward certain dwelling types. No spatial identifiers (LSOA/MSOA) in the
  public release.

**Prototype loader:** `src/data/efus_loader.py`

---

### 2. DESNZ LSOA Gas Consumption (2010–2024)

| Field | Value |
|---|---|
| **Path** | `data/raw/energy/desnz_lsoa_gas_2010_2024.xlsx` |
| **On disk** | Yes |
| **Licence** | Open Government Licence v3.0 |
| **Current role** | Unused |
| **Records** | ~33,000 LSOAs per year, 15 years |
| **Key columns** | LA code, MSOA code, LSOA code, Number of meters, Total consumption (kWh), Mean consumption (kWh), Median consumption (kWh) |
| **Header row** | Row 4 (0-indexed) in each sheet |

**Ground-truth potential:**
- Aggregate LSOA to MSOA and compare against model's `y_mean` (observed
  consumption per MSOA from the synthetic population).
- Tests whether the NEED-seed-based synthetic population reproduces the
  official DESNZ aggregate — a calibration check on population synthesis,
  not on the GP emulator directly.
- Weather-corrected data, so comparable across years.
- Full national coverage; can validate every one of the 6,853 MSOAs.

**Prototype script:** `src/research/desnz_lsoa_validation.py`

---

### 3. UKHLS Wave 13 (Understanding Society)

| Field | Value |
|---|---|
| **On disk** | Yes (used by `src/research/ukhls_convergent_validation.py`) |
| **Current role** | Active — convergent validity check (Spearman r = −0.75, p = 0.020, n = 9) |

Already integrated. Regional-level gas expenditure correlated with T*.
Extension possibilities: use dwelling-level fuel expenditure for
within-region validation, though sample size per MSOA is too small for
MSOA-level analysis.

---

### 4. EHS 2020 (English Housing Survey)

| Field | Value |
|---|---|
| **Path** | `data/raw/energy/ehs_2020_*.tab` |
| **On disk** | Yes |
| **Current role** | Unused |

**Ground-truth potential:**
- Fuel poverty prevalence by region/dwelling type — can validate whether
  T*-identified high-need areas align with official fuel poverty counts.
- Dwelling fabric data (wall type, insulation, glazing) at national scale.
- Limitation: tab-separated files, needs a loader. Sample is smaller than
  EFUS for physical parameters but larger for fuel poverty indicators.

---

### 5. SERL (Smart Energy Research Lab)

| Field | Value |
|---|---|
| **On disk** | No |
| **Access** | Application required (UKDS SN 8666) |

**Ground-truth potential:**
- ~13,000 homes with half-hourly smart meter data linked to EPCs.
- The gold standard for dwelling-level validation: actual consumption +
  actual fabric data from EPC register.
- Would enable direct GP emulator validation at dwelling level with
  real EPC-derived physical parameters, not categorical proxies.
- Requires data access application and ethics approval.

---

### 6. EPC Register (Energy Performance Certificates)

| Field | Value |
|---|---|
| **On disk** | No |
| **Access** | Open Data (opendatacommunities.org) |

**Ground-truth potential:**
- Millions of individual dwelling assessments with floor area, wall type,
  heating system, and modelled energy demand.
- Could enrich the synthetic population with real dwelling-level fabric
  data instead of NEED-derived archetypes.
- Limitation: EPC models (SAP/RdSAP) are themselves simplified physics
  models — validating GP against EPC predictions is model-vs-model, not
  model-vs-measurement.

---

## Recommended Priority

1. **DESNZ LSOA** — immediate, full-coverage MSOA-level calibration check
   (prototype ready).
2. **EFUS 2017** — dwelling-level GP emulator validation for the 147
   metered homes (prototype ready, needs EFUS→GP feature mapping).
3. **SERL** — strongest possible dwelling-level validation if access is
   secured (future work).
4. **EHS 2020** — fuel poverty cross-validation (requires loader, lower
   priority).
5. **EPC Register** — fabric enrichment (large download, model-vs-model
   limitation).
