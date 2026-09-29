import pandera as pa
from pandera import Check, Column

# Data Contract for the synthetic population DataFrame
population_schema = pa.DataFrameSchema(
    {
        "msoa21cd": Column(
            str,
            checks=[
                Check(
                    lambda s: s.str.startswith(("E02", "W02")),
                    error="MSOA code must start with E02 or W02",
                ),
                Check(lambda s: s.str.len() == 9, error="MSOA code must be exactly 9 characters"),
            ],
            nullable=False,
        ),
        "property_type": Column(
            str,
            Check.isin(["House", "Flat", "Bungalow"]),
            nullable=False,
        ),
        "tenure": Column(
            str,
            Check.isin(["Owned", "Social", "Private"]),
            nullable=False,
        ),
        "form_code": Column(
            int,
            Check.isin([0, 1, 2, 3]),
            nullable=False,
        ),
        "empirical_thermal_kwh": Column(
            float,
            Check.greater_than(0.0),
            nullable=False,
        ),
        "empirical_gas_kwh": Column(
            float,
            Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
        "floor_area": Column(
            float,
            Check.greater_than_or_equal_to(20.0),
            nullable=False,
        ),
        "wall_u": Column(
            float,
            Check.in_range(0.05, 3.5),
            nullable=False,
        ),
        "ach": Column(
            float,
            Check.in_range(0.10, 4.0),
            nullable=False,
        ),
        "wwr": Column(
            float,
            Check.in_range(0.01, 0.50),
            nullable=False,
        ),
    },
    coerce=True,
    strict=False,  # Allow other columns to exist without failing
)
