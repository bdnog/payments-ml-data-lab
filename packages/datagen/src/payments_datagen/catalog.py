"""Static reference data: merchant categories, synthetic BINs and countries."""

import polars as pl

# mcc, label used in merchant names, share of merchants, median ticket (minor units),
# probability that a purchase is in person (pos), high-risk category.
_MCC_ROWS = [
    ("5411", "Grocer", 0.14, 9_000, 0.80, False),
    ("5812", "Diner", 0.10, 6_500, 0.80, False),
    ("5814", "Snacks", 0.08, 3_000, 0.85, False),
    ("5541", "Fuel", 0.06, 15_000, 0.90, False),
    ("5311", "Department", 0.06, 22_000, 0.50, False),
    ("4111", "Transit", 0.06, 1_500, 0.30, False),
    ("4900", "Utilities", 0.05, 14_000, 0.00, False),
    ("5651", "Apparel", 0.07, 16_000, 0.50, False),
    ("5912", "Pharmacy", 0.06, 4_500, 0.60, False),
    ("7011", "Hotel", 0.03, 55_000, 0.20, False),
    ("5732", "Electronics", 0.06, 65_000, 0.20, True),
    ("5944", "Jewelers", 0.03, 90_000, 0.30, True),
    ("5816", "Games", 0.07, 4_000, 0.00, True),
    ("5967", "Direct", 0.04, 9_000, 0.00, True),
    ("4829", "Remit", 0.03, 50_000, 0.05, True),
]

MCC_TABLE = pl.DataFrame(
    _MCC_ROWS,
    schema={
        "mcc": pl.String,
        "label": pl.String,
        "weight": pl.Float64,
        "median_ticket_minor": pl.Int64,
        "p_pos": pl.Float64,
        "high_risk": pl.Boolean,
    },
    orient="row",
)

NAME_PREFIXES = [
    "Alder", "Birch", "Cedar", "Dune", "Ember", "Fjord", "Grove", "Harbor",
    "Iris", "Juniper", "Kestrel", "Lumen", "Maple", "Nimbus", "Orchid", "Pine",
]  # fmt: skip

HOME_COUNTRIES = ["BR", "AR", "US", "PT", "CL", "MX"]
HOME_COUNTRY_WEIGHTS = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
BINS_PER_FOREIGN_COUNTRY = 2
BR_BIN_COUNT = 30  # 30 BR + 5 countries x 2 = 40 synthetic BINs

# Countries an IP address can show up in (travel, proxies, fraud).
FOREIGN_IP_COUNTRIES = ["US", "PT", "AR", "CL", "MX", "ES", "FR", "RU", "NG", "VN"]
