import pandas as pd

df = pd.read_csv("data/raw/Sample - Superstore.csv", encoding="latin-1")

print(df.shape)     # (filas, columnas)
print(df.head())    # primeras 5 filas
df.info()           # columnas, tipos y valores no nulos

# --- Ranges of the numeric columns (min, max, mean...) ---
print(df.describe())

# --- Duplicates ---
# Row ID is excluded because it is only an index and is always different
print("Duplicates:", df.drop(columns="Row ID").duplicated().sum())

# --- Values of the category columns (to detect misspellings) ---
for col in ["Ship Mode", "Segment", "Country", "Region", "Category", "Sub-Category"]:
    print(col, df[col].nunique(), "values:", df[col].unique())


# =====================================================================
# CLEANING
# =====================================================================

# --- Remove the exact duplicate line (ignoring Row ID) ---
df = df.drop_duplicates(subset=df.columns.drop("Row ID"))

# --- Convert dates from text to datetime ---
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
print("Lines shipped before being ordered:", (df["Ship Date"] < df["Order Date"]).sum())

# --- Postal Code is a code, not a number: keep the leading zero ---
df["Postal Code"] = df["Postal Code"].astype(str).str.zfill(5)

# --- Drop columns that bring no information ---
# Country has a single value; Row ID is only an index
df = df.drop(columns=["Country", "Row ID"])

# --- Helper columns for the analysis ---
df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month

# Discount bands follow the discount values the store actually uses
print("Discount values used:", sorted(df["Discount"].unique()))
df["Discount Band"] = pd.cut(
    df["Discount"],
    bins=[-0.01, 0, 0.2, 0.4, 1],
    labels=["0%", "1-20%", "21-40%", "above 40%"],
)
# --- Save the clean dataset in a different file ---
df.to_csv("data/clean/superstore_clean.csv", index=False)
print("Clean dataset saved:", df.shape)