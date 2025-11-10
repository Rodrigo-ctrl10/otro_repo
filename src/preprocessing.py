import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from unidecode import unidecode
import os

TARGET_COL  = "target"
PERIOD_COL  = "p_codmes"
MISSING_TH  = 0.80
RANDOM_SEED = 42
OUTPUT_DIR=r"D:\proyect_mlops\data\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df=pd.read_csv(r"D:\proyect_mlops\data\raw\Data_CU_venta.csv")
df.head()

if TARGET_COL in df.columns:
    df[TARGET_COL]=pd.to_numeric(df[TARGET_COL],errors="coerce").fillna(0).astype(int)
else:
    raise ValueError(f"No se encontró la columna {TARGET_COL}.")

df[PERIOD_COL]=pd.to_numeric(df[PERIOD_COL],errors="coerce").replace([np.inf,-np.inf],np.nan)
m = df[PERIOD_COL].mode(dropna=True)
if m.empty:
    raise ValueError("p_codmes no tiene valores válidos para calcular la moda")
fill_val = m.iloc[0]


df[PERIOD_COL] = df[PERIOD_COL].fillna(fill_val).astype("Int64")


print("Moda usada:", fill_val)
print("Nulos restantes:", df[PERIOD_COL].isna().sum())

missing_frac=df.isna().mean()
df=df.drop(columns=missing_frac[missing_frac>MISSING_TH].index)

def imputacion_por_tipo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in out.columns:
        if out[c].dtype == "object":
            s = pd.to_numeric(out[c], errors="coerce")
            
            if s.notna().mean() > 0.8:
                out[c] = s

    cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = out.select_dtypes(include=["number", "boolean"]).columns.tolist()

    if num_cols:
        medians = out[num_cols].median(numeric_only=True)
        out[num_cols] = out[num_cols].fillna(medians)

    if cat_cols:
        modes = {}
        for c in cat_cols:
            m = out[c].mode(dropna=True)
            modes[c] = m.iloc[0] if not m.empty else "NA"
        out[cat_cols] = out[cat_cols].fillna(value=modes)

    return out

df=imputacion_por_tipo(df)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = [unidecode(str(c)).lower() for c in out.columns]
    base = [re.sub(r"[^\w]+", "_", c).strip("_") for c in raw]
    seen, final_cols = {}, []
    for c in base:
        seen[c] = seen.get(c, 0)
        final_cols.append(c if seen[c]==0 else f"{c}_{seen[c]}")
        seen[c] += 1
    out.columns = final_cols
    return out

df = clean_column_names(df)

def encode_categoricals(df: pd.DataFrame, suffix="_enc") -> pd.DataFrame:
    out = df.copy()
    cat_cols = [c for c in out.columns if out[c].dtype == "object" or str(out[c].dtype) == "category"]
    for c in cat_cols:
        le = LabelEncoder()
        out[c + suffix] = le.fit_transform(out[c].astype(str))
    return out

df = encode_categoricals(df, suffix="_enc")

num_like=df.select_dtypes(include=["number","boolean"]).columns.tolist()
features_cols=[c for c in num_like if c not in (TARGET_COL, PERIOD_COL)]

for c in df[[*features_cols]].select_dtypes(include=["boolean"]).columns:
    df[c] = df[c].astype("int8")


if PERIOD_COL not in df.columns:
    raise ValueError(f"No se encontró la columna de periodo '{PERIOD_COL}'.")

last_period=int(pd.Series(df[PERIOD_COL].dropna()).max())
df_hist=df[df[PERIOD_COL]!=last_period].copy()
df_test=df[df[PERIOD_COL]==last_period].copy()

if df_hist.empty or df_test.empty:
    raise ValueError("Revisar 'p_codmes': no se encuentra, revisar")

X_hist=df_hist[features_cols].copy()
y_hist=df_hist[TARGET_COL].copy()
X_test=df_test[features_cols].copy()
y_test=df_test[TARGET_COL].copy()

X_train,X_valid,y_train,y_valid=train_test_split(
    X_hist,y_hist,
    test_size=0.3,
    random_state=RANDOM_SEED,
    stratify=df_hist[PERIOD_COL]
)

X_train.assign(target=y_train).to_csv(f"{OUTPUT_DIR}/train.csv",index=False)
X_valid.assign(target=y_valid).to_csv(f"{OUTPUT_DIR}/valid.csv",index=False)
X_test.assign(target=y_test).to_csv(f"{OUTPUT_DIR}/test.csv",index=False)

print("Tamaños ->",
      f"train: {X_train.shape}, valid: {X_valid.shape}, test: {X_test.shape}")
print("CSVs guardados en:", OUTPUT_DIR)

