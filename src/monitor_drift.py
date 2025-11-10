import os
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = r"D:\proyect_mlops\data\processed"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
VALID_CSV = f"{DATA_DIR}/valid.csv"
TEST_CSV  = f"{DATA_DIR}/test.csv"
OUTPUT_DIR = r"D:\proyect_mlops\models"

TARGET_COL = "target"

QUANTILES = 1000 

# =========================
# Utilidades de histogramas
# =========================
def hist_probs(arr: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Frecuencia relativa en cada bin definido por breaks."""
    counts = np.histogram(arr, bins=breaks)[0].astype(float)
    total = counts.sum()
    if total == 0:
        # Si no hay datos (todo NaN), devuelve uniforme para no romper
        return np.ones_like(counts) / len(counts)
    probs = counts / total
    # Evitar ceros exactos (para log)
    return np.clip(probs, 1e-6, 1.0)

def breaks_from_ref(ref_values: np.ndarray, buckets: int) -> np.ndarray:
    """
    Bordes por cuantiles del conjunto de referencia (train).
    Se añaden -inf y +inf y se eliminan duplicados.
    """
    # cuantiles internos + extremos
    q = np.linspace(0, 100, buckets + 1)
    br = np.percentile(ref_values, q)
    # eliminar duplicados
    br = np.unique(br)
    if br.size < 2:
        # columna casi/constante: crea un bin mínimo
        v = br[0] if br.size == 1 else float(ref_values[0])
        br = np.array([v - 1e-6, v + 1e-6])
    # añadir -inf/inf como extremos
    br[0] = br[0]  # ya está el extremo inferior real
    br = np.concatenate(([-np.inf], br[1:-1], [np.inf])) if br.size > 2 else np.array([-np.inf, np.inf])
    return br

# =========================
# Métricas de drift
# =========================
def psi_score(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    p_ref = np.clip(p_ref, 1e-6, 1.0)
    p_cur = np.clip(p_cur, 1e-6, 1.0)
    return float(np.sum((p_ref - p_cur) * np.log(p_ref / p_cur)))

def kl_divergence(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    p_ref = np.clip(p_ref, 1e-6, 1.0)
    p_cur = np.clip(p_cur, 1e-6, 1.0)
    return float(np.sum(p_ref * np.log(p_ref / p_cur)))

def calculate_psi(expected_array: np.ndarray, actual_array: np.ndarray, buckets: int) -> tuple[float, float]:
    br = breaks_from_ref(expected_array, buckets)
    p_ref = hist_probs(expected_array, br)
    p_cur = hist_probs(actual_array,  br)
    return psi_score(p_ref, p_cur), kl_divergence(p_ref, p_cur)

def psi_alert(v: float) -> str:
    if not np.isfinite(v): return "n/a"
    if v < 0.05: return "check"
    if v < 0.15: return "regular"
    return "warning"

def kl_alert(v: float) -> str:
    if not np.isfinite(v): return "n/a"
    if v < 0.05: return "check"
    if v < 0.15: return "regular"
    return "warning"

def build_drift_report(df_ref: pd.DataFrame, df_cur: pd.DataFrame, feature_list: list[str], quantiles: int, split_name: str) -> pd.DataFrame:
    rows = []
    for col in feature_list:
        ref_vals = df_ref[col].dropna().values
        cur_vals = df_cur[col].dropna().values

        if ref_vals.size == 0 or cur_vals.size == 0:
            psi_val = np.nan
            kl_val  = np.nan
            ref_count = int(ref_vals.size)
            cur_count = int(cur_vals.size)
        else:
            psi_val, kl_val = calculate_psi(ref_vals, cur_vals, quantiles)
            ref_count = int(np.isfinite(ref_vals).sum())
            cur_count = int(np.isfinite(cur_vals).sum())

        rows.append({
            "split": split_name,
            "feature": col,
            "psi": psi_val,
            "psi_alert": psi_alert(psi_val),
            "kl_div": kl_val,
            "kl_alert": kl_alert(kl_val),
            "ref_count": ref_count,
            "cur_count": cur_count,
        })
    return pd.DataFrame(rows)

# =========================
# Carga de datos
# =========================
train = pd.read_csv(TRAIN_CSV)
valid = pd.read_csv(VALID_CSV)
test  = pd.read_csv(TEST_CSV)

# Solo columnas numéricas y comunes, excluyendo el target
common = train.columns.intersection(valid.columns).intersection(test.columns)
common_cols = [c for c in common if c != TARGET_COL and pd.api.types.is_numeric_dtype(train[c])]

# =========================
# Cálculo de reportes
# =========================
report_valid = build_drift_report(train, valid, common_cols, QUANTILES, "valid vs train")
report_test  = build_drift_report(train, test,  common_cols, QUANTILES, "test vs train")

# =========================
# Guardado de salidas
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_valid = os.path.join(OUTPUT_DIR, f"drift_valid_vs_train_{ts}.csv")
out_test  = os.path.join(OUTPUT_DIR, f"drift_test_vs_train_{ts}.csv")

report_valid.sort_values(["split","feature"]).to_csv(out_valid, index=False)
report_test.sort_values(["split","feature"]).to_csv(out_test, index=False)

print("✅ Archivos generados:")
print("-", out_valid)
print("-", out_test)

print("\n== Preview VALID ==")
print(report_valid.head(12).to_string(index=False))
print("\n== Preview TEST ==")
print(report_test.head(12).to_string(index=False))

print("monitoreo exitoso")