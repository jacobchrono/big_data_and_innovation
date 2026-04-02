# Synthetic Healthcare Claims Dataset with Fraud Injection
# https://chatgpt.com/share/69cc760b-06a4-83e8-90bb-7540f63d3947

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ----------------------------
# Configuration
# ----------------------------
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

N_MEMBERS = 50000
N_PROVIDERS = 10000
N_CLAIMS = 10000000

# Fraud injection rates
DUPLICATE_RATE = 0.015
EXCESSIVE_UNITS_RATE = 0.012
UPCODING_RATE = 0.010
UNBUNDLING_RATE = 0.008
PROVIDER_OUTLIER_RATE = 0.010

# ----------------------------
# Reference data
# These are illustrative and simplified.
# ----------------------------
diagnosis_codes = [
    "E11.9",    # Type 2 diabetes mellitus without complications
    "I10",      # Essential hypertension
    "J06.9",    # Acute upper respiratory infection, unspecified
    "M54.50",   # Low back pain, unspecified
    "R10.9",    # Unspecified abdominal pain
    "F41.9",    # Anxiety disorder, unspecified
    "N39.0",    # Urinary tract infection, site not specified
    "J20.9",    # Acute bronchitis, unspecified
    "R51.9",    # Headache, unspecified
    "Z00.00"    # General adult medical exam
]

# Simplified CPT-style procedure list with rough cost anchors
procedure_catalog = pd.DataFrame({
    "procedure_code": [
        "99213", "99214", "99215",  # office visits
        "80053", "83036", "85025",  # common labs
        "71046",                    # chest x-ray
        "93000",                    # ECG
        "90834",                    # psychotherapy
        "97110",                    # therapeutic exercises
        "11721",                    # nail debridement
        "36415",                    # venipuncture
        "G0439"                     # annual wellness visit
    ],
    "base_allowed": [
        95, 145, 210,
        35, 25, 22,
        85,
        55,
        120,
        40,
        60,
        8,
        175
    ],
    "typical_units": [
        1, 1, 1,
        1, 1, 1,
        1,
        1,
        1,
        2,
        1,
        1,
        1
    ],
    "category": [
        "E&M", "E&M", "E&M",
        "Lab", "Lab", "Lab",
        "Imaging",
        "Cardiology",
        "Behavioral",
        "Therapy",
        "Podiatry",
        "Lab",
        "Preventive"
    ]
})

place_of_service_options = ["11", "22", "02", "21"]  # office, outpatient, telehealth, inpatient
specialties = ["Primary Care", "Behavioral Health", "Cardiology", "Therapy", "Podiatry", "Internal Medicine"]

# ----------------------------
# Generate members and providers
# ----------------------------
members = pd.DataFrame({
    "member_id": [f"M{str(i).zfill(6)}" for i in range(1, N_MEMBERS + 1)],
    "member_age": np.random.randint(18, 90, N_MEMBERS),
    "member_gender": np.random.choice(["F", "M"], N_MEMBERS, p=[0.54, 0.46]),
    "chronic_score": np.random.poisson(1.5, N_MEMBERS)
})

providers = pd.DataFrame({
    "provider_id": [f"P{str(i).zfill(5)}" for i in range(1, N_PROVIDERS + 1)],
    "specialty": np.random.choice(specialties, N_PROVIDERS),
    "provider_risk_score": np.random.beta(2, 8, N_PROVIDERS)
})

# ----------------------------
# Generate base claims
# ----------------------------
claim_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

claims = pd.DataFrame({
    "claim_id": [f"C{str(i).zfill(8)}" for i in range(1, N_CLAIMS + 1)],
    "member_id": np.random.choice(members["member_id"], N_CLAIMS),
    "provider_id": np.random.choice(providers["provider_id"], N_CLAIMS),
    "service_date": np.random.choice(claim_dates, N_CLAIMS),
    "diagnosis_code": np.random.choice(diagnosis_codes, N_CLAIMS),
    "place_of_service": np.random.choice(place_of_service_options, N_CLAIMS, p=[0.65, 0.15, 0.10, 0.10]),
    "telehealth_flag": 0
})

claims["telehealth_flag"] = np.where(claims["place_of_service"] == "02", 1, 0)
claims["inpatient_flag"] = np.where(claims["place_of_service"] == "21", 1, 0)

claims = claims.merge(members, on="member_id", how="left")
claims = claims.merge(providers, on="provider_id", how="left")

# Assign a procedure based loosely on specialty
def choose_procedure(specialty):
    if specialty == "Primary Care":
        options = ["99213", "99214", "80053", "83036", "85025", "93000", "G0439"]
    elif specialty == "Behavioral Health":
        options = ["90834", "99213", "99214"]
    elif specialty == "Cardiology":
        options = ["93000", "99214", "99215", "71046"]
    elif specialty == "Therapy":
        options = ["97110", "99213"]
    elif specialty == "Podiatry":
        options = ["11721", "99213"]
    else:
        options = ["99213", "99214", "80053", "85025"]
    return np.random.choice(options)

claims["procedure_code"] = claims["specialty"].apply(choose_procedure)
claims = claims.merge(procedure_catalog, on="procedure_code", how="left")

# Units
claims["units"] = np.maximum(
    1,
    np.random.poisson(claims["typical_units"].fillna(1))
)
claims["units"] = claims["units"].clip(1, 6)

# Cost creation
noise = np.random.lognormal(mean=0, sigma=0.18, size=len(claims))
severity_factor = 1 + (claims["chronic_score"] * 0.03) + (claims["inpatient_flag"] * 0.25)
age_factor = 1 + ((claims["member_age"] - 50).clip(lower=0) * 0.002)

claims["allowed_amount"] = claims["base_allowed"] * claims["units"] * noise * severity_factor * age_factor
claims["allowed_amount"] = claims["allowed_amount"].round(2)

claims["billed_amount"] = (claims["allowed_amount"] * np.random.uniform(1.10, 1.45, len(claims))).round(2)
claims["paid_amount"] = (claims["allowed_amount"] * np.random.uniform(0.85, 1.00, len(claims))).round(2)

claims["fraud_label"] = 0
claims["fraud_type"] = "none"

# ----------------------------
# Inject fraud
# ----------------------------
fraud_frames = []

# 1. Duplicate billing
dup_n = int(len(claims) * DUPLICATE_RATE)
dup_rows = claims.sample(dup_n, random_state=SEED).copy()
dup_rows["claim_id"] = [f"FDUP{str(i).zfill(7)}" for i in range(1, dup_n + 1)]
dup_rows["fraud_label"] = 1
dup_rows["fraud_type"] = "duplicate_billing"
fraud_frames.append(dup_rows)

# 2. Excessive units
ex_units_n = int(len(claims) * EXCESSIVE_UNITS_RATE)
ex_units_rows = claims.sample(ex_units_n, random_state=SEED + 1).copy()
ex_units_rows["claim_id"] = [f"FEXU{str(i).zfill(7)}" for i in range(1, ex_units_n + 1)]
ex_units_rows["units"] = np.random.randint(8, 21, ex_units_n)
ex_units_rows["allowed_amount"] = (ex_units_rows["base_allowed"] * ex_units_rows["units"] * np.random.uniform(1.0, 1.3, ex_units_n)).round(2)
ex_units_rows["billed_amount"] = (ex_units_rows["allowed_amount"] * np.random.uniform(1.20, 1.60, ex_units_n)).round(2)
ex_units_rows["paid_amount"] = (ex_units_rows["allowed_amount"] * np.random.uniform(0.80, 1.00, ex_units_n)).round(2)
ex_units_rows["fraud_label"] = 1
ex_units_rows["fraud_type"] = "excessive_units"
fraud_frames.append(ex_units_rows)

# 3. Upcoding
upcoding_n = int(len(claims) * UPCODING_RATE)
upcoding_rows = claims.sample(upcoding_n, random_state=SEED + 2).copy()
upcoding_rows["claim_id"] = [f"FUPC{str(i).zfill(7)}" for i in range(1, upcoding_n + 1)]
upcoding_rows["procedure_code"] = "99215"
upcoding_rows["base_allowed"] = 210
upcoding_rows["category"] = "E&M"
upcoding_rows["units"] = 1
upcoding_rows["allowed_amount"] = (upcoding_rows["base_allowed"] * np.random.uniform(1.00, 1.20, upcoding_n)).round(2)
upcoding_rows["billed_amount"] = (upcoding_rows["allowed_amount"] * np.random.uniform(1.25, 1.60, upcoding_n)).round(2)
upcoding_rows["paid_amount"] = (upcoding_rows["allowed_amount"] * np.random.uniform(0.85, 1.00, upcoding_n)).round(2)
upcoding_rows["fraud_label"] = 1
upcoding_rows["fraud_type"] = "upcoding"
fraud_frames.append(upcoding_rows)

# 4. Unbundling, create multiple small lines for same encounter
unbundle_n = int(len(claims) * UNBUNDLING_RATE)
unbundle_seed = claims.sample(unbundle_n, random_state=SEED + 3).copy()

part_a = unbundle_seed.copy()
part_a["claim_id"] = [f"FUNA{str(i).zfill(7)}" for i in range(1, unbundle_n + 1)]
part_a["procedure_code"] = "36415"
part_a["base_allowed"] = 8
part_a["allowed_amount"] = np.random.uniform(6, 12, unbundle_n).round(2)
part_a["billed_amount"] = (part_a["allowed_amount"] * np.random.uniform(1.2, 1.5, unbundle_n)).round(2)
part_a["paid_amount"] = (part_a["allowed_amount"] * np.random.uniform(0.8, 1.0, unbundle_n)).round(2)
part_a["fraud_label"] = 1
part_a["fraud_type"] = "unbundling"

part_b = unbundle_seed.copy()
part_b["claim_id"] = [f"FUNB{str(i).zfill(7)}" for i in range(1, unbundle_n + 1)]
part_b["procedure_code"] = "85025"
part_b["base_allowed"] = 22
part_b["allowed_amount"] = np.random.uniform(18, 35, unbundle_n).round(2)
part_b["billed_amount"] = (part_b["allowed_amount"] * np.random.uniform(1.2, 1.5, unbundle_n)).round(2)
part_b["paid_amount"] = (part_b["allowed_amount"] * np.random.uniform(0.8, 1.0, unbundle_n)).round(2)
part_b["fraud_label"] = 1
part_b["fraud_type"] = "unbundling"

fraud_frames.extend([part_a, part_b])

# 5. Provider outlier behavior
prov_outlier_n = int(len(claims) * PROVIDER_OUTLIER_RATE)
high_risk_provider = providers.sample(3, random_state=SEED + 4)["provider_id"].tolist()
prov_rows = claims.sample(prov_outlier_n, random_state=SEED + 5).copy()
prov_rows["claim_id"] = [f"FPRO{str(i).zfill(7)}" for i in range(1, prov_outlier_n + 1)]
prov_rows["provider_id"] = np.random.choice(high_risk_provider, prov_outlier_n)
prov_rows["units"] = np.random.randint(4, 12, prov_outlier_n)
prov_rows["allowed_amount"] = (prov_rows["base_allowed"] * prov_rows["units"] * np.random.uniform(1.3, 1.8, prov_outlier_n)).round(2)
prov_rows["billed_amount"] = (prov_rows["allowed_amount"] * np.random.uniform(1.3, 1.7, prov_outlier_n)).round(2)
prov_rows["paid_amount"] = (prov_rows["allowed_amount"] * np.random.uniform(0.85, 1.0, prov_outlier_n)).round(2)
prov_rows["fraud_label"] = 1
prov_rows["fraud_type"] = "provider_outlier"
fraud_frames.append(prov_rows)

# Combine base + fraud
claims_all = pd.concat([claims] + fraud_frames, ignore_index=True)

# Add line_id
claims_all["line_id"] = ["L" + str(i).zfill(9) for i in range(1, len(claims_all) + 1)]

# Reorder columns
claims_all = claims_all[[
    "claim_id", "line_id", "member_id", "provider_id",
    "service_date", "place_of_service", "telehealth_flag", "inpatient_flag",
    "specialty", "diagnosis_code", "procedure_code", "category",
    "units", "billed_amount", "allowed_amount", "paid_amount",
    "member_age", "member_gender", "chronic_score",
    "provider_risk_score", "fraud_label", "fraud_type"
]]

# ----------------------------
# Feature engineering
# ----------------------------
# Same-day provider volume
provider_day_volume = (
    claims_all.groupby(["provider_id", "service_date"])
    .size()
    .reset_index(name="provider_day_claim_count")
)
claims_all = claims_all.merge(provider_day_volume, on=["provider_id", "service_date"], how="left")

# Same-day member-provider duplicate count
dup_count = (
    claims_all.groupby([
        "member_id", "provider_id", "service_date",
        "diagnosis_code", "procedure_code", "allowed_amount"
    ])
    .size()
    .reset_index(name="duplicate_signature_count")
)
claims_all = claims_all.merge(
    dup_count,
    on=["member_id", "provider_id", "service_date", "diagnosis_code", "procedure_code", "allowed_amount"],
    how="left"
)

# Provider average allowed amount
provider_avg = (
    claims_all.groupby("provider_id")["allowed_amount"]
    .mean()
    .reset_index(name="provider_avg_allowed")
)
claims_all = claims_all.merge(provider_avg, on="provider_id", how="left")

# Procedure average allowed amount
proc_avg = (
    claims_all.groupby("procedure_code")["allowed_amount"]
    .mean()
    .reset_index(name="procedure_avg_allowed")
)
claims_all = claims_all.merge(proc_avg, on="procedure_code", how="left")

claims_all["allowed_to_proc_avg_ratio"] = claims_all["allowed_amount"] / claims_all["procedure_avg_allowed"]
claims_all["billed_to_allowed_ratio"] = claims_all["billed_amount"] / claims_all["allowed_amount"]

# ----------------------------
# Rule-based detection
# ----------------------------
claims_all["rule_duplicate"] = (claims_all["duplicate_signature_count"] > 1).astype(int)
claims_all["rule_excessive_units"] = (claims_all["units"] >= 8).astype(int)
claims_all["rule_provider_volume"] = (claims_all["provider_day_claim_count"] >= 25).astype(int)
claims_all["rule_high_amount_ratio"] = (claims_all["allowed_to_proc_avg_ratio"] >= 2.0).astype(int)
claims_all["rule_high_billed_allowed"] = (claims_all["billed_to_allowed_ratio"] >= 1.5).astype(int)

rule_cols = [
    "rule_duplicate",
    "rule_excessive_units",
    "rule_provider_volume",
    "rule_high_amount_ratio",
    "rule_high_billed_allowed"
]

claims_all["rule_score"] = claims_all[rule_cols].sum(axis=1)
claims_all["rule_pred"] = (claims_all["rule_score"] >= 2).astype(int)

# ----------------------------
# Unsupervised anomaly detection
# ----------------------------
model_df = claims_all.copy()

feature_cols = [
    "units",
    "billed_amount",
    "allowed_amount",
    "paid_amount",
    "member_age",
    "chronic_score",
    "provider_risk_score",
    "provider_day_claim_count",
    "duplicate_signature_count",
    "provider_avg_allowed",
    "procedure_avg_allowed",
    "allowed_to_proc_avg_ratio",
    "billed_to_allowed_ratio",
    "telehealth_flag",
    "inpatient_flag"
]

X = model_df[feature_cols].fillna(0)

iso = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=SEED
)
iso.fit(X)

# IsolationForest returns -1 for anomaly
model_df["iso_pred"] = (iso.predict(X) == -1).astype(int)

# ----------------------------
# Optional supervised model
# Since we know the true synthetic labels, this is useful for evaluation.
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, model_df["fraud_label"], test_size=0.30, random_state=SEED, stratify=model_df["fraud_label"]
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=SEED,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# ----------------------------
# Evaluation
# ----------------------------
def summarize_performance(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame({
        "model": [name],
        "accuracy": [report["accuracy"]],
        "precision_fraud": [report["1"]["precision"]],
        "recall_fraud": [report["1"]["recall"]],
        "f1_fraud": [report["1"]["f1-score"]],
        "tn": [cm[0, 0]],
        "fp": [cm[0, 1]],
        "fn": [cm[1, 0]],
        "tp": [cm[1, 1]]
    })

rule_perf = summarize_performance("Rule-based", claims_all["fraud_label"], claims_all["rule_pred"])
iso_perf = summarize_performance("IsolationForest", model_df["fraud_label"], model_df["iso_pred"])
rf_perf = summarize_performance("RandomForest", y_test, y_pred)
rf_perf["roc_auc"] = roc_auc_score(y_test, y_prob)

performance = pd.concat([rule_perf, iso_perf, rf_perf], ignore_index=True)

# Fraud type breakdown
fraud_breakdown = (
    claims_all.groupby(["fraud_type", "fraud_label"])
    .size()
    .reset_index(name="count")
    .sort_values(["fraud_label", "count"], ascending=[False, False])
)

# Provider outlier summary
provider_summary = (
    claims_all.groupby("provider_id")
    .agg(
        total_lines=("line_id", "count"),
        avg_allowed=("allowed_amount", "mean"),
        fraud_rate=("fraud_label", "mean"),
        max_day_volume=("provider_day_claim_count", "max")
    )
    .reset_index()
    .sort_values(["fraud_rate", "avg_allowed", "max_day_volume"], ascending=False)
)

# Save outputs
claims_all.to_csv(OUTPUT_DIR / "synthetic_claims_with_fraud.csv", index=False)
performance.to_csv(OUTPUT_DIR / "model_performance_summary.csv", index=False)
fraud_breakdown.to_csv(OUTPUT_DIR / "fraud_breakdown.csv", index=False)
provider_summary.to_csv(OUTPUT_DIR / "provider_summary.csv", index=False)

# Print quick summary
print("\nDataset shape:", claims_all.shape)
print("\nFraud rate:", claims_all["fraud_label"].mean().round(4))
print("\nFraud type counts:")
print(claims_all["fraud_type"].value_counts())

print("\nPerformance summary:")
print(performance)

print("\nTop suspicious providers:")
print(provider_summary.head(10))