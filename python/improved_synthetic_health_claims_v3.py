# Jacob Clement
# https://chatgpt.com/share/69cc760b-06a4-83e8-90bb-7540f63d3947
# powershell command to run:
# cd "C:\Users\jakeq\OneDrive\Documents\GitHub\big_data_and_innovation\python"
# python .\improved_synthetic_health_claims_v3.py --n_encounters 1000000 --n_members 120000 --n_providers 6000 --chunk_size 100000
# 04/01/2026
# Improved Synthetic Healthcare Claims Dataset with Fraud Injection and Detection Features:


from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data"

# -----------------------------
# Reference tables
# -----------------------------
DIAGNOSIS_CODES = np.array([
    "E11.9", "I10", "E78.5", "J06.9", "J18.9", "J45.909", "M54.50", "M25.561",
    "R07.9", "R10.9", "R51.9", "N39.0", "K21.9", "K52.9", "F41.9", "F32.A",
    "I25.10", "I48.91", "Z00.00", "Z00.01", "Z12.11", "Z23", "Z79.4", "R73.03",
    "G47.33", "R06.02", "E66.9", "B34.9", "J20.9", "S93.401A", "M17.11", "R11.2",
], dtype=object)

PROFESSIONAL_PROCEDURES = pd.DataFrame([
    ("99212", 70.0, 1, "E&M"), ("99213", 95.0, 1, "E&M"), ("99214", 145.0, 1, "E&M"),
    ("99215", 210.0, 1, "E&M"), ("93000", 55.0, 1, "Cardiology"), ("71046", 85.0, 1, "Imaging"),
    ("80048", 28.0, 1, "Lab"), ("80053", 35.0, 1, "Lab"), ("83036", 25.0, 1, "Lab"),
    ("85025", 22.0, 1, "Lab"), ("84443", 30.0, 1, "Lab"), ("36415", 8.0, 1, "Lab"),
    ("90834", 120.0, 1, "Behavioral"), ("90837", 165.0, 1, "Behavioral"),
    ("97110", 40.0, 2, "Therapy"), ("97112", 45.0, 2, "Therapy"), ("97140", 38.0, 2, "Therapy"),
    ("11721", 60.0, 1, "Podiatry"), ("G0438", 160.0, 1, "Preventive"), ("G0439", 175.0, 1, "Preventive"),
    ("90686", 32.0, 1, "Preventive"), ("90471", 28.0, 1, "Preventive"), ("81002", 12.0, 1, "Lab"),
    ("20550", 110.0, 1, "Procedure"), ("20610", 135.0, 1, "Procedure"), ("12002", 185.0, 1, "Procedure"),
], columns=["procedure_code", "base_allowed", "typical_units", "category"])

INSTITUTIONAL_LINES = pd.DataFrame([
    ("0450", "99284", 850.0, 1, "ED"), ("0450", "99285", 1400.0, 1, "ED"),
    ("0360", "27447", 12000.0, 1, "OR"), ("0360", "47562", 6500.0, 1, "OR"),
    ("0250", "", 75.0, 3, "Pharmacy"), ("0270", "", 60.0, 2, "Supplies"),
    ("0300", "80053", 140.0, 1, "Lab"), ("0300", "85025", 95.0, 1, "Lab"),
    ("0251", "", 210.0, 1, "Drugs"), ("0636", "J1885", 55.0, 1, "Drugs"),
    ("0762", "", 450.0, 6, "Observation"), ("0258", "", 110.0, 1, "IV Therapy"),
    ("0259", "", 150.0, 1, "Other Pharmacy"), ("0274", "", 240.0, 1, "Prosthetic"),
    ("0510", "99281", 220.0, 1, "Clinic"), ("0510", "99282", 350.0, 1, "Clinic"),
    ("0710", "", 320.0, 1, "Recovery"), ("0278", "", 45.0, 1, "Medical Supplies"),
], columns=["revenue_code", "procedure_code", "base_allowed", "typical_units", "category"])

PHARMACY_PRODUCTS = pd.DataFrame([
    ("00093-7424", "METFORMIN_500MG", 14.0, "generic"),
    ("00093-1047", "LISINOPRIL_10MG", 9.0, "generic"),
    ("59762-3720", "ATORVASTATIN_20MG", 11.0, "generic"),
    ("00054-4727", "ALBUTEROL_HFA", 55.0, "brand"),
    ("00781-1506", "OZEMPIC_1MG", 935.0, "brand"),
    ("50458-0578", "ELIQUIS_5MG", 590.0, "brand"),
    ("00002-8215", "HUMALOG_KWIKPEN", 620.0, "brand"),
    ("0591-0461", "AMOXICILLIN_500MG", 12.0, "generic"),
], columns=["ndc11", "drug_name", "base_allowed", "drug_type"])

ADMIN_CLAIMS = pd.DataFrame([
    ("ADM_FEE", 6.0, "Administrative"),
    ("ADM_APPEAL", 12.0, "Administrative"),
    ("ADM_RECOVERY", -18.0, "Administrative"),
    ("ADM_INTEREST", 3.0, "Administrative"),
], columns=["procedure_code", "base_allowed", "category"])

SPECIALTY_TO_CODES = {
    "Primary Care": np.array(["99212", "99213", "99214", "80048", "80053", "83036", "85025", "84443", "36415", "G0439", "90471", "90686", "81002"], dtype=object),
    "Internal Medicine": np.array(["99213", "99214", "99215", "80053", "83036", "85025", "93000"], dtype=object),
    "Behavioral Health": np.array(["90834", "90837", "99213", "99214"], dtype=object),
    "Cardiology": np.array(["93000", "99214", "99215", "71046"], dtype=object),
    "Therapy": np.array(["97110", "97112", "97140", "99213"], dtype=object),
    "Podiatry": np.array(["11721", "99213", "99214"], dtype=object),
    "Urgent Care": np.array(["99213", "99214", "71046", "80053", "85025", "12002"], dtype=object),
    "Orthopedics": np.array(["20610", "20550", "99213", "99214", "71046"], dtype=object),
}

DENIAL_REASONS = np.array([
    "CO-16_MISSING_INFO", "CO-50_NOT_MEDICALLY_NECESSARY", "CO-97_INCLUDED_IN_PAYMENT",
    "CO-151_NO_AUTH", "PR-204_NOT_COVERED", "CO-29_TIMELY_FILING", "N/A"
], dtype=object)

CONTRACT_RATES = pd.DataFrame([
    ("COMM_A", 1.00), ("COMM_B", 0.92), ("COMM_C", 0.85), ("MEDICARE_LIKE", 0.78), ("OON", 0.55)
], columns=["contract_rate_id", "rate_factor"])

RULE_PRIORITY = [
    ("rule_duplicate", "duplicate_signature"),
    ("rule_excessive_units", "excessive_units"),
    ("rule_ncci_ptp_like", "coding_combo_edit"),
    ("rule_mue_like", "mue_like_units"),
    ("rule_egregious_charge", "egregious_billed_charge"),
    ("rule_high_pharmacy_cost", "high_pharmacy_cost"),
    ("rule_member_submission_outlier", "member_submission_outlier"),
    ("rule_pricing_error", "pricing_error_variance"),
    ("rule_denied_high_charge", "denied_high_charge"),
    ("rule_high_amount_ratio", "high_code_cost_ratio"),
]

CATEGORICAL_COLS = [
    "claim_family", "encounter_type", "specialty", "provider_type", "provider_class", "place_of_service",
    "diagnosis_code", "revenue_code", "procedure_code", "category", "member_gender", "fraud_type",
    "contract_rate_id", "claim_status", "denial_reason", "ndc11", "drug_name", "source_system",
    "rule_primary_reason", "price_method", "zelis_edit_type"
]


# -----------------------------
# Helpers
# -----------------------------
def save_frame(df: pd.DataFrame, path_without_ext: Path) -> str:
    """Write parquet when available, otherwise pickle."""
    try:
        df.to_parquet(path_without_ext.with_suffix(".parquet"), index=False)
        return ".parquet"
    except Exception:
        df.to_pickle(path_without_ext.with_suffix(".pkl"))
        return ".pkl"


def make_members(n_members: int, rng: np.random.Generator) -> pd.DataFrame:
    ages = rng.integers(0, 91, size=n_members, dtype=np.int16)
    genders = rng.choice(np.array(["F", "M"], dtype=object), size=n_members, p=[0.53, 0.47])
    chronic_score = rng.poisson(1.35, size=n_members).astype(np.int8)
    return pd.DataFrame({
        "member_id": np.arange(1, n_members + 1, dtype=np.int32),
        "member_age": ages,
        "member_gender": genders,
        "chronic_score": chronic_score,
        "high_risk_member": ((ages >= 70) | (chronic_score >= 4)).astype(np.int8),
    })


def make_providers(n_providers: int, rng: np.random.Generator) -> pd.DataFrame:
    specialties = np.array([
        "Primary Care", "Internal Medicine", "Behavioral Health", "Cardiology", "Therapy",
        "Podiatry", "Urgent Care", "Hospital", "Orthopedics", "DME"
    ], dtype=object)
    probs = np.array([0.19, 0.16, 0.08, 0.08, 0.10, 0.05, 0.10, 0.16, 0.05, 0.03])
    specialty = rng.choice(specialties, size=n_providers, p=probs)
    provider_type = np.where(np.isin(specialty, ["Hospital", "DME"]), "facility", "professional")

    provider_class = rng.choice(np.array(["clean", "low", "medium", "high"], dtype=object), size=n_providers, p=[0.83, 0.10, 0.05, 0.02])
    fraud_propensity = np.select(
        [provider_class == "clean", provider_class == "low", provider_class == "medium", provider_class == "high"],
        [0.0, rng.uniform(0.001, 0.01, n_providers), rng.uniform(0.015, 0.05, n_providers), rng.uniform(0.06, 0.16, n_providers)],
        default=0.0,
    ).astype(np.float32)

    base_volume = np.select(
        [specialty == "Hospital", specialty == "Urgent Care", specialty == "DME"],
        [rng.integers(3000, 15000, size=n_providers), rng.integers(1200, 5000, size=n_providers), rng.integers(400, 1800, size=n_providers)],
        default=rng.integers(250, 2200, size=n_providers),
    ).astype(np.int32)

    contract_ids = rng.choice(CONTRACT_RATES["contract_rate_id"].to_numpy(dtype=object), size=n_providers, p=[0.23, 0.23, 0.18, 0.26, 0.10])

    return pd.DataFrame({
        "provider_id": np.arange(1, n_providers + 1, dtype=np.int32),
        "specialty": specialty,
        "provider_type": provider_type,
        "provider_class": provider_class,
        "fraud_propensity": fraud_propensity,
        "base_volume": base_volume,
        "contract_rate_id": contract_ids,
    })


def weighted_provider_draw(providers: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
    weights = providers["base_volume"].to_numpy(dtype=np.float64)
    weights = weights / weights.sum()
    return rng.choice(providers["provider_id"].to_numpy(dtype=np.int32), size=n, p=weights)


def generate_encounters(start_id: int, n_encounters: int, members: pd.DataFrame, providers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    provider_ids = weighted_provider_draw(providers, n_encounters, rng)
    p = providers.set_index("provider_id").loc[provider_ids].reset_index()
    member_ids = rng.choice(members["member_id"].to_numpy(dtype=np.int32), size=n_encounters)
    m = members.set_index("member_id").loc[member_ids].reset_index()

    encounter_type = rng.choice(
        np.array(["professional", "institutional", "pharmacy", "member_submitted", "administrative"], dtype=object),
        size=n_encounters,
        p=[0.58, 0.18, 0.16, 0.05, 0.03],
    )

    # Hospitals skew heavily toward institutional/administrative encounters.
    hosp_mask = p["specialty"].to_numpy(dtype=object) == "Hospital"
    encounter_type[hosp_mask] = rng.choice(np.array(["institutional", "administrative", "pharmacy"], dtype=object), size=hosp_mask.sum(), p=[0.78, 0.16, 0.06])

    # Member submitted claims should mostly be out-of-network style.
    member_sub_mask = encounter_type == "member_submitted"
    p.loc[member_sub_mask, "contract_rate_id"] = "OON"

    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n_encounters), unit="D")
    dx = rng.choice(DIAGNOSIS_CODES, size=n_encounters)

    pos = np.empty(n_encounters, dtype=object)
    pos[encounter_type == "professional"] = rng.choice(np.array(["11", "22", "02", "23"], dtype=object), size=(encounter_type == "professional").sum(), p=[0.60, 0.18, 0.10, 0.12])
    pos[encounter_type == "institutional"] = rng.choice(np.array(["21", "22", "23"], dtype=object), size=(encounter_type == "institutional").sum(), p=[0.30, 0.30, 0.40])
    pos[encounter_type == "pharmacy"] = "01"
    pos[encounter_type == "member_submitted"] = rng.choice(np.array(["11", "22", "99"], dtype=object), size=(encounter_type == "member_submitted").sum(), p=[0.35, 0.15, 0.50])
    pos[encounter_type == "administrative"] = "00"

    # Encounters expand to lines. Hospitals can be many lines.
    lines = np.ones(n_encounters, dtype=np.int16)
    inst = encounter_type == "institutional"
    if inst.any():
        inst_pos = pos[inst]
        line_counts = np.where(
            inst_pos == "21",
            rng.integers(8, 31, inst.sum()),
            np.where(inst_pos == "23", rng.integers(3, 13, inst.sum()), rng.integers(2, 9, inst.sum()))
        )
        lines[inst] = line_counts.astype(np.int16)
    pharm = encounter_type == "pharmacy"
    if pharm.any():
        lines[pharm] = 1
    admin = encounter_type == "administrative"
    if admin.any():
        lines[admin] = 1
    member_sub = encounter_type == "member_submitted"
    if member_sub.any():
        lines[member_sub] = rng.integers(1, 4, member_sub.sum()).astype(np.int16)

    encounter_ids = np.arange(start_id, start_id + n_encounters, dtype=np.int64)
    df = pd.DataFrame({
        "encounter_id": encounter_ids,
        "member_id": member_ids.astype(np.int32),
        "provider_id": provider_ids.astype(np.int32),
        "service_date": dates,
        "encounter_type": encounter_type,
        "place_of_service": pos,
        "diagnosis_code": dx,
        "member_age": m["member_age"].to_numpy(dtype=np.int16),
        "member_gender": m["member_gender"].to_numpy(dtype=object),
        "chronic_score": m["chronic_score"].to_numpy(dtype=np.int8),
        "high_risk_member": m["high_risk_member"].to_numpy(dtype=np.int8),
        "specialty": p["specialty"].to_numpy(dtype=object),
        "provider_type": p["provider_type"].to_numpy(dtype=object),
        "provider_class": p["provider_class"].to_numpy(dtype=object),
        "provider_fraud_propensity": p["fraud_propensity"].to_numpy(dtype=np.float32),
        "contract_rate_id": p["contract_rate_id"].to_numpy(dtype=object),
        "planned_lines": lines,
    })
    return df


def expand_to_lines(encounters: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    lines = encounters.loc[encounters.index.repeat(encounters["planned_lines"])].copy()
    lines["line_in_claim"] = lines.groupby("encounter_id", sort=False).cumcount().add(1).astype(np.int16)
    lines["claim_id"] = lines["encounter_id"].astype(np.int64)
    lines["line_id"] = np.arange(1, len(lines) + 1, dtype=np.int64)

    n = len(lines)
    etype = lines["encounter_type"].to_numpy(dtype=object)
    procedure = np.full(n, "", dtype=object)
    revenue = np.full(n, "", dtype=object)
    category = np.full(n, "", dtype=object)
    ndc11 = np.full(n, "", dtype=object)
    drug_name = np.full(n, "", dtype=object)
    source_system = np.full(n, "core_claims", dtype=object)
    units = np.ones(n, dtype=np.int16)
    billed = np.zeros(n, dtype=np.float32)
    contracted_allowed = np.zeros(n, dtype=np.float32)
    paid = np.zeros(n, dtype=np.float32)
    denied = np.zeros(n, dtype=np.int8)
    zelis_edit_flag = np.zeros(n, dtype=np.int8)
    zelis_edit_type = np.full(n, "none", dtype=object)
    price_method = np.full(n, "contract_rate", dtype=object)
    egregious_charge_flag = np.zeros(n, dtype=np.int8)
    pricing_error_flag = np.zeros(n, dtype=np.int8)

    # Professional and member-submitted lines.
    prof_mask = np.isin(etype, ["professional", "member_submitted"])
    if prof_mask.any():
        sp = lines.loc[prof_mask, "specialty"].to_numpy(dtype=object)
        proc = np.empty(prof_mask.sum(), dtype=object)
        for unique_sp in np.unique(sp):
            idx = np.where(sp == unique_sp)[0]
            choices = SPECIALTY_TO_CODES.get(unique_sp, SPECIALTY_TO_CODES["Primary Care"])
            proc[idx] = rng.choice(choices, size=len(idx))
        ref = PROFESSIONAL_PROCEDURES.set_index("procedure_code").loc[proc]
        procedure[prof_mask] = proc
        category[prof_mask] = ref["category"].to_numpy(dtype=object)
        units[prof_mask] = np.maximum(1, rng.poisson(ref["typical_units"].to_numpy())).astype(np.int16)
        base = ref["base_allowed"].to_numpy(dtype=np.float32)
        noise = rng.lognormal(0.0, 0.20, prof_mask.sum()).astype(np.float32)
        billed[prof_mask] = base * units[prof_mask] * noise * rng.uniform(1.10, 1.55, prof_mask.sum())
        source_system[prof_mask] = np.where(etype[prof_mask] == "member_submitted", "member_portal", "core_claims")
        price_method[prof_mask] = np.where(etype[prof_mask] == "member_submitted", "manual_reprice", "contract_rate")

    # Institutional lines.
    inst_mask = etype == "institutional"
    if inst_mask.any():
        sample = INSTITUTIONAL_LINES.sample(n=inst_mask.sum(), replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
        revenue[inst_mask] = sample["revenue_code"].to_numpy(dtype=object)
        procedure[inst_mask] = sample["procedure_code"].to_numpy(dtype=object)
        category[inst_mask] = sample["category"].to_numpy(dtype=object)
        units[inst_mask] = np.maximum(1, rng.poisson(sample["typical_units"].to_numpy())).astype(np.int16)
        base = sample["base_allowed"].to_numpy(dtype=np.float32)
        noise = rng.lognormal(0.05, 0.28, inst_mask.sum()).astype(np.float32)
        severity = 1.0 + lines.loc[inst_mask, "chronic_score"].to_numpy(dtype=np.float32) * 0.05 + (lines.loc[inst_mask, "place_of_service"].astype(str).to_numpy() == "21") * 0.30
        billed[inst_mask] = base * units[inst_mask] * noise * severity * rng.uniform(1.15, 1.75, inst_mask.sum())
        source_system[inst_mask] = "facility_837i"

    # Pharmacy lines.
    pharm_mask = etype == "pharmacy"
    if pharm_mask.any():
        sample = PHARMACY_PRODUCTS.sample(n=pharm_mask.sum(), replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
        ndc11[pharm_mask] = sample["ndc11"].to_numpy(dtype=object)
        drug_name[pharm_mask] = sample["drug_name"].to_numpy(dtype=object)
        procedure[pharm_mask] = "RX_B1"
        category[pharm_mask] = "Pharmacy"
        days_supply = rng.choice(np.array([30, 30, 30, 90], dtype=np.int16), size=pharm_mask.sum())
        units[pharm_mask] = days_supply
        base = sample["base_allowed"].to_numpy(dtype=np.float32)
        billed[pharm_mask] = base * np.where(days_supply == 90, rng.uniform(2.2, 2.8, pharm_mask.sum()), 1.0) * rng.uniform(1.0, 1.12, pharm_mask.sum())
        source_system[pharm_mask] = "pharmacy_ncpdp"
        price_method[pharm_mask] = "ingredient_plus_fee"

    # Administrative lines.
    admin_mask = etype == "administrative"
    if admin_mask.any():
        sample = ADMIN_CLAIMS.sample(n=admin_mask.sum(), replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
        procedure[admin_mask] = sample["procedure_code"].to_numpy(dtype=object)
        category[admin_mask] = sample["category"].to_numpy(dtype=object)
        billed[admin_mask] = sample["base_allowed"].to_numpy(dtype=np.float32)
        source_system[admin_mask] = "admin_platform"
        price_method[admin_mask] = "administrative"

    # Contracted allowed and adjudication.
    rate_map = CONTRACT_RATES.set_index("contract_rate_id")["rate_factor"]
    rate = lines["contract_rate_id"].map(rate_map).to_numpy(dtype=np.float32)
    rate = np.where(etype == "member_submitted", rate * rng.uniform(0.85, 1.00, n), rate)
    contracted_allowed = np.where(billed >= 0, billed * rate, billed).astype(np.float32)

    claim_status = np.full(n, "paid", dtype=object)
    denial_reason = np.full(n, "N/A", dtype=object)

    # Base denials.
    deny_roll = rng.random(n)
    deny_prob = np.select(
        [etype == "member_submitted", etype == "administrative", etype == "pharmacy"],
        [0.11, 0.03, 0.06],
        default=0.08,
    )
    denied_mask = (deny_roll < deny_prob) & (billed > 0)
    denied[denied_mask] = 1
    claim_status[denied_mask] = "denied"
    denial_reason[denied_mask] = rng.choice(DENIAL_REASONS[:-1], size=denied_mask.sum())
    paid = np.where(denied_mask, 0.0, contracted_allowed * rng.uniform(0.80, 1.00, n)).astype(np.float32)

    # Simulated external coding edits, named after the workflow the user cares about.
    # This is a simplified vendor-style edit flag, not a proprietary implementation.
    ptp_like = (
        (lines["encounter_type"].astype(str).to_numpy() == "professional") &
        np.isin(procedure, ["36415", "80048", "80053", "85025"]) &
        (lines["line_in_claim"].to_numpy() > 1)
    )
    mue_like = (
        ((etype == "professional") & (units >= 8)) |
        ((etype == "institutional") & (units >= 20)) |
        ((etype == "pharmacy") & (units >= 90))
    )
    zelis_edit_flag[ptp_like | mue_like] = 1
    zelis_edit_type[ptp_like] = "NCCI_PTP_LIKE"
    zelis_edit_type[mue_like] = np.where(zelis_edit_type[mue_like] == "none", "MUE_LIKE", zelis_edit_type[mue_like])

    # Pricing and billing anomalies.
    pricing_error_roll = rng.random(n) < 0.004
    pricing_error_flag[pricing_error_roll] = 1
    price_method[pricing_error_roll] = "pricing_error"
    contracted_allowed[pricing_error_roll] *= rng.uniform(1.35, 2.50, pricing_error_roll.sum())
    paid[pricing_error_roll] = np.where(denied[pricing_error_roll] == 1, 0.0, contracted_allowed[pricing_error_roll] * rng.uniform(0.85, 1.00, pricing_error_roll.sum()))

    egregious_roll = rng.random(n) < 0.003
    egregious_charge_flag[egregious_roll] = 1
    billed[egregious_roll] *= rng.uniform(3.0, 10.0, egregious_roll.sum())

    df = lines[[
        "encounter_id", "claim_id", "line_id", "member_id", "provider_id", "service_date", "encounter_type",
        "place_of_service", "diagnosis_code", "member_age", "member_gender", "chronic_score", "high_risk_member",
        "specialty", "provider_type", "provider_class", "provider_fraud_propensity", "contract_rate_id", "line_in_claim"
    ]].copy()
    df["claim_family"] = np.where(df["encounter_type"].isin(["institutional", "administrative"]), df["encounter_type"], np.where(df["encounter_type"] == "pharmacy", "pharmacy", "professional"))
    df["revenue_code"] = revenue
    df["procedure_code"] = procedure
    df["category"] = category
    df["ndc11"] = ndc11
    df["drug_name"] = drug_name
    df["units"] = units.astype(np.int16)
    df["billed_amount"] = np.round(billed, 2)
    df["contracted_allowed"] = np.round(contracted_allowed, 2)
    df["allowed_amount"] = np.round(contracted_allowed, 2)
    df["paid_amount"] = np.round(paid, 2)
    df["claim_status"] = claim_status
    df["denial_reason"] = denial_reason
    df["denied_flag"] = denied
    df["zelis_edit_flag"] = zelis_edit_flag
    df["zelis_edit_type"] = zelis_edit_type
    df["egregious_charge_flag"] = egregious_charge_flag
    df["pricing_error_flag"] = pricing_error_flag
    df["price_method"] = price_method
    df["telehealth_flag"] = (df["place_of_service"].astype(str) == "02").astype(np.int8)
    df["inpatient_flag"] = (df["place_of_service"].astype(str) == "21").astype(np.int8)
    df["emergency_flag"] = (df["place_of_service"].astype(str) == "23").astype(np.int8)
    df["source_system"] = source_system
    return df


def inject_provider_level_fraud(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df = df.copy()
    df["fraud_label"] = 0
    df["fraud_type"] = "none"

    fraud_roll = rng.random(n)
    fraudulent = fraud_roll < df["provider_fraud_propensity"].to_numpy(dtype=np.float32)

    if fraudulent.any():
        choices = np.array([
            "duplicate", "upcoding", "excessive_units", "unbundling", "high_cost_facility",
            "pharmacy_refill_abuse", "member_reimbursement_abuse", "pricing_error_abuse", "egregious_charge"
        ], dtype=object)
        probs = np.array([0.18, 0.16, 0.15, 0.11, 0.12, 0.10, 0.06, 0.05, 0.07])
        idx = np.where(fraudulent)[0]
        modes = rng.choice(choices, size=len(idx), p=probs)
        df.iloc[idx, df.columns.get_loc("fraud_label")] = 1
        df.iloc[idx, df.columns.get_loc("fraud_type")] = modes

        dup_idx = idx[modes == "duplicate"]
        if len(dup_idx):
            seed = df.iloc[dup_idx].sample(frac=0.55, random_state=int(rng.integers(0, 1_000_000_000))).copy()
            if len(seed):
                seed["line_id"] = np.arange(int(df["line_id"].max()) + 1, int(df["line_id"].max()) + 1 + len(seed), dtype=np.int64)
                seed["fraud_label"] = 1
                seed["fraud_type"] = "duplicate"
                df = pd.concat([df, seed], ignore_index=True)

        up_idx = idx[modes == "upcoding"]
        prof_up = up_idx[df.iloc[up_idx]["claim_family"].astype(str).to_numpy() == "professional"]
        if len(prof_up):
            df.loc[df.index[prof_up], "procedure_code"] = "99215"
            df.loc[df.index[prof_up], "category"] = "E&M"
            df.loc[df.index[prof_up], "billed_amount"] *= rng.uniform(1.35, 1.90, len(prof_up))
            df.loc[df.index[prof_up], "allowed_amount"] *= rng.uniform(1.20, 1.60, len(prof_up))
            df.loc[df.index[prof_up], "paid_amount"] = np.where(df.loc[df.index[prof_up], "denied_flag"] == 1, 0.0, df.loc[df.index[prof_up], "allowed_amount"] * rng.uniform(0.85, 1.00, len(prof_up)))

        ex_idx = idx[modes == "excessive_units"]
        if len(ex_idx):
            fam = df.iloc[ex_idx]["claim_family"].astype(str).to_numpy()
            new_units = np.where(fam == "professional", rng.integers(8, 20, len(ex_idx)), np.where(fam == "pharmacy", rng.integers(90, 181, len(ex_idx)), rng.integers(18, 40, len(ex_idx)))).astype(np.int16)
            df.loc[df.index[ex_idx], "units"] = new_units
            df.loc[df.index[ex_idx], "billed_amount"] = (df.loc[df.index[ex_idx], "billed_amount"].to_numpy(dtype=np.float32) * rng.uniform(1.4, 2.8, len(ex_idx))).astype(np.float32)
            df.loc[df.index[ex_idx], "allowed_amount"] = (df.loc[df.index[ex_idx], "allowed_amount"].to_numpy(dtype=np.float32) * rng.uniform(1.2, 2.2, len(ex_idx))).astype(np.float32)
            df.loc[df.index[ex_idx], "paid_amount"] = np.where(df.loc[df.index[ex_idx], "denied_flag"] == 1, 0.0, df.loc[df.index[ex_idx], "allowed_amount"].to_numpy(dtype=np.float32) * rng.uniform(0.80, 1.00, len(ex_idx))).astype(np.float32)

        unb_idx = idx[modes == "unbundling"]
        if len(unb_idx):
            prof_unb = unb_idx[df.iloc[unb_idx]["claim_family"].astype(str).to_numpy() == "professional"]
            if len(prof_unb):
                df.loc[df.index[prof_unb], "procedure_code"] = rng.choice(np.array(["36415", "80048", "85025"], dtype=object), size=len(prof_unb))
                df.loc[df.index[prof_unb], "category"] = "Lab"
                df.loc[df.index[prof_unb], "zelis_edit_flag"] = 1
                df.loc[df.index[prof_unb], "zelis_edit_type"] = "NCCI_PTP_LIKE"

        fac_idx = idx[modes == "high_cost_facility"]
        inst_fac = fac_idx[df.iloc[fac_idx]["claim_family"].astype(str).to_numpy() == "institutional"]
        if len(inst_fac):
            df.loc[df.index[inst_fac], "billed_amount"] = (df.loc[df.index[inst_fac], "billed_amount"].to_numpy(dtype=np.float32) * rng.uniform(1.8, 3.5, len(inst_fac))).astype(np.float32)
            df.loc[df.index[inst_fac], "allowed_amount"] = (df.loc[df.index[inst_fac], "allowed_amount"].to_numpy(dtype=np.float32) * rng.uniform(1.4, 2.6, len(inst_fac))).astype(np.float32)
            df.loc[df.index[inst_fac], "paid_amount"] = np.where(df.loc[df.index[inst_fac], "denied_flag"] == 1, 0.0, df.loc[df.index[inst_fac], "allowed_amount"].to_numpy(dtype=np.float32) * rng.uniform(0.82, 1.00, len(inst_fac))).astype(np.float32)

        pharm_idx = idx[modes == "pharmacy_refill_abuse"]
        pharm_real = pharm_idx[df.iloc[pharm_idx]["claim_family"].astype(str).to_numpy() == "pharmacy"]
        if len(pharm_real):
            df.loc[df.index[pharm_real], "units"] = rng.choice(np.array([90, 120, 180], dtype=np.int16), size=len(pharm_real))
            df.loc[df.index[pharm_real], "billed_amount"] *= rng.uniform(1.4, 2.5, len(pharm_real))
            df.loc[df.index[pharm_real], "allowed_amount"] *= rng.uniform(1.2, 2.0, len(pharm_real))
            df.loc[df.index[pharm_real], "paid_amount"] = np.where(df.loc[df.index[pharm_real], "denied_flag"] == 1, 0.0, df.loc[df.index[pharm_real], "allowed_amount"] * rng.uniform(0.9, 1.0, len(pharm_real)))

        member_idx = idx[modes == "member_reimbursement_abuse"]
        member_real = member_idx[df.iloc[member_idx]["encounter_type"].astype(str).to_numpy() == "member_submitted"]
        if len(member_real):
            df.loc[df.index[member_real], "billed_amount"] *= rng.uniform(1.6, 4.0, len(member_real))
            df.loc[df.index[member_real], "allowed_amount"] *= rng.uniform(1.1, 2.2, len(member_real))
            df.loc[df.index[member_real], "paid_amount"] = np.where(df.loc[df.index[member_real], "denied_flag"] == 1, 0.0, df.loc[df.index[member_real], "allowed_amount"] * rng.uniform(0.65, 0.95, len(member_real)))

        price_idx = idx[modes == "pricing_error_abuse"]
        if len(price_idx):
            df.loc[df.index[price_idx], "pricing_error_flag"] = 1
            df.loc[df.index[price_idx], "price_method"] = "pricing_error"
            df.loc[df.index[price_idx], "allowed_amount"] *= rng.uniform(1.5, 3.0, len(price_idx))
            df.loc[df.index[price_idx], "paid_amount"] = np.where(df.loc[df.index[price_idx], "denied_flag"] == 1, 0.0, df.loc[df.index[price_idx], "allowed_amount"] * rng.uniform(0.85, 1.0, len(price_idx)))

        eg_idx = idx[modes == "egregious_charge"]
        if len(eg_idx):
            df.loc[df.index[eg_idx], "egregious_charge_flag"] = 1
            df.loc[df.index[eg_idx], "billed_amount"] *= rng.uniform(3.0, 9.0, len(eg_idx))

    df["fraud_label"] = df["fraud_label"].astype(np.int8)
    return df


def add_detection_features(df: pd.DataFrame) -> pd.DataFrame:
    dup_cols = ["member_id", "provider_id", "service_date", "diagnosis_code", "procedure_code", "revenue_code", "ndc11", "allowed_amount"]
    df["duplicate_signature_count"] = df.groupby(dup_cols, observed=True)["line_id"].transform("size").astype(np.int16)
    df["provider_day_claim_count"] = df.groupby(["provider_id", "service_date"], observed=True)["line_id"].transform("size").astype(np.int16)
    df["provider_avg_allowed"] = df.groupby("provider_id", observed=True)["allowed_amount"].transform("mean").astype(np.float32)
    code_key = np.where(df["claim_family"].astype(str).to_numpy() == "pharmacy", df["ndc11"].astype(str).to_numpy(), np.where(df["claim_family"].astype(str).to_numpy() == "institutional", df["revenue_code"].astype(str).to_numpy(), df["procedure_code"].astype(str).to_numpy()))
    df["code_key"] = pd.Categorical(code_key)
    df["code_avg_allowed"] = df.groupby("code_key", observed=True)["allowed_amount"].transform("mean").astype(np.float32)
    df["allowed_to_code_avg_ratio"] = (df["allowed_amount"] / np.maximum(df["code_avg_allowed"], 1)).astype(np.float32)
    df["billed_to_allowed_ratio"] = (np.where(np.abs(df["allowed_amount"]) > 0, df["billed_amount"] / np.maximum(df["allowed_amount"], 1), 0)).astype(np.float32)
    return df


def apply_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    fam = df["claim_family"].astype(str).to_numpy()
    etype = df["encounter_type"].astype(str).to_numpy()
    revenue = df["revenue_code"].astype(str).to_numpy()
    ndc = df["ndc11"].astype(str).to_numpy()

    df["rule_duplicate"] = (df["duplicate_signature_count"] > 1).astype(np.int8)
    df["rule_excessive_units"] = (((fam == "professional") & (df["units"] >= 8)) | ((fam == "institutional") & (df["units"] >= 20)) | ((fam == "pharmacy") & (df["units"] >= 90))).astype(np.int8)
    df["rule_high_amount_ratio"] = (df["allowed_to_code_avg_ratio"] >= 2.2).astype(np.int8)
    df["rule_egregious_charge"] = ((df["billed_amount"] > np.maximum(df["allowed_amount"], 1) * 4.5) | (df["egregious_charge_flag"] == 1)).astype(np.int8)
    df["rule_pricing_error"] = ((df["pricing_error_flag"] == 1) | ((df["price_method"].astype(str) == "pricing_error") & (df["paid_amount"] > df["allowed_amount"] * 0.95))).astype(np.int8)
    df["rule_denied_high_charge"] = ((df["denied_flag"] == 1) & (df["billed_amount"] >= df["billed_amount"].quantile(0.98))).astype(np.int8)
    df["rule_member_submission_outlier"] = ((etype == "member_submitted") & ((df["billed_amount"] > df["allowed_amount"] * 2.5) | (df["claim_status"].astype(str) == "denied"))).astype(np.int8)
    df["rule_high_pharmacy_cost"] = ((fam == "pharmacy") & ((df["allowed_to_code_avg_ratio"] >= 1.8) | (df["units"] >= 90) | (ndc == "00781-1506"))).astype(np.int8)
    df["rule_ncci_ptp_like"] = (((df["zelis_edit_type"].astype(str) == "NCCI_PTP_LIKE") | ((df["procedure_code"].astype(str).isin(["36415", "80048", "80053", "85025"])) & (df["line_in_claim"] > 1)))).astype(np.int8)
    df["rule_mue_like"] = (((df["zelis_edit_type"].astype(str) == "MUE_LIKE") | ((fam == "institutional") & (df["units"] >= 20)) | ((fam == "professional") & (df["units"] >= 8)))).astype(np.int8)

    rule_cols = [c for c in df.columns if c.startswith("rule_") and c not in {"rule_primary_reason", "rule_score", "rule_pred"}]
    df["rule_score"] = df[rule_cols].sum(axis=1).astype(np.int16)
    df["rule_pred"] = (df["rule_score"] >= 2).astype(np.int8)

    primary = np.full(len(df), "no_rule", dtype=object)
    for col, reason in RULE_PRIORITY:
        cond = df[col].to_numpy() == 1
        primary[(primary == "no_rule") & cond] = reason
    df["rule_primary_reason"] = primary
    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def update_provider_agg(store: dict[int, dict], chunk: pd.DataFrame) -> None:
    grp = chunk.groupby(["provider_id", "specialty", "provider_class"], observed=True).agg(
        total_lines=("line_id", "size"),
        fraud_lines=("fraud_label", "sum"),
        flagged_lines=("rule_pred", "sum"),
        allowed_sum=("allowed_amount", "sum"),
        max_day_volume=("provider_day_claim_count", "max"),
        avg_rule_score_sum=("rule_score", "sum"),
    ).reset_index()
    for row in grp.itertuples(index=False):
        key = int(row.provider_id)
        current = store.get(key)
        if current is None:
            store[key] = {
                "provider_id": key,
                "specialty": row.specialty,
                "provider_class": row.provider_class,
                "total_lines": int(row.total_lines),
                "fraud_lines": int(row.fraud_lines),
                "flagged_lines": int(row.flagged_lines),
                "allowed_sum": float(row.allowed_sum),
                "max_day_volume": int(row.max_day_volume),
                "avg_rule_score_sum": float(row.avg_rule_score_sum),
            }
        else:
            current["total_lines"] += int(row.total_lines)
            current["fraud_lines"] += int(row.fraud_lines)
            current["flagged_lines"] += int(row.flagged_lines)
            current["allowed_sum"] += float(row.allowed_sum)
            current["max_day_volume"] = max(current["max_day_volume"], int(row.max_day_volume))
            current["avg_rule_score_sum"] += float(row.avg_rule_score_sum)


def merge_count_series(store: pd.Series | None, chunk_series: pd.Series) -> pd.Series:
    if store is None:
        return chunk_series
    return store.add(chunk_series, fill_value=0)


def run_generation(
    n_encounters: int = 1_000_000,
    n_members: int = 120_000,
    n_providers: int = 6_000,
    chunk_size: int = 100_000,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving files to: {output_dir.resolve()}")

    rng = np.random.default_rng(SEED)
    members = make_members(n_members, rng)
    providers = make_providers(n_providers, rng)
    members.to_csv(output_dir / "members.csv", index=False)
    providers.to_csv(output_dir / "providers.csv", index=False)

    next_encounter_id = 1
    chunk_files = []
    sample_parts = []
    provider_store: dict[int, dict] = {}
    fraud_counts = None
    rule_reason_counts = None
    detect_counts = None
    total_rows = 0
    total_tp = total_fp = total_fn = total_tn = 0

    n_chunks = (n_encounters + chunk_size - 1) // chunk_size
    for chunk_id in range(n_chunks):
        n_this = min(chunk_size, n_encounters - chunk_id * chunk_size)
        chunk_rng = np.random.default_rng(SEED + chunk_id + 1)
        encounters = generate_encounters(next_encounter_id, n_this, members, providers, chunk_rng)
        next_encounter_id += n_this

        lines = expand_to_lines(encounters, chunk_rng)
        lines = inject_provider_level_fraud(lines, chunk_rng)
        lines = add_detection_features(lines)
        lines = apply_rule_flags(lines)
        lines = cast_categoricals(lines)

        ext = save_frame(lines, output_dir / f"claims_lines_chunk_{chunk_id:03d}")
        chunk_files.append(f"claims_lines_chunk_{chunk_id:03d}{ext}")
        sample_parts.append(lines.sample(min(20000, len(lines)), random_state=SEED + chunk_id))

        total_rows += len(lines)
        total_tp += int(((lines["fraud_label"] == 1) & (lines["rule_pred"] == 1)).sum())
        total_fp += int(((lines["fraud_label"] == 0) & (lines["rule_pred"] == 1)).sum())
        total_fn += int(((lines["fraud_label"] == 1) & (lines["rule_pred"] == 0)).sum())
        total_tn += int(((lines["fraud_label"] == 0) & (lines["rule_pred"] == 0)).sum())

        fraud_counts = merge_count_series(fraud_counts, lines["fraud_type"].astype(str).value_counts())
        rule_reason_counts = merge_count_series(rule_reason_counts, lines["rule_primary_reason"].astype(str).value_counts())
        detect_grp = lines.groupby("fraud_type", observed=True).agg(n=("line_id", "size"), caught=("rule_pred", "sum"), avg_rule_score=("rule_score", "mean"))
        if detect_counts is None:
            detect_counts = detect_grp.copy()
        else:
            detect_counts = detect_counts.add(detect_grp, fill_value=0)
        update_provider_agg(provider_store, lines)

        print(f"chunk {chunk_id + 1}/{n_chunks}: encounters={n_this:,}, lines={len(lines):,}")

    sample = pd.concat(sample_parts, ignore_index=True)
    if len(sample) > 100_000:
        sample = sample.sample(100_000, random_state=SEED)
    sample.to_csv(output_dir / "synthetic_claim_lines_sample_100k.csv", index=False)

    fraud_breakdown = fraud_counts.rename_axis("fraud_type").reset_index(name="count").sort_values("count", ascending=False)
    fraud_breakdown.to_csv(output_dir / "fraud_breakdown.csv", index=False)

    rule_reason_summary = rule_reason_counts.rename_axis("rule_primary_reason").reset_index(name="count").sort_values("count", ascending=False)
    rule_reason_summary.to_csv(output_dir / "rule_reason_summary.csv", index=False)

    provider_summary = pd.DataFrame(provider_store.values())
    provider_summary["observed_fraud_rate"] = provider_summary["fraud_lines"] / provider_summary["total_lines"]
    provider_summary["observed_flag_rate"] = provider_summary["flagged_lines"] / provider_summary["total_lines"]
    provider_summary["avg_allowed"] = provider_summary["allowed_sum"] / provider_summary["total_lines"]
    provider_summary["avg_rule_score"] = provider_summary["avg_rule_score_sum"] / provider_summary["total_lines"]
    provider_summary = provider_summary.sort_values(["observed_fraud_rate", "observed_flag_rate", "avg_rule_score"], ascending=False)
    provider_summary.to_csv(output_dir / "provider_summary.csv", index=False)

    detection_by_fraud_type = detect_counts.reset_index().rename(columns={"index": "fraud_type"})
    detection_by_fraud_type["rule_recall_within_type"] = detection_by_fraud_type["caught"] / detection_by_fraud_type["n"]
    detection_by_fraud_type.to_csv(output_dir / "detection_by_fraud_type.csv", index=False)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    metrics = pd.DataFrame([{
        "model": "Rule-based baseline",
        "tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn,
        "precision": precision, "recall": recall, "f1": f1,
        "fraud_rate": total_tp + total_fn,
        "flag_rate": (total_tp + total_fp) / max(total_rows, 1),
        "total_rows": total_rows,
        "total_encounters_requested": n_encounters,
    }])
    metrics["fraud_rate"] = (total_tp + total_fn) / max(total_rows, 1)
    metrics.to_csv(output_dir / "rule_metrics.csv", index=False)

    manifest = pd.DataFrame({"file_name": chunk_files})
    manifest.to_csv(output_dir / "chunk_manifest.csv", index=False)

    print(f"\nencounters_requested={n_encounters:,}")
    print(f"rows_generated={total_rows:,}")
    print(f"fraud_rate={metrics.loc[0, 'fraud_rate']:.4f}")
    print(f"flag_rate={metrics.loc[0, 'flag_rate']:.4f}")
    print("\nTop fraud types:")
    print(fraud_breakdown.head(10).to_string(index=False))
    print("\nTop rule reasons:")
    print(rule_reason_summary.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate large synthetic healthcare claim lines with institutional, pharmacy, admin, denials, edits, and pricing anomalies.")
    parser.add_argument("--n_encounters", type=int, default=1_000_000, help="Number of encounters/claims before line expansion.")
    parser.add_argument("--n_members", type=int, default=120_000)
    parser.add_argument("--n_providers", type=int, default=6_000)
    parser.add_argument("--chunk_size", type=int, default=100_000, help="Encounter chunk size.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generation(
        n_encounters=args.n_encounters,
        n_members=args.n_members,
        n_providers=args.n_providers,
        chunk_size=args.chunk_size,
        output_dir=Path(args.output_dir),
    )
