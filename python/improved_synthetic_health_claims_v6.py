from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------
# Reference tables
# -----------------------------
PROCEDURE_ROWS = [
    # Professional / outpatient CPT / HCPCS-style rows
    ("99212", "Office/outpatient E&M", "professional", "Primary Care", "11", 55.0, 1, None, None),
    ("99213", "Office/outpatient E&M", "professional", "Primary Care", "11", 90.0, 1, None, None),
    ("99214", "Office/outpatient E&M", "professional", "Primary Care", "11", 140.0, 1, None, None),
    ("99215", "Office/outpatient E&M", "professional", "Primary Care", "11", 205.0, 1, None, None),
    ("99221", "Initial hospital care", "professional", "Hospitalist", "21", 135.0, 1, None, None),
    ("99222", "Initial hospital care", "professional", "Hospitalist", "21", 195.0, 1, None, None),
    ("99223", "Initial hospital care", "professional", "Hospitalist", "21", 255.0, 1, None, None),
    ("99231", "Subsequent hospital care", "professional", "Hospitalist", "21", 75.0, 1, None, None),
    ("99232", "Subsequent hospital care", "professional", "Hospitalist", "21", 115.0, 1, None, None),
    ("99233", "Subsequent hospital care", "professional", "Hospitalist", "21", 165.0, 1, None, None),
    ("93000", "Electrocardiogram", "professional", "Cardiology", "22", 55.0, 1, None, None),
    ("71046", "Chest x-ray", "professional", "Radiology", "22", 70.0, 1, None, None),
    ("70450", "CT head", "professional", "Radiology", "22", 260.0, 1, None, None),
    ("74177", "CT abdomen/pelvis", "professional", "Radiology", "22", 430.0, 1, None, None),
    ("80053", "Comprehensive metabolic panel", "professional", "Laboratory", "22", 25.0, 1, None, None),
    ("85025", "CBC with differential", "professional", "Laboratory", "22", 20.0, 1, None, None),
    ("83036", "Hemoglobin A1c", "professional", "Laboratory", "11", 22.0, 1, None, None),
    ("36415", "Venipuncture", "professional", "Laboratory", "11", 6.0, 1, None, None),
    ("90834", "Psychotherapy 45 minutes", "professional", "Behavioral Health", "11", 120.0, 1, None, None),
    ("97110", "Therapeutic exercise", "professional", "Physical Therapy", "11", 38.0, 2, None, 4),
    ("97112", "Neuromuscular reeducation", "professional", "Physical Therapy", "11", 42.0, 2, None, 4),
    ("97530", "Therapeutic activities", "professional", "Occupational Therapy", "11", 46.0, 2, None, 4),
    ("92507", "Speech/hearing treatment", "professional", "Speech Therapy", "11", 58.0, 1, None, 2),
    ("88305", "Surgical pathology", "professional", "Pathology", "22", 85.0, 1, None, None),
    ("00840", "Anesthesia lower abdomen", "professional", "Anesthesiology", "21", 360.0, 1, None, None),
    ("47562", "Laparoscopic cholecystectomy", "professional", "Surgery", "21", 1450.0, 1, None, None),
    ("27447", "Total knee arthroplasty", "professional", "Surgery", "21", 1850.0, 1, None, None),
    ("29881", "Knee arthroscopy", "professional", "Surgery", "22", 1150.0, 1, None, None),
    ("G0439", "Annual wellness visit", "professional", "Primary Care", "11", 175.0, 1, None, None),
    # Institutional / revenue code anchored rows
    ("REV0450", "Emergency room", "institutional", "Hospital", "23", 850.0, 1, "0450", None),
    ("REV0360", "Operating room services", "institutional", "Hospital", "21", 4200.0, 1, "0360", None),
    ("REV0250", "Pharmacy general", "institutional", "Hospital", "21", 145.0, 2, "0250", None),
    ("REV0300", "Lab general", "institutional", "Hospital", "21", 95.0, 2, "0300", None),
    ("REV0320", "Radiology diagnostic", "institutional", "Hospital", "21", 340.0, 1, "0320", None),
    ("REV0420", "Physical therapy", "institutional", "Hospital", "21", 180.0, 1, "0420", None),
    ("REV0430", "Occupational therapy", "institutional", "Hospital", "21", 165.0, 1, "0430", None),
    ("REV0440", "Speech therapy", "institutional", "Hospital", "21", 170.0, 1, "0440", None),
    ("REV0278", "Medical/surgical supplies", "institutional", "Hospital", "21", 210.0, 2, "0278", None),
    ("REV0120", "Semi-private room and board", "institutional", "Hospital", "21", 1250.0, 2, "0120", None),
]

PROCEDURE_CATALOG = pd.DataFrame(
    PROCEDURE_ROWS,
    columns=[
        "code", "description", "claim_family", "specialty", "default_pos",
        "base_allowed", "typical_units", "revenue_code", "mue_limit"
    ]
)

DX_CODES = np.array([
    "I10", "E11.9", "J18.9", "N39.0", "R07.9", "R10.9", "M54.50", "M17.11",
    "K80.20", "F41.9", "S83.241A", "R51.9", "Z00.00", "Z79.4", "Z96.651",
    "M62.81", "R26.2", "R13.10", "G89.29", "J44.9", "I48.91", "E78.5"
], dtype=object)

NDC_ROWS = [
    ("00093-7424-01", "Metformin", "generic", 9.0, 30),
    ("00172-3928-60", "Lisinopril", "generic", 7.0, 30),
    ("00071-0155-23", "Ozempic", "brand", 935.0, 28),
    ("00006-0074-61", "Humalog", "brand", 325.0, 30),
    ("00597-0087-17", "Albuterol", "generic", 24.0, 1),
    ("54868-6266-00", "Hydrocodone/APAP", "controlled", 18.0, 20),
    ("00002-3227-01", "Eliquis", "brand", 585.0, 30),
    ("00456-1200-01", "Atorvastatin", "generic", 8.0, 30),
]
NDC_CATALOG = pd.DataFrame(NDC_ROWS, columns=["ndc", "drug_name", "drug_type", "base_allowed", "days_supply"])

DENIAL_REASONS = np.array([
    "timely_filing", "eligibility", "prior_auth", "noncovered", "duplicate", "coordination_of_benefits"
], dtype=object)

PAYMENT_TYPES = np.array(["check", "electronic", "checkless"], dtype=object)
PAYMENT_TYPE_PROBS = np.array([0.18, 0.72, 0.10])

# -----------------------------
# Helper builders
# -----------------------------
def build_members(n_members: int) -> pd.DataFrame:
    ages = rng.integers(0, 90, n_members, dtype=np.int16)
    chronic = rng.poisson(1.7, n_members).astype(np.int8)
    metal = rng.choice(["bronze", "silver", "gold", "self_funded"], size=n_members, p=[0.15, 0.32, 0.13, 0.40])
    deductible = np.select(
        [metal == "bronze", metal == "silver", metal == "gold", metal == "self_funded"],
        [7000, 4000, 1500, 2500],
        default=3000
    ).astype(np.float32)
    coinsurance = np.select(
        [metal == "bronze", metal == "silver", metal == "gold", metal == "self_funded"],
        [0.40, 0.25, 0.10, 0.20],
        default=0.20
    ).astype(np.float32)
    members = pd.DataFrame({
        "member_id": [f"M{ix:07d}" for ix in range(1, n_members + 1)],
        "member_age": ages,
        "member_gender": rng.choice(["F", "M"], size=n_members, p=[0.53, 0.47]),
        "chronic_score": chronic,
        "plan_type": metal,
        "annual_deductible": deductible,
        "coinsurance_rate": coinsurance,
    })
    members["deductible_remaining_start"] = np.round(members["annual_deductible"] * rng.uniform(0.15, 1.0, n_members), 2).astype(np.float32)
    return members


def build_providers(n_providers: int) -> pd.DataFrame:
    provider_types = np.array([
        "Primary Care", "Hospital", "Radiology", "Hospitalist", "Surgery", "Anesthesiology",
        "Pathology", "Physical Therapy", "Occupational Therapy", "Speech Therapy",
        "Behavioral Health", "Laboratory", "Cardiology", "Pharmacy", "Administration"
    ], dtype=object)
    probs = np.array([0.14, 0.10, 0.06, 0.08, 0.07, 0.04, 0.04, 0.06, 0.03, 0.02, 0.08, 0.06, 0.06, 0.08, 0.08])
    specialty = rng.choice(provider_types, size=n_providers, p=probs)
    provider_class = rng.choice(["clean", "low_risk", "medium_risk", "bad_actor"], size=n_providers, p=[0.84, 0.10, 0.045, 0.015])
    provider_fraud_propensity = np.select(
        [provider_class == "clean", provider_class == "low_risk", provider_class == "medium_risk", provider_class == "bad_actor"],
        [0.0005, 0.006, 0.02, 0.08],
        default=0.001
    ).astype(np.float32)
    contract_discount = np.select(
        [specialty == "Hospital", specialty == "Pharmacy", specialty == "Administration"],
        [0.38, 0.22, 0.10],
        default=0.28
    ).astype(np.float32)
    return pd.DataFrame({
        "provider_id": [f"P{ix:06d}" for ix in range(1, n_providers + 1)],
        "specialty": specialty,
        "provider_class": provider_class,
        "provider_fraud_propensity": provider_fraud_propensity,
        "contract_discount_rate": contract_discount,
    })


def weighted_pick(options: np.ndarray, probs: np.ndarray, size: int) -> np.ndarray:
    probs = probs / probs.sum()
    return rng.choice(options, size=size, p=probs)


def build_professional_lines(enc: pd.DataFrame) -> pd.DataFrame:
    specialty_to_codes = {
        "Primary Care": ["99212", "99213", "99214", "99215", "80053", "85025", "83036", "36415", "G0439"],
        "Behavioral Health": ["90834", "99213", "99214"],
        "Cardiology": ["93000", "99213", "99214"],
        "Physical Therapy": ["97110", "97112"],
        "Occupational Therapy": ["97530"],
        "Speech Therapy": ["92507"],
        "Radiology": ["71046", "70450", "74177"],
        "Hospitalist": ["99221", "99222", "99223", "99231", "99232", "99233"],
        "Surgery": ["47562", "27447", "29881"],
        "Anesthesiology": ["00840"],
        "Pathology": ["88305"],
        "Laboratory": ["80053", "85025", "36415"],
    }
    repeat_by_specialty = {
        "Physical Therapy": (1, 4),
        "Occupational Therapy": (1, 3),
        "Speech Therapy": (1, 2),
        "Primary Care": (1, 2),
        "Behavioral Health": (1, 1),
        "Cardiology": (1, 2),
        "Radiology": (1, 2),
        "Hospitalist": (1, 2),
        "Surgery": (1, 1),
        "Anesthesiology": (1, 1),
        "Pathology": (1, 1),
        "Laboratory": (1, 2),
    }
    parts: List[pd.DataFrame] = []
    for spec, grp in enc.groupby("provider_specialty", sort=False):
        codes = np.array(specialty_to_codes.get(spec, ["99213"]))
        lo, hi = repeat_by_specialty.get(spec, (1, 2))
        repeats = rng.integers(lo, hi + 1, len(grp))
        repeated = grp.loc[np.repeat(grp.index.to_numpy(), repeats)].copy()
        repeated["claim_line_no"] = repeated.groupby("encounter_id").cumcount().add(1).astype(np.int16)
        chosen_codes = rng.choice(codes, size=len(repeated))
        repeated["service_code"] = chosen_codes
        repeated = repeated.merge(PROCEDURE_CATALOG.add_prefix("proc_"), left_on="service_code", right_on="proc_code", how="left")
        repeated["revenue_code"] = repeated["proc_revenue_code"].fillna("")
        repeated["claim_family"] = repeated["proc_claim_family"]
        repeated["place_of_service"] = repeated["proc_default_pos"]
        units = np.maximum(1, rng.poisson(repeated["proc_typical_units"].fillna(1).to_numpy())).astype(np.int16)
        repeated["units"] = units
        noise = rng.lognormal(mean=0.0, sigma=0.20, size=len(repeated))
        severity = (1 + repeated["chronic_score"].to_numpy() * 0.035 + (repeated["place_of_service"].to_numpy() == "21") * 0.15)
        allowed = repeated["proc_base_allowed"].to_numpy() * repeated["units"].to_numpy() * noise * severity
        repeated["allowed_amount"] = np.round(allowed, 2).astype(np.float32)
        parts.append(repeated)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_institutional_lines(enc: pd.DataFrame) -> pd.DataFrame:
    inst = enc.copy()
    inst["claim_family"] = "institutional"
    inst["place_of_service"] = np.where(inst["encounter_type"] == "ed_facility", "23", "21")
    line_counts = np.where(inst["encounter_type"] == "ed_facility", rng.integers(2, 6, len(inst)), rng.integers(6, 18, len(inst)))
    inst = inst.loc[np.repeat(inst.index.to_numpy(), line_counts)].copy()
    inst["claim_line_no"] = inst.groupby("encounter_id").cumcount().add(1).astype(np.int16)
    # code mix by encounter type
    ed_codes = np.array(["REV0450", "REV0300", "REV0320", "REV0250"])
    ip_codes = np.array(["REV0120", "REV0300", "REV0320", "REV0360", "REV0250", "REV0278", "REV0420", "REV0430", "REV0440"])
    mask_ed = inst["encounter_type"].eq("ed_facility").to_numpy()
    service_codes = np.empty(len(inst), dtype=object)
    service_codes[mask_ed] = rng.choice(ed_codes, mask_ed.sum())
    service_codes[~mask_ed] = rng.choice(ip_codes, (~mask_ed).sum())
    inst["service_code"] = service_codes
    inst = inst.merge(PROCEDURE_CATALOG.add_prefix("proc_"), left_on="service_code", right_on="proc_code", how="left")
    inst["revenue_code"] = inst["proc_revenue_code"].fillna("")
    units = np.maximum(1, rng.poisson(inst["proc_typical_units"].fillna(1).to_numpy())).astype(np.int16)
    inst["units"] = units
    noise = rng.lognormal(mean=0.12, sigma=0.34, size=len(inst))
    severity = 1 + inst["chronic_score"].to_numpy() * 0.05 + (inst["encounter_type"].eq("inpatient_facility").to_numpy() * 0.35)
    allowed = inst["proc_base_allowed"].to_numpy() * inst["units"].to_numpy() * noise * severity
    inst["allowed_amount"] = np.round(allowed, 2).astype(np.float32)
    return inst


def build_pharmacy_lines(enc: pd.DataFrame) -> pd.DataFrame:
    pharm = enc.copy()
    pharm["claim_family"] = "pharmacy"
    pharm["claim_line_no"] = 1
    ndc = NDC_CATALOG.sample(n=len(pharm), replace=True, random_state=int(rng.integers(1, 1_000_000))).reset_index(drop=True)
    pharm = pd.concat([pharm.reset_index(drop=True), ndc], axis=1)
    fills = np.where(pharm["drug_type"].eq("controlled"), rng.integers(10, 31, len(pharm)), pharm["days_supply"].to_numpy())
    pharm["days_supply"] = fills.astype(np.int16)
    pharm["units"] = 1
    pharm["service_code"] = pharm["ndc"]
    pharm["revenue_code"] = ""
    pharm["place_of_service"] = "01"
    allowed = pharm["base_allowed"].to_numpy() * rng.lognormal(mean=0.0, sigma=0.10, size=len(pharm))
    pharm["allowed_amount"] = np.round(allowed, 2).astype(np.float32)
    return pharm


def build_member_admin_lines(enc: pd.DataFrame) -> pd.DataFrame:
    out = enc.copy()
    out["claim_line_no"] = 1
    out["units"] = 1
    out["revenue_code"] = ""
    out["days_supply"] = np.nan
    out["drug_name"] = None
    out["drug_type"] = None
    out["ndc"] = None
    out["base_allowed"] = np.where(out["encounter_type"].eq("admin"), rng.uniform(2, 25, len(out)), rng.uniform(35, 400, len(out)))
    out["service_code"] = np.where(out["encounter_type"].eq("admin"), "ADMIN001", "MSUB001")
    out["claim_family"] = np.where(out["encounter_type"].eq("admin"), "administrative", "member_submitted")
    out["place_of_service"] = np.where(out["encounter_type"].eq("admin"), "99", "11")
    out["allowed_amount"] = np.round(out["base_allowed"] * rng.lognormal(0.0, 0.25, len(out)), 2).astype(np.float32)
    return out


def apply_billing_and_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    discount = np.where(df["provider_specialty"].eq("Hospital"), 0.35, df["contract_discount_rate"].fillna(0.25))
    df["contract_rate"] = np.round(df["allowed_amount"] * (1 - discount), 2).astype(np.float32)
    bill_mult = np.where(df["claim_family"].eq("institutional"), rng.uniform(1.8, 4.5, len(df)), rng.uniform(1.15, 2.8, len(df)))
    df["billed_amount"] = np.round(np.maximum(df["allowed_amount"], df["contract_rate"]) * bill_mult, 2).astype(np.float32)
    payment_types = weighted_pick(PAYMENT_TYPES, PAYMENT_TYPE_PROBS, len(df))
    df["payment_type"] = payment_types
    denial_prob = np.select(
        [df["claim_family"].eq("member_submitted"), df["claim_family"].eq("administrative"), df["claim_family"].eq("pharmacy")],
        [0.10, 0.02, 0.06],
        default=0.04
    )
    denied = rng.random(len(df)) < denial_prob
    df["claim_status"] = np.where(denied, "denied", "paid")
    denial_reason = np.full(len(df), "", dtype=object)
    denial_reason[denied] = rng.choice(DENIAL_REASONS, denied.sum())
    df["denial_reason"] = denial_reason
    # deductible / coinsurance approximation using member starting point + ytd trend
    ytd_fraction = df["service_date"].dt.dayofyear.to_numpy() / 365.0
    est_remaining = np.maximum(0, df["deductible_remaining_start"].to_numpy() * (1 - ytd_fraction * rng.uniform(0.65, 1.05, len(df))))
    apply_ded = np.minimum(df["contract_rate"].to_numpy(), est_remaining)
    patient_coins = np.maximum(0, (df["contract_rate"].to_numpy() - apply_ded) * df["coinsurance_rate"].to_numpy())
    payer = np.maximum(0, df["contract_rate"].to_numpy() - apply_ded - patient_coins)
    payer[denied] = 0.0
    apply_ded[denied] = 0.0
    patient_coins[denied] = 0.0
    df["applied_to_deductible"] = np.round(apply_ded, 2).astype(np.float32)
    df["coinsurance_amount"] = np.round(patient_coins, 2).astype(np.float32)
    df["paid_amount"] = np.round(payer, 2).astype(np.float32)
    return df


def link_professional_to_facility(enc_base: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    facility = enc_base[enc_base["encounter_type"].isin(["inpatient_facility", "ed_facility"])].copy()
    if facility.empty:
        return pd.DataFrame(columns=enc_base.columns)
    prof_specs = ["Hospitalist", "Radiology", "Surgery", "Anesthesiology", "Pathology", "Physical Therapy", "Occupational Therapy", "Speech Therapy"]
    probs = np.array([1.0, 0.85, 0.30, 0.18, 0.20, 0.26, 0.12, 0.06])
    specialty_col = "specialty" if "specialty" in providers.columns else "provider_specialty"
    provider_map = {spec: providers.loc[providers[specialty_col].eq(spec), "provider_id"].to_numpy() for spec in prof_specs}
    rows = []
    next_id = 0
    for spec, prob in zip(prof_specs, probs):
        take = facility.loc[rng.random(len(facility)) < prob].copy()
        if take.empty or len(provider_map.get(spec, [])) == 0:
            continue
        take["provider_id"] = rng.choice(provider_map[spec], len(take))
        take["provider_specialty"] = spec
        take["encounter_type"] = f"linked_{spec.lower().replace(' ', '_')}"
        take["parent_encounter_id"] = take["encounter_id"]
        take["linked_facility_claim"] = 1
        take["encounter_id"] = [f"LE{next_id + i:09d}" for i in range(len(take))]
        next_id += len(take)
        rows.append(take)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=enc_base.columns)


def inject_fraud(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fraud_label"] = 0
    df["fraud_type"] = "none"
    df["rule_primary_reason"] = "none"
    df["pricing_error_signal"] = np.int8(0)
    df["egregious_charge_signal"] = np.int8(0)
    df["upcoding_signal"] = np.int8(0)
    df["member_submit_outlier_signal"] = np.int8(0)
    propensity = df["provider_fraud_propensity"].to_numpy().astype(float)
    fraud_draw = rng.random(len(df)) < propensity

    # ensure numeric columns can safely accept floating-point reassignments
    for col in ["contract_rate", "billed_amount", "allowed_amount", "coinsurance_rate", "deductible_remaining_start"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # specific patterns
    pricing_mask = fraud_draw & (rng.random(len(df)) < 0.20)
    df.loc[pricing_mask, "contract_rate"] = np.round(df.loc[pricing_mask, "contract_rate"] * rng.uniform(1.15, 1.85, pricing_mask.sum()), 2)
    df.loc[pricing_mask, ["fraud_label", "fraud_type", "rule_primary_reason"]] = [1, "pricing_error", "price_above_contract"]
    df.loc[pricing_mask, "pricing_error_signal"] = np.int8(1)

    egregious_mask = fraud_draw & (df["claim_family"].isin(["institutional", "professional"])) & (rng.random(len(df)) < 0.16)
    df.loc[egregious_mask, "billed_amount"] = np.round(df.loc[egregious_mask, "contract_rate"] * rng.uniform(6.0, 18.0, egregious_mask.sum()), 2)
    df.loc[egregious_mask, ["fraud_label", "fraud_type", "rule_primary_reason"]] = [1, "egregious_billing", "egregious_billed_charge"]
    df.loc[egregious_mask, "egregious_charge_signal"] = np.int8(1)

    excessive_mask = fraud_draw & df["service_code"].isin(PROCEDURE_CATALOG.loc[PROCEDURE_CATALOG["mue_limit"].notna(), "code"]) & (rng.random(len(df)) < 0.18)
    limits = df.loc[excessive_mask, "service_code"].map(PROCEDURE_CATALOG.set_index("code")["mue_limit"]).astype(float)
    df.loc[excessive_mask, "units"] = (limits.fillna(2) + rng.integers(2, 8, excessive_mask.sum())).astype(np.int16)
    df.loc[excessive_mask, ["fraud_label", "fraud_type", "rule_primary_reason"]] = [1, "excessive_units", "mue_like_units"]

    upcoding_mask = fraud_draw & df["service_code"].isin(["99212", "99213", "99214", "99221", "99222", "99231", "99232"]) & (rng.random(len(df)) < 0.16)
    upcode_map = {"99212": "99214", "99213": "99215", "99214": "99215", "99221": "99223", "99222": "99223", "99231": "99233", "99232": "99233"}
    new_codes = df.loc[upcoding_mask, "service_code"].map(upcode_map)
    df.loc[upcoding_mask, "service_code"] = new_codes.values
    remap = PROCEDURE_CATALOG.set_index("code")
    df.loc[upcoding_mask, "allowed_amount"] = np.round(remap.loc[new_codes.values, "base_allowed"].to_numpy() * rng.uniform(0.95, 1.20, upcoding_mask.sum()), 2)
    df.loc[upcoding_mask, ["fraud_label", "fraud_type", "rule_primary_reason"]] = [1, "upcoding", "level_shift"]
    df.loc[upcoding_mask, "upcoding_signal"] = np.int8(1)

    member_abuse_mask = fraud_draw & df["claim_family"].eq("member_submitted") & (rng.random(len(df)) < 0.20)
    df.loc[member_abuse_mask, "billed_amount"] = np.round(df.loc[member_abuse_mask, "allowed_amount"] * rng.uniform(3.0, 10.0, member_abuse_mask.sum()), 2)
    df.loc[member_abuse_mask, ["fraud_label", "fraud_type", "rule_primary_reason"]] = [1, "member_reimbursement_abuse", "outlier_member_submit"]
    df.loc[member_abuse_mask, "member_submit_outlier_signal"] = np.int8(1)

    # external edit flags (Zelis-like named honestly)
    df["ncci_like_edit_flag"] = ((df["service_code"].eq("36415") & df["encounter_id"].isin(df.loc[df["service_code"].isin(["80053", "85025"]), "encounter_id"])) ).astype(np.int8)
    mue_lookup = PROCEDURE_CATALOG.set_index("code")["mue_limit"].dropna().to_dict()
    df["mue_like_edit_flag"] = df.apply(lambda r: int(r["units"] > mue_lookup.get(r["service_code"], 9999)), axis=1).astype(np.int8)
    df["external_edit_flag"] = ((df["ncci_like_edit_flag"] == 1) | (df["mue_like_edit_flag"] == 1)).astype(np.int8)

    # duplicates by copying a small subset of already suspicious or high propensity lines
    dup_seed = df.loc[(df["provider_class"].isin(["medium_risk", "bad_actor"])) & (rng.random(len(df)) < 0.003)].copy()
    if not dup_seed.empty:
        dup_seed["claim_line_no"] = dup_seed["claim_line_no"] + 100
        dup_seed["fraud_label"] = 1
        dup_seed["fraud_type"] = "duplicate_billing"
        dup_seed["rule_primary_reason"] = "duplicate_signature"
        dup_seed["pricing_error_signal"] = np.int8(dup_seed["pricing_error_signal"])
        dup_seed["egregious_charge_signal"] = np.int8(dup_seed["egregious_charge_signal"])
        dup_seed["upcoding_signal"] = np.int8(dup_seed["upcoding_signal"])
        dup_seed["member_submit_outlier_signal"] = np.int8(dup_seed["member_submit_outlier_signal"])
        df = pd.concat([df, dup_seed], ignore_index=True)

    # recompute downstream amounts after fraud changes
    denied = df["claim_status"].eq("denied").to_numpy()
    ytd_fraction = df["service_date"].dt.dayofyear.to_numpy() / 365.0
    est_remaining = np.maximum(0, df["deductible_remaining_start"].to_numpy() * (1 - ytd_fraction * 0.9))
    apply_ded = np.minimum(df["contract_rate"].to_numpy(), est_remaining)
    coins = np.maximum(0, (df["contract_rate"].to_numpy() - apply_ded) * df["coinsurance_rate"].to_numpy())
    paid = np.maximum(0, df["contract_rate"].to_numpy() - apply_ded - coins)
    paid[denied] = 0.0
    apply_ded[denied] = 0.0
    coins[denied] = 0.0
    df["applied_to_deductible"] = np.round(apply_ded, 2).astype(np.float32)
    df["coinsurance_amount"] = np.round(coins, 2).astype(np.float32)
    df["paid_amount"] = np.round(paid, 2).astype(np.float32)
    return df


def build_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dup_counts = out.groupby([
        "member_id", "provider_id", "service_date", "service_code", "contract_rate"
    ]).size().rename("duplicate_signature_count")
    out = out.merge(dup_counts, on=["member_id", "provider_id", "service_date", "service_code", "contract_rate"], how="left")
    pday = out.groupby(["provider_id", "service_date"]).size().rename("provider_day_claim_count")
    out = out.merge(pday, on=["provider_id", "service_date"], how="left")
    provider_avg = out.groupby("provider_id")["contract_rate"].mean().rename("provider_avg_contract_rate")
    code_avg = out.groupby("service_code")["contract_rate"].mean().rename("service_code_avg_contract_rate")
    out = out.merge(provider_avg, on="provider_id", how="left")
    out = out.merge(code_avg, on="service_code", how="left")

    out["billed_to_contract_ratio"] = np.round(out["billed_amount"] / np.maximum(out["contract_rate"], 1.0), 4)
    out["contract_to_allowed_ratio"] = np.round(out["contract_rate"] / np.maximum(out["allowed_amount"], 1.0), 4)
    out["provider_to_code_ratio"] = np.round(out["provider_avg_contract_rate"] / np.maximum(out["service_code_avg_contract_rate"], 1.0), 4)

    # More sensitive rule thresholds so the review queue is realistically broader than the true fraud set.
    out["rule_duplicate"] = (out["duplicate_signature_count"] > 1).astype(np.int8)
    out["rule_provider_volume"] = (out["provider_day_claim_count"] >= 30).astype(np.int8)
    out["rule_egregious_billed"] = (out["billed_to_contract_ratio"] >= 6.0).astype(np.int8)
    out["rule_price_above_contract"] = (out["contract_to_allowed_ratio"] >= 1.15).astype(np.int8)
    out["rule_external_edit"] = pd.to_numeric(out["external_edit_flag"], errors="coerce").fillna(0).astype(np.int8)
    out["rule_denied_repeat"] = ((out["claim_status"] == "denied") & (out["provider_class"].isin(["medium_risk", "bad_actor"]))).astype(np.int8)
    out["rule_upcoding_pattern"] = (
        pd.to_numeric(out.get("upcoding_signal", 0), errors="coerce").fillna(0).astype(np.int8)
        | (((out["service_code"].isin(["99215", "99223", "99233"])) & (out["provider_to_code_ratio"] >= 1.20)).astype(np.int8))
    ).astype(np.int8)
    out["rule_member_submit_outlier"] = (
        pd.to_numeric(out.get("member_submit_outlier_signal", 0), errors="coerce").fillna(0).astype(np.int8)
        | ((out["claim_family"].eq("member_submitted")) & (out["billed_to_contract_ratio"] >= 2.5)).astype(np.int8)
    ).astype(np.int8)

    # Strong signals are allowed to trigger a review on their own.
    out["strong_rule_hit"] = np.maximum.reduce([
        out["rule_duplicate"].to_numpy(dtype=np.int8),
        pd.to_numeric(out.get("pricing_error_signal", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int8),
        pd.to_numeric(out.get("egregious_charge_signal", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int8),
        pd.to_numeric(out.get("upcoding_signal", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int8),
        pd.to_numeric(out.get("member_submit_outlier_signal", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int8),
    ]).astype(np.int8)

    rule_cols = [
        "rule_duplicate",
        "rule_provider_volume",
        "rule_egregious_billed",
        "rule_price_above_contract",
        "rule_external_edit",
        "rule_denied_repeat",
        "rule_upcoding_pattern",
        "rule_member_submit_outlier",
    ]
    for col in rule_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int8)

    out["rule_score"] = out[rule_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(np.int16)
    secondary_review = (
        ((out["rule_egregious_billed"] == 1) & (out["provider_class"].isin(["medium_risk", "bad_actor"])))
        | ((out["rule_external_edit"] == 1) & (out["claim_family"].eq("institutional")) & (~out["provider_class"].eq("clean")))
    )
    out["flagged_for_review"] = ((out["rule_score"] >= 2) | (out["strong_rule_hit"] >= 1) | secondary_review).astype(np.int8)

    # Assign a rule-driven primary reason for flagged rows.
    reason_pairs = [
        ("rule_duplicate", "duplicate_signature"),
        ("rule_egregious_billed", "egregious_billed_charge"),
        ("rule_price_above_contract", "price_above_contract"),
        ("rule_external_edit", "external_edit"),
        ("rule_upcoding_pattern", "upcoding_pattern"),
        ("rule_member_submit_outlier", "outlier_member_submit"),
        ("rule_provider_volume", "provider_volume_outlier"),
        ("rule_denied_repeat", "denied_repeat_pattern"),
    ]
    out["rule_primary_reason_detected"] = "no_rule"
    for col, reason in reason_pairs:
        mask = out["rule_primary_reason_detected"].eq("no_rule") & (out[col] == 1)
        out.loc[mask, "rule_primary_reason_detected"] = reason
    return out


def summarize(df: pd.DataFrame, output_dir: Path) -> None:
    fraud_breakdown = df.groupby(["fraud_type", "claim_family"]).size().reset_index(name="count").sort_values("count", ascending=False)
    provider_summary = df.groupby(["provider_id", "provider_specialty", "provider_class"]).agg(
        total_lines=("service_code", "size"),
        paid_lines=("claim_status", lambda s: int((s == "paid").sum())),
        denied_lines=("claim_status", lambda s: int((s == "denied").sum())),
        fraud_rate=("fraud_label", "mean"),
        avg_contract=("contract_rate", "mean"),
        avg_billed=("billed_amount", "mean"),
        avg_paid=("paid_amount", "mean"),
        flagged_rate=("flagged_for_review", "mean"),
    ).reset_index().sort_values(["fraud_rate", "flagged_rate", "avg_billed"], ascending=[False, False, False])

    family_summary = df.groupby(["claim_family", "claim_status", "payment_type"]).agg(
        lines=("service_code", "size"),
        billed_total=("billed_amount", "sum"),
        contract_total=("contract_rate", "sum"),
        paid_total=("paid_amount", "sum"),
        deductible_total=("applied_to_deductible", "sum"),
        coinsurance_total=("coinsurance_amount", "sum"),
    ).reset_index()

    claim_key = (
        df["encounter_id"].astype(str)
        + "|" + df["provider_id"].astype(str)
        + "|" + df["claim_family"].astype(str)
    )
    claim_level = df.assign(claim_key=claim_key).groupby("claim_key").agg(
        claim_fraud=("fraud_label", "max"),
        claim_flag=("flagged_for_review", "max")
    ).reset_index()
    encounter_level = df.groupby("encounter_id").agg(
        encounter_fraud=("fraud_label", "max"),
        encounter_flag=("flagged_for_review", "max")
    ).reset_index()

    line_fraud_rate = float(df["fraud_label"].mean())
    line_flag_rate = float(df["flagged_for_review"].mean())
    claim_fraud_rate = float(claim_level["claim_fraud"].mean())
    claim_flag_rate = float(claim_level["claim_flag"].mean())
    encounter_fraud_rate = float(encounter_level["encounter_fraud"].mean())
    encounter_flag_rate = float(encounter_level["encounter_flag"].mean())

    method_summary = pd.DataFrame({
        "metric": [
            "rows", "unique_members", "unique_providers", "unique_encounters",
            "line_fraud_rate", "line_flag_rate",
            "claim_fraud_rate", "claim_flag_rate",
            "encounter_fraud_rate", "encounter_flag_rate",
            "denial_rate", "pharmacy_share", "institutional_share", "linked_facility_professional_share"
        ],
        "value": [
            len(df), df["member_id"].nunique(), df["provider_id"].nunique(), df["encounter_id"].nunique(),
            round(line_fraud_rate, 6), round(line_flag_rate, 6),
            round(claim_fraud_rate, 6), round(claim_flag_rate, 6),
            round(encounter_fraud_rate, 6), round(encounter_flag_rate, 6),
            round((df["claim_status"] == "denied").mean(), 6), round((df["claim_family"] == "pharmacy").mean(), 6),
            round((df["claim_family"] == "institutional").mean(), 6), round(df["linked_facility_claim"].mean(), 6)
        ]
    })

    calibration_summary = pd.DataFrame({
        "level": ["line", "claim", "encounter"],
        "fraud_rate": [round(line_fraud_rate, 6), round(claim_fraud_rate, 6), round(encounter_fraud_rate, 6)],
        "flag_rate": [round(line_flag_rate, 6), round(claim_flag_rate, 6), round(encounter_flag_rate, 6)],
    })
    calibration_summary["flag_minus_fraud"] = (calibration_summary["flag_rate"] - calibration_summary["fraud_rate"]).round(6)
    calibration_summary["flag_rate_exceeds_fraud_rate"] = calibration_summary["flag_rate"] > calibration_summary["fraud_rate"]

    rule_reason = df.groupby(["rule_primary_reason_detected", "flagged_for_review"]).size().reset_index(name="count").sort_values("count", ascending=False)
    detection_by_fraud = df.groupby("fraud_type").agg(
        lines=("service_code", "size"),
        flagged_rate=("flagged_for_review", "mean"),
        avg_rule_score=("rule_score", "mean")
    ).reset_index().sort_values("lines", ascending=False)

    history_summary = df.groupby(["member_id", "provider_id"]).agg(
        first_service_date=("service_date", "min"),
        last_service_date=("service_date", "max"),
        line_count=("service_code", "size"),
        paid_total=("paid_amount", "sum")
    ).reset_index()
    history_summary["days_between_first_last"] = (history_summary["last_service_date"] - history_summary["first_service_date"]).dt.days
    claim_status_reason = df.groupby(["claim_status", "denial_reason"]).size().reset_index(name="count").sort_values("count", ascending=False)

    fraud_breakdown.to_csv(output_dir / "fraud_breakdown.csv", index=False)
    provider_summary.to_csv(output_dir / "provider_summary.csv", index=False)
    family_summary.to_csv(output_dir / "claim_family_financial_summary.csv", index=False)
    method_summary.to_csv(output_dir / "dataset_summary_metrics.csv", index=False)
    calibration_summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    rule_reason.to_csv(output_dir / "rule_reason_summary.csv", index=False)
    detection_by_fraud.to_csv(output_dir / "detection_by_fraud_type.csv", index=False)
    history_summary.to_csv(output_dir / "member_provider_history_summary.csv", index=False)
    claim_status_reason.to_csv(output_dir / "claim_status_reason_summary.csv", index=False)


def write_chunk(df: pd.DataFrame, output_dir: Path, ix: int) -> str:
    stem = output_dir / f"claims_lines_chunk_{ix:03d}"
    try:
        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path.name
    except Exception:
        path = stem.with_suffix(".pkl")
        df.to_pickle(path)
        return path.name


def generate_chunk(start_encounter: int, n_encounters: int, members: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    member_idx = rng.integers(0, len(members), n_encounters)
    provider_idx = rng.integers(0, len(providers), n_encounters)
    base = pd.DataFrame({
        "encounter_id": [f"E{start_encounter + i:09d}" for i in range(n_encounters)],
        "episode_id": [f"EP{start_encounter + i:09d}" for i in range(n_encounters)],
        "member_id": members.iloc[member_idx]["member_id"].to_numpy(),
        "provider_id": providers.iloc[provider_idx]["provider_id"].to_numpy(),
        "service_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n_encounters), unit="D"),
    })
    base = base.merge(members, on="member_id", how="left")
    base = base.merge(providers.rename(columns={"specialty": "provider_specialty"}), on="provider_id", how="left")
    base["diagnosis_code"] = rng.choice(DX_CODES, size=n_encounters)
    base["parent_encounter_id"] = ""
    base["linked_facility_claim"] = 0

    # encounter type determined from specialty mix
    specialty = base["provider_specialty"].to_numpy()
    encounter_type = np.full(n_encounters, "professional", dtype=object)
    encounter_type[specialty == "Hospital"] = weighted_pick(np.array(["inpatient_facility", "ed_facility"]), np.array([0.45, 0.55]), (specialty == "Hospital").sum())
    encounter_type[specialty == "Pharmacy"] = "pharmacy"
    encounter_type[specialty == "Administration"] = "admin"
    # a small slice of primary care / others become member submitted reimbursement claims
    msub_mask = (specialty == "Primary Care") & (rng.random(n_encounters) < 0.04)
    encounter_type[msub_mask] = "member_submitted"
    base["encounter_type"] = encounter_type

    prof = build_professional_lines(base.loc[base["encounter_type"].eq("professional")].copy())
    inst = build_institutional_lines(base.loc[base["encounter_type"].isin(["inpatient_facility", "ed_facility"])].copy())
    pharm = build_pharmacy_lines(base.loc[base["encounter_type"].eq("pharmacy")].copy())
    other = build_member_admin_lines(base.loc[base["encounter_type"].isin(["member_submitted", "admin"])].copy())

    linked = link_professional_to_facility(base, providers)
    linked_prof = build_professional_lines(linked) if not linked.empty else pd.DataFrame()

    df = pd.concat([prof, inst, pharm, other, linked_prof], ignore_index=True, sort=False)
    for col in ["ndc", "drug_name", "drug_type", "days_supply", "proc_mue_limit", "proc_description", "proc_base_allowed"]:
        if col not in df.columns:
            df[col] = np.nan
    df = apply_billing_and_payment_features(df)
    df = inject_fraud(df)
    df = build_rules(df)
    df["line_id"] = [f"L{start_encounter:09d}_{i:08d}" for i in range(len(df))]
    ordered = [
        "line_id", "episode_id", "encounter_id", "parent_encounter_id", "linked_facility_claim",
        "member_id", "provider_id", "provider_specialty", "provider_class", "provider_fraud_propensity",
        "service_date", "claim_family", "encounter_type", "place_of_service", "revenue_code",
        "diagnosis_code", "service_code", "drug_name", "drug_type", "ndc", "days_supply",
        "claim_line_no", "units", "member_age", "member_gender", "chronic_score", "plan_type",
        "annual_deductible", "deductible_remaining_start", "coinsurance_rate",
        "allowed_amount", "contract_rate", "billed_amount", "applied_to_deductible",
        "coinsurance_amount", "paid_amount", "claim_status", "denial_reason", "payment_type",
        "contract_discount_rate", "ncci_like_edit_flag", "mue_like_edit_flag", "external_edit_flag",
        "duplicate_signature_count", "provider_day_claim_count", "provider_avg_contract_rate",
        "service_code_avg_contract_rate", "billed_to_contract_ratio", "contract_to_allowed_ratio",
        "rule_duplicate", "rule_provider_volume", "rule_egregious_billed", "rule_price_above_contract",
        "rule_external_edit", "rule_denied_repeat", "rule_upcoding_pattern", "rule_member_submit_outlier",
        "strong_rule_hit", "rule_score", "flagged_for_review",
        "pricing_error_signal", "egregious_charge_signal", "upcoding_signal", "member_submit_outlier_signal",
        "fraud_label", "fraud_type", "rule_primary_reason", "rule_primary_reason_detected"
    ]
    for col in ordered:
        if col not in df.columns:
            df[col] = np.nan
    return df[ordered].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic healthcare claims generator with fraud and summary outputs")
    parser.add_argument("--n_encounters", type=int, default=250_000)
    parser.add_argument("--n_members", type=int, default=120_000)
    parser.add_argument("--n_providers", type=int, default=6_000)
    parser.add_argument("--chunk_size", type=int, default=100_000)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving files to: {output_dir.resolve()}")

    members = build_members(args.n_members)
    providers = build_providers(args.n_providers)
    members.to_csv(output_dir / "members.csv", index=False)
    providers.to_csv(output_dir / "providers.csv", index=False)

    chunk_files: List[str] = []
    combined_parts: List[pd.DataFrame] = []
    n_chunks = math.ceil(args.n_encounters / args.chunk_size)

    for ix in range(n_chunks):
        start = ix * args.chunk_size
        n = min(args.chunk_size, args.n_encounters - start)
        print(f"Building chunk {ix + 1} of {n_chunks} with {n:,} encounters...")
        chunk_df = generate_chunk(start_encounter=start, n_encounters=n, members=members, providers=providers)
        fname = write_chunk(chunk_df, output_dir, ix)
        chunk_files.append(fname)
        combined_parts.append(chunk_df.sample(min(150_000, len(chunk_df)), random_state=SEED + ix))
        print(f"  rows written: {len(chunk_df):,} -> {fname}")

    manifest = pd.DataFrame({"chunk_file": chunk_files})
    manifest.to_csv(output_dir / "chunk_manifest.csv", index=False)

    combined = pd.concat(combined_parts, ignore_index=True)
    # save a manageable combined sample for inspection
    combined.sample(min(250_000, len(combined)), random_state=SEED).to_csv(output_dir / "synthetic_claims_sample_250k.csv", index=False)
    summarize(combined, output_dir)
    print("Done.")
    print(f"Sample summary rows available in: {output_dir / 'synthetic_claims_sample_250k.csv'}")


if __name__ == "__main__":
    main()
