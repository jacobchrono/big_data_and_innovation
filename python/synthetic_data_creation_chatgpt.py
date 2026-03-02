import os
import math
import random
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# 0) Real-ish code subsets
# -----------------------------
# ICD-10-CM: real diagnosis codes (subset)
ICD10_CODES = [
    # metabolic / chronic
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("I10", "Essential (primary) hypertension"),
    ("E78.5", "Hyperlipidemia, unspecified"),
    ("E66.9", "Obesity, unspecified"),
    # respiratory / infectious
    ("J06.9", "Acute upper respiratory infection, unspecified"),
    ("J45.909", "Asthma, unspecified, uncomplicated"),
    ("J18.9", "Pneumonia, unspecified organism"),
    # MSK / pain
    ("M54.5", "Low back pain"),
    ("M25.561", "Pain in right knee"),
    # mental health
    ("F41.1", "Generalized anxiety disorder"),
    ("F32.9", "Major depressive disorder, single episode, unspecified"),
    # GI
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
    # injury
    ("S93.401A", "Sprain of unspecified ligament of right ankle, initial encounter"),
    # screening / preventive
    ("Z00.00", "Encounter for general adult medical exam without abnormal findings"),
    ("Z12.11", "Encounter for screening for malignant neoplasm of colon"),
]

# CPT: curated subset of real, widely-known codes (NOT a complete CPT set)
# If you have a licensed CPT file, feed your own list instead.
CPT_CODES = [
    ("99213", "Office/outpatient visit, established patient, low/moderate MDM"),
    ("99214", "Office/outpatient visit, established patient, moderate MDM"),
    ("99215", "Office/outpatient visit, established patient, high MDM"),
    ("93000", "Electrocardiogram with interpretation and report"),
    ("80053", "Comprehensive metabolic panel"),
    ("85025", "Complete CBC with differential"),
    ("71046", "Chest X-ray, 2 views"),
    ("45378", "Colonoscopy, diagnostic"),
    ("20610", "Arthrocentesis, major joint"),
]

# Simple specialty map (tiny, extend as needed)
SPECIALTIES = [
    ("FAMILY",  "Family Medicine"),
    ("IM",      "Internal Medicine"),
    ("ER",      "Emergency Medicine"),
    ("CARD",    "Cardiology"),
    ("ORTHO",   "Orthopedics"),
    ("RAD",     "Radiology"),
    ("LAB",     "Laboratory"),
    ("GI",      "Gastroenterology"),
]

PLACE_OF_SERVICE = [
    (11, "Office"),
    (21, "Inpatient Hospital"),
    (22, "Outpatient Hospital"),
    (23, "Emergency Room - Hospital"),
]


# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class SynthConfig:
    out_dir: str = "synthetic_claims_lake"
    seed: int = 7

    n_members: int = 500_000
    n_providers: int = 40_000

    start_month: str = "2024-01"
    end_month: str = "2025-12"

    # claims volume controls (per member-month)
    base_claim_rate: float = 0.35     # expected claim headers per member-month
    avg_lines_per_claim: float = 2.4  # average line count on each claim

    # fraud injection
    fraud_claim_rate: float = 0.008   # fraction of claim headers labeled fraud
    fraud_provider_rate: float = 0.004 # fraction of providers "bad actors"

    # amounts
    billed_multiplier_mean: float = 1.8  # billed ~= allowed * multiplier
    billed_multiplier_sd: float = 0.35

    # parquet partitioning
    partition_cols: Tuple[str, ...] = ("year", "month")


def month_range(start_ym: str, end_ym: str) -> List[str]:
    start = pd.Period(start_ym, freq="M")
    end = pd.Period(end_ym, freq="M")
    periods = pd.period_range(start, end, freq="M")
    return [str(p) for p in periods]  # "YYYY-MM"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rng_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# 2) Dimension generators
# -----------------------------
def gen_members(cfg: SynthConfig) -> pd.DataFrame:
    rng_all(cfg.seed)
    member_id = np.arange(1, cfg.n_members + 1, dtype=np.int64)

    # age distribution: mixture (kids, adults, seniors)
    mix = np.random.choice([0, 1, 2], size=cfg.n_members, p=[0.22, 0.62, 0.16])
    ages = np.where(mix == 0, np.random.randint(0, 18, cfg.n_members),
           np.where(mix == 1, np.random.randint(18, 65, cfg.n_members),
                            np.random.randint(65, 90, cfg.n_members)))

    sex = np.random.choice(["F", "M"], size=cfg.n_members, p=[0.51, 0.49])

    # a crude "risk score" proxy, lognormal-ish
    risk = np.clip(np.random.lognormal(mean=0.0, sigma=0.55, size=cfg.n_members), 0.3, 6.0)

    # geography: state + 3-digit zip prefix
    states = np.random.choice(
        ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "CO", "AZ", "WA", "MA"],
        size=cfg.n_members,
        p=[0.12, 0.11, 0.09, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.04, 0.04, 0.04, 0.03]
    )
    zip3 = np.random.randint(100, 999, size=cfg.n_members)

    # plan assignment
    plan_id = np.random.choice(["HMO_SILVER", "PPO_GOLD", "HDHP_BRONZE"], size=cfg.n_members, p=[0.45, 0.35, 0.20])

    df = pd.DataFrame({
        "member_id": member_id,
        "age": ages.astype(np.int16),
        "sex": sex,
        "risk_score": risk.astype(np.float32),
        "state": states,
        "zip3": zip3.astype(np.int16),
        "plan_id": plan_id,
    })
    return df


def gen_providers(cfg: SynthConfig) -> pd.DataFrame:
    rng_all(cfg.seed + 1)
    provider_id = np.arange(1, cfg.n_providers + 1, dtype=np.int64)

    spec_codes = [s[0] for s in SPECIALTIES]
    spec = np.random.choice(spec_codes, size=cfg.n_providers,
                            p=[0.30, 0.22, 0.10, 0.07, 0.08, 0.08, 0.10, 0.05])

    states = np.random.choice(
        ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "CO", "AZ", "WA", "MA"],
        size=cfg.n_providers
    )

    network_status = np.random.choice(["INN", "OON"], size=cfg.n_providers, p=[0.83, 0.17])

    # Choose a small set of "bad actor" providers
    bad_flag = np.zeros(cfg.n_providers, dtype=np.int8)
    n_bad = max(1, int(cfg.n_providers * cfg.fraud_provider_rate))
    bad_idx = np.random.choice(np.arange(cfg.n_providers), size=n_bad, replace=False)
    bad_flag[bad_idx] = 1

    df = pd.DataFrame({
        "provider_id": provider_id,
        "specialty": spec,
        "state": states,
        "network_status": network_status,
        "is_bad_provider": bad_flag,
    })
    return df


# -----------------------------
# 3) Fee schedule model (simple, extendable)
# -----------------------------
def allowed_amount(proc_code: str, specialty: str, pos: int) -> float:
    """
    Very simplified allowed-amount generator.
    In reality you'd key off fee schedule, geography, contract, modifiers, etc.
    """
    base = {
        "99213": 90,
        "99214": 140,
        "99215": 220,
        "93000": 55,
        "80053": 40,
        "85025": 25,
        "71046": 70,
        "45378": 750,
        "20610": 120,
    }.get(proc_code, 100)

    # Specialty uplift
    uplift = {
        "FAMILY": 1.0, "IM": 1.05, "ER": 1.15, "CARD": 1.35,
        "ORTHO": 1.25, "RAD": 1.20, "LAB": 0.90, "GI": 1.30
    }.get(specialty, 1.0)

    # Place-of-service facility effect
    pos_factor = {11: 1.0, 22: 1.25, 23: 1.35, 21: 1.55}.get(pos, 1.0)

    # Add noise
    amt = base * uplift * pos_factor
    return float(np.random.lognormal(mean=math.log(max(amt, 1.0)), sigma=0.18))


# -----------------------------
# 4) Claim generation per month chunk
# -----------------------------
def gen_month_claims(
    cfg: SynthConfig,
    members: pd.DataFrame,
    providers: pd.DataFrame,
    ym: str,
    claim_id_start: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Generates one month of:
      - claim_header
      - claim_line
      - claim_dx
      - fraud_labels
    Returns updated claim_id_end+1.
    """
    year, month = ym.split("-")
    year_i, month_i = int(year), int(month)

    # Member-month exposure
    n = len(members)
    # Expected claims per member-month scaled by risk score
    lam = cfg.base_claim_rate * members["risk_score"].to_numpy()
    # cap extreme utilization a bit
    lam = np.clip(lam, 0.02, 3.0)

    # number of claims for each member
    n_claims = np.random.poisson(lam=lam).astype(np.int16)
    claim_rows = int(n_claims.sum())

    if claim_rows == 0:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), claim_id_start)

    # Build claim_header rows
    member_ids_rep = np.repeat(members["member_id"].to_numpy(), n_claims)
    # Assign providers: weighted toward in-state providers (simple)
    prov_ids = providers["provider_id"].to_numpy()
    prov_states = providers["state"].to_numpy()
    mem_states = members.set_index("member_id").loc[member_ids_rep, "state"].to_numpy()

    # Pick provider indexes: bias matching state
    idx_all = np.random.randint(0, len(providers), size=claim_rows)
    # force some same-state matches
    same_state_mask = np.random.rand(claim_rows) < 0.65
    if same_state_mask.any():
        # sample from providers with same state (fallback to random if none)
        for st in np.unique(mem_states[same_state_mask]):
            m = same_state_mask & (mem_states == st)
            candidates = np.where(prov_states == st)[0]
            if len(candidates) > 0:
                idx_all[m] = np.random.choice(candidates, size=m.sum(), replace=True)

    provider_ids_rep = prov_ids[idx_all]
    specialty_rep = providers.set_index("provider_id").loc[provider_ids_rep, "specialty"].to_numpy()
    net_rep = providers.set_index("provider_id").loc[provider_ids_rep, "network_status"].to_numpy()
    badprov_rep = providers.set_index("provider_id").loc[provider_ids_rep, "is_bad_provider"].to_numpy()

    claim_ids = np.arange(claim_id_start, claim_id_start + claim_rows, dtype=np.int64)

    # service dates inside month
    days_in_month = pd.Period(ym, freq="M").days_in_month
    svc_day = np.random.randint(1, days_in_month + 1, size=claim_rows)
    svc_date = pd.to_datetime([f"{ym}-{d:02d}" for d in svc_day])

    claim_type = np.random.choice(["PROF", "INST"], size=claim_rows, p=[0.86, 0.14])
    pos = np.random.choice([p[0] for p in PLACE_OF_SERVICE], size=claim_rows, p=[0.70, 0.12, 0.10, 0.08])

    # Initial fraud labeling (we’ll adjust based on bad providers too)
    fraud_flag = (np.random.rand(claim_rows) < cfg.fraud_claim_rate) | ((badprov_rep == 1) & (np.random.rand(claim_rows) < 0.20))
    fraud_flag = fraud_flag.astype(np.int8)

    header = pd.DataFrame({
        "claim_id": claim_ids,
        "member_id": member_ids_rep.astype(np.int64),
        "provider_id": provider_ids_rep.astype(np.int64),
        "claim_type": claim_type,
        "place_of_service": pos.astype(np.int16),
        "service_date": svc_date,
        "network_status": net_rep,
        "year": year_i,
        "month": month_i,
    })

    # Diagnosis bridge: 1–4 diagnoses typical on professional claims, more on institutional
    max_dx = np.where(claim_type == "INST", np.random.randint(3, 10, size=claim_rows), np.random.randint(1, 5, size=claim_rows))
    icd_codes = [c[0] for c in ICD10_CODES]
    dx_list = []
    for cid, k in zip(claim_ids, max_dx):
        # pick diagnoses with slight chronic bias
        probs = np.array([0.12, 0.12, 0.10, 0.07, 0.08, 0.06, 0.05, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 0.03, 0.02, 0.01])
        probs = probs / probs.sum()
        chosen = np.random.choice(icd_codes, size=int(k), replace=True, p=probs)
        # ensure rank 1 exists
        for rank, code in enumerate(chosen, start=1):
            dx_list.append((cid, rank, code))
    claim_dx = pd.DataFrame(dx_list, columns=["claim_id", "dx_rank", "icd10_code"])

    # Claim lines: lines per claim ~ Poisson around avg_lines_per_claim
    lines_per = np.maximum(1, np.random.poisson(cfg.avg_lines_per_claim, size=claim_rows)).astype(np.int16)
    line_rows = int(lines_per.sum())

    # repeat claim attributes to line level
    claim_ids_line = np.repeat(claim_ids, lines_per)
    member_ids_line = np.repeat(member_ids_rep, lines_per)
    provider_ids_line = np.repeat(provider_ids_rep, lines_per)
    spec_line = np.repeat(specialty_rep, lines_per)
    pos_line = np.repeat(pos, lines_per)
    svc_date_line = np.repeat(svc_date, lines_per)
    fraud_line_flag = np.repeat(fraud_flag, lines_per)

    # pick proc codes with specialty bias
    proc_codes = [c[0] for c in CPT_CODES]
    proc_probs_by_spec: Dict[str, List[float]] = {
        "FAMILY": [0.40, 0.25, 0.05, 0.03, 0.10, 0.10, 0.02, 0.03, 0.02],
        "IM":     [0.35, 0.30, 0.08, 0.04, 0.10, 0.08, 0.02, 0.02, 0.01],
        "ER":     [0.20, 0.30, 0.15, 0.07, 0.08, 0.08, 0.06, 0.03, 0.03],
        "CARD":   [0.20, 0.25, 0.08, 0.22, 0.08, 0.07, 0.05, 0.03, 0.02],
        "ORTHO":  [0.18, 0.22, 0.05, 0.02, 0.07, 0.06, 0.10, 0.02, 0.28],
        "RAD":    [0.05, 0.07, 0.03, 0.02, 0.05, 0.05, 0.65, 0.05, 0.03],
        "LAB":    [0.03, 0.04, 0.01, 0.01, 0.45, 0.40, 0.02, 0.02, 0.02],
        "GI":     [0.12, 0.15, 0.05, 0.02, 0.06, 0.06, 0.03, 0.45, 0.06],
    }

    # generate proc codes
    procs = []
    for s in spec_line:
        probs = np.array(proc_probs_by_spec.get(s, [1/len(proc_codes)] * len(proc_codes)))
        probs = probs / probs.sum()
        procs.append(np.random.choice(proc_codes, p=probs))
    procs = np.array(procs, dtype=object)

    units = np.maximum(1, np.random.poisson(lam=1.1, size=line_rows)).astype(np.int16)

    # amounts
    allowed = np.array([allowed_amount(pc, sp, ps) for pc, sp, ps in zip(procs, spec_line, pos_line)], dtype=np.float32)
    billed_mult = np.random.normal(cfg.billed_multiplier_mean, cfg.billed_multiplier_sd, size=line_rows)
    billed_mult = np.clip(billed_mult, 1.05, 4.0).astype(np.float32)
    billed = (allowed * billed_mult * units).astype(np.float32)

    # paid: simplify as allowed * (1 - member cost share)
    # plan + network changes member share
    member_share = np.where(np.array(list(np.repeat(net_rep, lines_per))) == "OON", 0.45, 0.22)
    member_share = np.clip(member_share + np.random.normal(0, 0.05, size=line_rows), 0.05, 0.80).astype(np.float32)
    paid = (allowed * units * (1 - member_share)).astype(np.float32)

    line_num = np.concatenate([np.arange(1, k + 1) for k in lines_per]).astype(np.int16)

    # Basic dx pointer: point to rank 1 mostly
    dx_pointer = np.ones(line_rows, dtype=np.int8)

    lines = pd.DataFrame({
        "claim_id": claim_ids_line.astype(np.int64),
        "line_num": line_num,
        "member_id": member_ids_line.astype(np.int64),
        "provider_id": provider_ids_line.astype(np.int64),
        "service_date": svc_date_line,
        "place_of_service": pos_line.astype(np.int16),
        "specialty": spec_line,
        "proc_code": procs,
        "units": units,
        "billed_amount": billed,
        "allowed_amount": (allowed * units).astype(np.float32),
        "paid_amount": paid,
        "dx_pointer_primary": dx_pointer,
        "is_fraud_line": fraud_line_flag.astype(np.int8),
        "year": year_i,
        "month": month_i,
    })

    # -----------------------------
    # Fraud injection transforms
    # -----------------------------
    fraud_labels = []
    if fraud_flag.sum() > 0:
        fraud_claim_ids = claim_ids[fraud_flag == 1]

        # Assign a fraud type per fraudulent claim
        fraud_types = np.random.choice(
            ["DUPLICATE", "UPCODE", "PHANTOM", "EXCESS_UNITS", "UNBUNDLE"],
            size=len(fraud_claim_ids),
            p=[0.25, 0.25, 0.15, 0.20, 0.15]
        )

        # Apply at line level where relevant
        for cid, ftype in zip(fraud_claim_ids, fraud_types):
            fraud_labels.append((cid, 1, ftype))

            idx = lines.index[lines["claim_id"] == cid].to_numpy()
            if len(idx) == 0:
                continue

            if ftype == "UPCODE":
                # push office visits upward where present, else swap first line to higher value code
                em_map = {"99213": "99215", "99214": "99215"}
                pick = idx[0]
                pc = lines.at[pick, "proc_code"]
                if pc in em_map:
                    lines.at[pick, "proc_code"] = em_map[pc]
                else:
                    lines.at[pick, "proc_code"] = "99215"
                # inflate allowed a bit
                lines.at[pick, "allowed_amount"] *= 1.45
                lines.at[pick, "billed_amount"] *= 1.55
                lines.at[pick, "paid_amount"] *= 1.35

            elif ftype == "EXCESS_UNITS":
                pick = np.random.choice(idx)
                lines.at[pick, "units"] = int(min(50, max(8, lines.at[pick, "units"] * np.random.randint(6, 18))))
                # recompute billed/allowed/paid roughly
                unit = lines.at[pick, "units"]
                # per-unit allowed approximated from current allowed/units
                per = float(lines.at[pick, "allowed_amount"]) / max(1, int(lines.at[pick, "units"]))
                # since we overwrote units, just scale amounts aggressively
                lines.at[pick, "allowed_amount"] *= (unit / max(1, unit // np.random.randint(6, 18)))
                lines.at[pick, "billed_amount"] *= 1.8
                lines.at[pick, "paid_amount"] *= 1.5

            elif ftype == "PHANTOM":
                # add a line that "doesn't fit" and tends to pay high
                new_line = lines.loc[idx[0]].copy()
                new_line["line_num"] = int(lines.loc[idx, "line_num"].max() + 1)
                new_line["proc_code"] = np.random.choice(["45378", "99215"])  # expensive-ish in our subset
                new_line["units"] = 1
                new_line["allowed_amount"] *= 2.2
                new_line["billed_amount"] *= 2.7
                new_line["paid_amount"] *= 2.0
                lines = pd.concat([lines, pd.DataFrame([new_line])], ignore_index=True)

            elif ftype == "UNBUNDLE":
                # replace one line with multiple cheaper components (conceptual)
                pick = idx[0]
                lines.at[pick, "proc_code"] = "80053"
                lines.at[pick, "allowed_amount"] *= 0.65
                lines.at[pick, "billed_amount"] *= 0.70
                # add another lab line
                add = lines.loc[pick].copy()
                add["line_num"] = int(lines.loc[idx, "line_num"].max() + 1)
                add["proc_code"] = "85025"
                add["allowed_amount"] *= 0.85
                add["billed_amount"] *= 0.90
                add["paid_amount"] *= 0.85
                lines = pd.concat([lines, pd.DataFrame([add])], ignore_index=True)

            elif ftype == "DUPLICATE":
                # duplicate all lines for this claim with a new claim_id later in the month
                # (real duplicates often show up with same data, sometimes a day offset)
                dup_claim_id = int(lines["claim_id"].max() + np.random.randint(10_000, 50_000))
                sub = lines[lines["claim_id"] == cid].copy()
                sub["claim_id"] = dup_claim_id
                sub["service_date"] = sub["service_date"] + pd.to_timedelta(np.random.choice([0, 1, 2]), unit="D")
                sub["is_fraud_line"] = 1
                # label the duplicate claim too
                fraud_labels.append((dup_claim_id, 1, "DUPLICATE"))
                header_dup = header[header["claim_id"] == cid].copy()
                header_dup["claim_id"] = dup_claim_id
                header_dup["service_date"] = header_dup["service_date"] + pd.to_timedelta(np.random.choice([0, 1, 2]), unit="D")
                header = pd.concat([header, header_dup], ignore_index=True)
                lines = pd.concat([lines, sub], ignore_index=True)

        fraud_df = pd.DataFrame(fraud_labels, columns=["claim_id", "fraud_flag", "fraud_type"]).drop_duplicates("claim_id")
    else:
        fraud_df = pd.DataFrame(columns=["claim_id", "fraud_flag", "fraud_type"])

    # Recompute header totals from lines (post-fraud changes)
    totals = lines.groupby("claim_id", as_index=False).agg(
        total_billed=("billed_amount", "sum"),
        total_allowed=("allowed_amount", "sum"),
        total_paid=("paid_amount", "sum"),
        line_count=("line_num", "max"),
    )
    header = header.merge(totals, on="claim_id", how="left")

    claim_id_next = int(max(header["claim_id"].max(), lines["claim_id"].max()) + 1)
    return header, lines, claim_dx, fraud_df, claim_id_next


# -----------------------------
# 5) Parquet writing (partitioned)
# -----------------------------
def write_partitioned_parquet(df: pd.DataFrame, base_path: str, partition_cols: Tuple[str, ...], filename: str) -> None:
    """
    Writes df to base_path/filename partitioned by partition_cols using pyarrow.
    Creates a dataset-like folder layout.
    """
    if df.empty:
        return

    ensure_dir(base_path)
    table = pa.Table.from_pandas(df, preserve_index=False)

    # We write one file per call into a partitioned dataset directory.
    pq.write_to_dataset(
        table,
        root_path=os.path.join(base_path, filename),
        partition_cols=list(partition_cols),
        existing_data_behavior="overwrite_or_ignore",
        compression="snappy",
        use_dictionary=True
    )


# -----------------------------
# 6) Orchestrator
# -----------------------------
def build_synthetic_lake(cfg: SynthConfig) -> None:
    rng_all(cfg.seed)
    ensure_dir(cfg.out_dir)

    print("Generating dimensions...")
    members = gen_members(cfg)
    providers = gen_providers(cfg)

    # Write dimensions once
    write_partitioned_parquet(members.assign(year=0, month=0), cfg.out_dir, ("year", "month"), "dim_member")
    write_partitioned_parquet(providers.assign(year=0, month=0), cfg.out_dir, ("year", "month"), "dim_provider")

    claim_id_next = 1
    months = month_range(cfg.start_month, cfg.end_month)

    print(f"Generating claims from {cfg.start_month} to {cfg.end_month} ({len(months)} months)...")
    for i, ym in enumerate(months, start=1):
        header, lines, dx, fraud, claim_id_next = gen_month_claims(cfg, members, providers, ym, claim_id_next)

        write_partitioned_parquet(header, cfg.out_dir, cfg.partition_cols, "fact_claim_header")
        write_partitioned_parquet(lines, cfg.out_dir, cfg.partition_cols, "fact_claim_line")
        write_partitioned_parquet(dx.assign(year=int(ym[:4]), month=int(ym[5:])), cfg.out_dir, cfg.partition_cols, "bridge_claim_dx")
        if not fraud.empty:
            # fraud doesn't naturally have partition columns; add them from header if possible
            fraud2 = fraud.merge(header[["claim_id", "year", "month"]], on="claim_id", how="left")
            write_partitioned_parquet(fraud2, cfg.out_dir, cfg.partition_cols, "fraud_labels")

        if i % 3 == 0:
            print(f"  wrote through month {ym}: headers={len(header):,}, lines={len(lines):,}, next_claim_id={claim_id_next:,}")

    print("Done.")
    print(f"Output lake at: {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    cfg = SynthConfig(
        out_dir="synthetic_claims_lake",
        n_members=300_000,      # scale up as your machine allows
        n_providers=25_000,
        start_month="2024-01",
        end_month="2025-12",
        base_claim_rate=0.40,
        avg_lines_per_claim=2.6,
        fraud_claim_rate=0.010,
        fraud_provider_rate=0.005
    )
    build_synthetic_lake(cfg)