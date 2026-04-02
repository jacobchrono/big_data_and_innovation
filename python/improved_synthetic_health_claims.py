# Jacob Clement
# 04/01/2026
# Improved Synthetic Healthcare Claims Dataset with Fraud Injection and Detection Features:
# This script generates a synthetic healthcare claims dataset with enhanced realism, including more detailed provider and member characteristics, a wider variety of     # procedures and diagnosis codes, and more sophisticated fraud injection patterns. It also includes the generation of features that could be used for fraud detection    # modeling, such as duplicate claim signatures and provider-day volume metrics.
# run in powershell from project root:
# python improved_synthetic_health_claims.py --n_claims 1000000 --n_members 120000 --n_providers 6000 --chunk_size 250000

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


SEED = 42
DEFAULT_OUTPUT_DIR = Path('/mnt/data/synthetic_claims_v2_output')

# -----------------------------
# Reference tables
# -----------------------------
# Notes:
# - This is a prototype, not a production claims simulator.
# - CPT/HCPCS usage here is simplified to create realistic-ish data distributions.
# - Institutional revenue code examples are intentionally limited and simplified.

DIAGNOSIS_CODES = np.array([
    'E11.9', 'I10', 'E78.5', 'J06.9', 'J20.9', 'J18.9', 'M54.50', 'M25.561',
    'M25.562', 'R10.9', 'R07.9', 'R51.9', 'N39.0', 'K21.9', 'K52.9', 'F41.9',
    'F32.A', 'Z00.00', 'Z00.01', 'Z79.4', 'Z79.84', 'R73.03', 'E66.9', 'G47.33',
    'J45.909', 'B34.9', 'R05.9', 'R06.02', 'I25.10', 'I48.91', 'Z12.11', 'Z23'
], dtype=object)

# code, base_allowed, typical_units, category, care_setting_hint
PROFESSIONAL_PROCEDURES = pd.DataFrame([
    ('99212', 70.0, 1, 'E&M', 'professional'),
    ('99213', 95.0, 1, 'E&M', 'professional'),
    ('99214', 145.0, 1, 'E&M', 'professional'),
    ('99215', 210.0, 1, 'E&M', 'professional'),
    ('93000', 55.0, 1, 'Cardiology', 'professional'),
    ('71046', 85.0, 1, 'Imaging', 'professional'),
    ('80048', 28.0, 1, 'Lab', 'professional'),
    ('80053', 35.0, 1, 'Lab', 'professional'),
    ('83036', 25.0, 1, 'Lab', 'professional'),
    ('85025', 22.0, 1, 'Lab', 'professional'),
    ('84443', 30.0, 1, 'Lab', 'professional'),
    ('36415', 8.0, 1, 'Lab', 'professional'),
    ('90471', 28.0, 1, 'Preventive', 'professional'),
    ('90686', 32.0, 1, 'Preventive', 'professional'),
    ('90834', 120.0, 1, 'Behavioral', 'professional'),
    ('90837', 165.0, 1, 'Behavioral', 'professional'),
    ('97110', 40.0, 2, 'Therapy', 'professional'),
    ('97112', 45.0, 2, 'Therapy', 'professional'),
    ('97140', 38.0, 2, 'Therapy', 'professional'),
    ('11721', 60.0, 1, 'Podiatry', 'professional'),
    ('G0438', 160.0, 1, 'Preventive', 'professional'),
    ('G0439', 175.0, 1, 'Preventive', 'professional'),
], columns=['procedure_code', 'base_allowed', 'typical_units', 'category', 'care_setting_hint'])

# Institutional lines with revenue codes. HCPCS is optional in real claims; here we keep some lines HCPCS-linked.
# Revenue code examples chosen to make the institutional side visibly different.
INSTITUTIONAL_PROCEDURES = pd.DataFrame([
    ('0450', '99284', 850.0, 1, 'ED', 'institutional'),
    ('0450', '99285', 1400.0, 1, 'ED', 'institutional'),
    ('0510', '99281', 220.0, 1, 'Clinic', 'institutional'),
    ('0510', '99282', 350.0, 1, 'Clinic', 'institutional'),
    ('0360', '47562', 6500.0, 1, 'OR', 'institutional'),
    ('0360', '27447', 12000.0, 1, 'OR', 'institutional'),
    ('0762', None, 450.0, 6, 'Observation', 'institutional'),
    ('0300', '80053', 140.0, 1, 'Lab', 'institutional'),
    ('0300', '85025', 95.0, 1, 'Lab', 'institutional'),
    ('0250', None, 75.0, 3, 'Pharmacy', 'institutional'),
    ('0270', None, 60.0, 2, 'Supplies', 'institutional'),
    ('0636', 'J1885', 55.0, 1, 'Drugs', 'institutional'),
], columns=['revenue_code', 'procedure_code', 'base_allowed', 'typical_units', 'category', 'care_setting_hint'])

SPECIALTY_TO_CODES = {
    'Primary Care': np.array(['99212', '99213', '99214', '80048', '80053', '83036', '85025', '84443', '36415', 'G0439', '90471', '90686'], dtype=object),
    'Internal Medicine': np.array(['99213', '99214', '99215', '80053', '83036', '85025', '93000'], dtype=object),
    'Behavioral Health': np.array(['90834', '90837', '99213', '99214'], dtype=object),
    'Cardiology': np.array(['93000', '99214', '99215', '71046'], dtype=object),
    'Therapy': np.array(['97110', '97112', '97140', '99213'], dtype=object),
    'Podiatry': np.array(['11721', '99213', '99214'], dtype=object),
    'Urgent Care': np.array(['99213', '99214', '71046', '80053', '85025'], dtype=object),
}

PROFESSIONAL_POS = np.array(['11', '22', '02', '23'], dtype=object)
PROFESSIONAL_POS_PROB = np.array([0.62, 0.17, 0.09, 0.12])

FACILITY_POS = np.array(['21', '22', '23'], dtype=object)
FACILITY_POS_PROB = np.array([0.22, 0.34, 0.44])


# -----------------------------
# Helpers
# -----------------------------
def make_members(n_members: int, rng: np.random.Generator) -> pd.DataFrame:
    ages = rng.integers(0, 91, size=n_members, dtype=np.int16)
    genders = rng.choice(np.array(['F', 'M'], dtype=object), size=n_members, p=[0.53, 0.47])
    chronic_score = rng.poisson(1.35, size=n_members).astype(np.int8)
    high_risk = ((ages >= 70) | (chronic_score >= 4)).astype(np.int8)
    return pd.DataFrame({
        'member_id': np.arange(1, n_members + 1, dtype=np.int32),
        'member_age': ages,
        'member_gender': genders,
        'chronic_score': chronic_score,
        'high_risk_member': high_risk,
    })


def make_providers(n_providers: int, rng: np.random.Generator) -> pd.DataFrame:
    specialties = np.array([
        'Primary Care', 'Internal Medicine', 'Behavioral Health', 'Cardiology',
        'Therapy', 'Podiatry', 'Urgent Care', 'Hospital'
    ], dtype=object)
    specialty_prob = np.array([0.23, 0.18, 0.10, 0.08, 0.11, 0.06, 0.10, 0.14])
    specialty = rng.choice(specialties, size=n_providers, p=specialty_prob)

    provider_type = np.where(specialty == 'Hospital', 'facility', 'professional')

    # Fraud propensity: most providers clean, a few suspicious, very few bad actors.
    provider_class = rng.choice(
        np.array(['clean', 'low', 'medium', 'high'], dtype=object),
        size=n_providers,
        p=[0.82, 0.11, 0.05, 0.02]
    )
    fraud_propensity = np.select(
        [provider_class == 'clean', provider_class == 'low', provider_class == 'medium', provider_class == 'high'],
        [0.0, rng.uniform(0.002, 0.01, n_providers), rng.uniform(0.02, 0.06, n_providers), rng.uniform(0.08, 0.18, n_providers)],
        default=0.0,
    ).astype(np.float32)

    base_volume = np.select(
        [provider_type == 'facility', specialty == 'Urgent Care'],
        [rng.integers(2000, 12000, size=n_providers), rng.integers(1000, 4500, size=n_providers)],
        default=rng.integers(350, 2200, size=n_providers),
    ).astype(np.int32)

    return pd.DataFrame({
        'provider_id': np.arange(1, n_providers + 1, dtype=np.int32),
        'specialty': specialty,
        'provider_type': provider_type,
        'provider_class': provider_class,
        'fraud_propensity': fraud_propensity,
        'base_volume': base_volume,
    })


def weighted_provider_draw(providers: pd.DataFrame, n_claims: int, rng: np.random.Generator) -> np.ndarray:
    weights = providers['base_volume'].to_numpy(dtype=np.float64)
    weights = weights / weights.sum()
    return rng.choice(providers['provider_id'].to_numpy(), size=n_claims, p=weights)


def build_code_maps() -> tuple[dict[str, tuple], dict[str, tuple], np.ndarray, np.ndarray]:
    prof_map = {
        row['procedure_code']: (row['base_allowed'], row['typical_units'], row['category'])
        for _, row in PROFESSIONAL_PROCEDURES.iterrows()
    }
    inst_map = {
        f"{row['revenue_code']}|{row['procedure_code']}": (row['base_allowed'], row['typical_units'], row['category'])
        for _, row in INSTITUTIONAL_PROCEDURES.iterrows()
    }
    prof_codes = PROFESSIONAL_PROCEDURES['procedure_code'].to_numpy(dtype=object)
    inst_keys = np.array([f"{r}|{p}" for r, p in zip(INSTITUTIONAL_PROCEDURES['revenue_code'], INSTITUTIONAL_PROCEDURES['procedure_code'])], dtype=object)
    return prof_map, inst_map, prof_codes, inst_keys


def assign_professional_codes(specialties: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(len(specialties), dtype=object)
    unique_specialties = np.unique(specialties)
    for sp in unique_specialties:
        idx = np.where(specialties == sp)[0]
        choices = SPECIALTY_TO_CODES.get(sp, SPECIALTY_TO_CODES['Primary Care'])
        out[idx] = rng.choice(choices, size=len(idx))
    return out


def assign_institutional_codes(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    probs = np.array([0.22, 0.08, 0.09, 0.06, 0.06, 0.05, 0.10, 0.09, 0.08, 0.08, 0.05, 0.04])
    chosen = INSTITUTIONAL_PROCEDURES.sample(n=n, replace=True, weights=probs, random_state=int(rng.integers(0, 1_000_000_000)))
    rev = chosen['revenue_code'].to_numpy(dtype=object)
    cpt = chosen['procedure_code'].fillna('').to_numpy(dtype=object)
    return rev, cpt


def generate_chunk(
    chunk_id: int,
    start_claim_id: int,
    n_claims: int,
    members: pd.DataFrame,
    providers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    provider_ids = weighted_provider_draw(providers, n_claims, rng)
    provider_slice = providers.set_index('provider_id').loc[provider_ids].reset_index()

    member_ids = rng.choice(members['member_id'].to_numpy(), size=n_claims)
    member_slice = members.set_index('member_id').loc[member_ids].reset_index()

    dates = pd.Timestamp('2025-01-01') + pd.to_timedelta(rng.integers(0, 365, n_claims), unit='D')
    dx = rng.choice(DIAGNOSIS_CODES, size=n_claims)

    is_facility = provider_slice['provider_type'].to_numpy() == 'facility'
    claim_family = np.where(is_facility, 'institutional', 'professional')

    pos = np.empty(n_claims, dtype=object)
    pos[~is_facility] = rng.choice(PROFESSIONAL_POS, size=(~is_facility).sum(), p=PROFESSIONAL_POS_PROB)
    pos[is_facility] = rng.choice(FACILITY_POS, size=is_facility.sum(), p=FACILITY_POS_PROB)

    telehealth = (pos == '02').astype(np.int8)
    inpatient = (pos == '21').astype(np.int8)
    emergency = (pos == '23').astype(np.int8)

    procedure_code = np.empty(n_claims, dtype=object)
    revenue_code = np.full(n_claims, '', dtype=object)
    category = np.empty(n_claims, dtype=object)
    base_allowed = np.empty(n_claims, dtype=np.float32)
    typical_units = np.empty(n_claims, dtype=np.int16)

    prof_mask = ~is_facility
    facility_mask = is_facility

    prof_codes = assign_professional_codes(provider_slice.loc[prof_mask, 'specialty'].to_numpy(dtype=object), rng)
    procedure_code[prof_mask] = prof_codes
    prof_lookup = PROFESSIONAL_PROCEDURES.set_index('procedure_code').loc[prof_codes]
    category[prof_mask] = prof_lookup['category'].to_numpy(dtype=object)
    base_allowed[prof_mask] = prof_lookup['base_allowed'].to_numpy(dtype=np.float32)
    typical_units[prof_mask] = prof_lookup['typical_units'].to_numpy(dtype=np.int16)

    inst_rev, inst_cpt = assign_institutional_codes(facility_mask.sum(), rng)
    revenue_code[facility_mask] = inst_rev
    procedure_code[facility_mask] = inst_cpt
    inst_lookup = INSTITUTIONAL_PROCEDURES.assign(proc_fill=INSTITUTIONAL_PROCEDURES['procedure_code'].fillna('')).set_index(['revenue_code', 'proc_fill']).loc[list(zip(inst_rev, inst_cpt))]
    category[facility_mask] = inst_lookup['category'].to_numpy(dtype=object)
    base_allowed[facility_mask] = inst_lookup['base_allowed'].to_numpy(dtype=np.float32)
    typical_units[facility_mask] = inst_lookup['typical_units'].to_numpy(dtype=np.int16)

    units = np.maximum(1, rng.poisson(np.maximum(typical_units, 1))).astype(np.int16)
    units = np.where(is_facility, np.clip(units, 1, 12), np.clip(units, 1, 6)).astype(np.int16)

    noise = rng.lognormal(mean=0.0, sigma=0.22, size=n_claims).astype(np.float32)
    age_factor = 1.0 + np.clip(member_slice['member_age'].to_numpy() - 50, 0, None) * 0.0025
    severity_factor = 1.0 + member_slice['chronic_score'].to_numpy() * 0.04 + inpatient * 0.35 + emergency * 0.08

    allowed = (base_allowed * units * noise * age_factor * severity_factor).astype(np.float32)
    billed = (allowed * rng.uniform(1.08, 1.50, n_claims)).astype(np.float32)
    paid = (allowed * rng.uniform(0.72, 1.00, n_claims)).astype(np.float32)

    # Fraud assignment at provider level.
    provider_fraud_prop = provider_slice['fraud_propensity'].to_numpy(dtype=np.float32)
    fraud_roll = rng.random(n_claims)
    fraudulent = fraud_roll < provider_fraud_prop

    fraud_type = np.full(n_claims, 'none', dtype=object)

    # Fraud modes among fraudulent lines.
    # Not all provider fraud is intentional on every claim, and most providers have zero fraud.
    if fraudulent.any():
        fraud_modes = rng.choice(
            np.array(['duplicate', 'upcoding', 'excessive_units', 'unbundling', 'high_cost_facility'], dtype=object),
            size=fraudulent.sum(),
            p=[0.22, 0.22, 0.20, 0.16, 0.20],
        )
        f_idx = np.where(fraudulent)[0]
        fraud_type[f_idx] = fraud_modes

        # duplicate: later detection uses duplicate signatures and replay counts
        dup_idx = f_idx[fraud_modes == 'duplicate']
        if len(dup_idx):
            billed[dup_idx] *= 1.0
            paid[dup_idx] *= 1.0

        # upcoding: move mid-level office visits to higher-level office visits when professional
        up_idx = f_idx[fraud_modes == 'upcoding']
        if len(up_idx):
            prof_up_idx = up_idx[claim_family[up_idx] == 'professional']
            if len(prof_up_idx):
                procedure_code[prof_up_idx] = '99215'
                category[prof_up_idx] = 'E&M'
                base_allowed[prof_up_idx] = 210.0
                units[prof_up_idx] = 1
                allowed[prof_up_idx] = base_allowed[prof_up_idx] * rng.uniform(1.0, 1.25, len(prof_up_idx))
                billed[prof_up_idx] = allowed[prof_up_idx] * rng.uniform(1.25, 1.65, len(prof_up_idx))
                paid[prof_up_idx] = allowed[prof_up_idx] * rng.uniform(0.8, 1.0, len(prof_up_idx))
            fac_up_idx = up_idx[claim_family[up_idx] == 'institutional']
            if len(fac_up_idx):
                revenue_code[fac_up_idx] = '0360'
                category[fac_up_idx] = 'OR'
                allowed[fac_up_idx] *= rng.uniform(2.0, 3.0, len(fac_up_idx))
                billed[fac_up_idx] *= rng.uniform(2.0, 3.0, len(fac_up_idx))
                paid[fac_up_idx] *= rng.uniform(1.7, 2.6, len(fac_up_idx))

        # excessive units
        ex_idx = f_idx[fraud_modes == 'excessive_units']
        if len(ex_idx):
            units[ex_idx] = np.where(claim_family[ex_idx] == 'institutional', rng.integers(12, 35, len(ex_idx)), rng.integers(8, 20, len(ex_idx))).astype(np.int16)
            allowed[ex_idx] = base_allowed[ex_idx] * units[ex_idx] * rng.uniform(1.0, 1.35, len(ex_idx))
            billed[ex_idx] = allowed[ex_idx] * rng.uniform(1.2, 1.7, len(ex_idx))
            paid[ex_idx] = allowed[ex_idx] * rng.uniform(0.8, 1.0, len(ex_idx))

        # unbundling: many small component lines with same member/provider/date
        unb_idx = f_idx[fraud_modes == 'unbundling']
        if len(unb_idx):
            billed[unb_idx] *= rng.uniform(1.2, 1.6, len(unb_idx))
            paid[unb_idx] *= rng.uniform(1.05, 1.25, len(unb_idx))
            if np.any(claim_family[unb_idx] == 'institutional'):
                revenue_code[unb_idx[claim_family[unb_idx] == 'institutional']] = '0300'
                category[unb_idx[claim_family[unb_idx] == 'institutional']] = 'Lab'
            if np.any(claim_family[unb_idx] == 'professional'):
                procedure_code[unb_idx[claim_family[unb_idx] == 'professional']] = '36415'
                category[unb_idx[claim_family[unb_idx] == 'professional']] = 'Lab'

        # high-cost facility inflation
        fac_idx = f_idx[fraud_modes == 'high_cost_facility']
        if len(fac_idx):
            revenue_code[fac_idx] = np.where(claim_family[fac_idx] == 'institutional', revenue_code[fac_idx], revenue_code[fac_idx])
            allowed[fac_idx] *= rng.uniform(1.6, 3.2, len(fac_idx))
            billed[fac_idx] *= rng.uniform(1.8, 3.4, len(fac_idx))
            paid[fac_idx] *= rng.uniform(1.3, 2.4, len(fac_idx))

    claim_ids = np.arange(start_claim_id, start_claim_id + n_claims, dtype=np.int64)
    df = pd.DataFrame({
        'claim_id': claim_ids,
        'member_id': member_ids.astype(np.int32),
        'provider_id': provider_ids.astype(np.int32),
        'service_date': dates,
        'claim_family': claim_family,
        'specialty': provider_slice['specialty'].to_numpy(dtype=object),
        'provider_type': provider_slice['provider_type'].to_numpy(dtype=object),
        'provider_class': provider_slice['provider_class'].to_numpy(dtype=object),
        'provider_fraud_propensity': provider_fraud_prop,
        'place_of_service': pos,
        'telehealth_flag': telehealth,
        'inpatient_flag': inpatient,
        'emergency_flag': emergency,
        'diagnosis_code': dx,
        'revenue_code': revenue_code,
        'procedure_code': procedure_code,
        'category': category,
        'units': units,
        'billed_amount': np.round(billed, 2),
        'allowed_amount': np.round(allowed, 2),
        'paid_amount': np.round(paid, 2),
        'member_age': member_slice['member_age'].to_numpy(dtype=np.int16),
        'member_gender': member_slice['member_gender'].to_numpy(dtype=object),
        'chronic_score': member_slice['chronic_score'].to_numpy(dtype=np.int8),
        'high_risk_member': member_slice['high_risk_member'].to_numpy(dtype=np.int8),
        'fraud_label': fraudulent.astype(np.int8),
        'fraud_type': fraud_type,
        'chunk_id': np.full(n_claims, chunk_id, dtype=np.int16),
    })

    # Create true duplicate replay rows from some duplicate fraud cases.
    dup_seed = df.index[df['fraud_type'] == 'duplicate'].to_numpy()
    if len(dup_seed):
        replay_n = max(1, int(len(dup_seed) * 0.7))
        replay_idx = rng.choice(dup_seed, size=replay_n, replace=False)
        replay = df.loc[replay_idx].copy()
        replay['claim_id'] = np.arange(df['claim_id'].max() + 1, df['claim_id'].max() + 1 + replay_n, dtype=np.int64)
        replay['fraud_label'] = 1
        replay['fraud_type'] = 'duplicate'
        df = pd.concat([df, replay], ignore_index=True)

    # Memory-friendly dtypes
    cat_cols = ['claim_family', 'specialty', 'provider_type', 'provider_class', 'place_of_service',
                'diagnosis_code', 'revenue_code', 'procedure_code', 'category', 'member_gender', 'fraud_type']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    return df


def add_detection_features(df: pd.DataFrame) -> pd.DataFrame:
    # duplicate signature count
    dup_cols = ['member_id', 'provider_id', 'service_date', 'diagnosis_code', 'procedure_code', 'revenue_code', 'allowed_amount']
    df['duplicate_signature_count'] = df.groupby(dup_cols, observed=True)['claim_id'].transform('size').astype(np.int16)

    # provider/day volume
    df['provider_day_claim_count'] = df.groupby(['provider_id', 'service_date'], observed=True)['claim_id'].transform('size').astype(np.int16)

    # provider averages and code-level averages
    df['provider_avg_allowed'] = df.groupby('provider_id', observed=True)['allowed_amount'].transform('mean').astype(np.float32)
    code_key = np.where(df['claim_family'].astype(str).to_numpy() == 'institutional',
                        df['revenue_code'].astype(str).to_numpy(),
                        df['procedure_code'].astype(str).to_numpy())
    df['code_key'] = pd.Categorical(code_key)
    df['code_avg_allowed'] = df.groupby('code_key', observed=True)['allowed_amount'].transform('mean').astype(np.float32)
    df['allowed_to_code_avg_ratio'] = (df['allowed_amount'] / np.maximum(df['code_avg_allowed'], 1)).astype(np.float32)
    df['billed_to_allowed_ratio'] = (df['billed_amount'] / np.maximum(df['allowed_amount'], 1)).astype(np.float32)

    # Provider fraud profile features
    df['provider_fraud_rate_proxy'] = df.groupby('provider_id', observed=True)['fraud_label'].transform('mean').astype(np.float32)

    return df


def apply_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    df['rule_duplicate'] = (df['duplicate_signature_count'] > 1).astype(np.int8)
    df['rule_excessive_units'] = (
        ((df['claim_family'].astype(str) == 'professional') & (df['units'] >= 8)) |
        ((df['claim_family'].astype(str) == 'institutional') & (df['units'] >= 12))
    ).astype(np.int8)
    df['rule_provider_volume'] = (df['provider_day_claim_count'] >= 20).astype(np.int8)
    df['rule_high_amount_ratio'] = (df['allowed_to_code_avg_ratio'] >= 2.0).astype(np.int8)
    df['rule_high_billed_allowed'] = (df['billed_to_allowed_ratio'] >= 1.55).astype(np.int8)
    df['rule_suspicious_or'] = (
        (df['revenue_code'].astype(str) == '0360') & (df['allowed_to_code_avg_ratio'] >= 1.8)
    ).astype(np.int8)
    df['rule_suspicious_ed'] = (
        (df['revenue_code'].astype(str) == '0450') & (df['allowed_to_code_avg_ratio'] >= 1.7)
    ).astype(np.int8)

    rule_cols = [
        'rule_duplicate', 'rule_excessive_units', 'rule_provider_volume',
        'rule_high_amount_ratio', 'rule_high_billed_allowed', 'rule_suspicious_or', 'rule_suspicious_ed'
    ]
    df['rule_score'] = df[rule_cols].sum(axis=1).astype(np.int16)
    df['rule_pred'] = (df['rule_score'] >= 2).astype(np.int8)

    # instructive label for paper / output review
    primary_reason = np.full(len(df), 'no_rule', dtype=object)
    conditions = [
        df['rule_duplicate'].to_numpy() == 1,
        df['rule_excessive_units'].to_numpy() == 1,
        df['rule_suspicious_or'].to_numpy() == 1,
        df['rule_suspicious_ed'].to_numpy() == 1,
        df['rule_high_amount_ratio'].to_numpy() == 1,
        df['rule_provider_volume'].to_numpy() == 1,
        df['rule_high_billed_allowed'].to_numpy() == 1,
    ]
    reasons = [
        'duplicate_signature', 'excessive_units', 'high_or_cost', 'high_ed_cost',
        'high_code_cost_ratio', 'high_provider_day_volume', 'high_billed_allowed_ratio'
    ]
    for cond, reason in zip(conditions, reasons):
        primary_reason[(primary_reason == 'no_rule') & cond] = reason
    df['rule_primary_reason'] = pd.Categorical(primary_reason)
    return df


def summarize_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    fraud_breakdown = (
        df.groupby(['fraud_type', 'fraud_label'], observed=True)
          .size()
          .reset_index(name='count')
          .sort_values(['fraud_label', 'count'], ascending=[False, False])
    )

    provider_summary = (
        df.groupby(['provider_id', 'specialty', 'provider_class'], observed=True)
          .agg(
              total_claims=('claim_id', 'size'),
              avg_allowed=('allowed_amount', 'mean'),
              observed_fraud_rate=('fraud_label', 'mean'),
              max_day_volume=('provider_day_claim_count', 'max'),
              avg_rule_score=('rule_score', 'mean')
          )
          .reset_index()
          .sort_values(['observed_fraud_rate', 'avg_rule_score', 'avg_allowed'], ascending=False)
    )

    rule_reason_summary = (
        df.groupby(['rule_primary_reason', 'fraud_label'], observed=True)
          .size()
          .reset_index(name='count')
          .sort_values('count', ascending=False)
    )

    detection_by_fraud_type = (
        df.groupby('fraud_type', observed=True)
          .agg(
              n=('claim_id', 'size'),
              caught_by_rules=('rule_pred', 'sum'),
              avg_rule_score=('rule_score', 'mean')
          )
          .reset_index()
    )
    detection_by_fraud_type['rule_recall_within_type'] = np.where(
        detection_by_fraud_type['n'] > 0,
        detection_by_fraud_type['caught_by_rules'] / detection_by_fraud_type['n'],
        0.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fraud_breakdown.to_csv(output_dir / 'fraud_breakdown.csv', index=False)
    provider_summary.to_csv(output_dir / 'provider_summary.csv', index=False)
    rule_reason_summary.to_csv(output_dir / 'rule_reason_summary.csv', index=False)
    detection_by_fraud_type.to_csv(output_dir / 'detection_by_fraud_type.csv', index=False)

    # simple overall metrics for rules only, scalable and light
    tp = int(((df['fraud_label'] == 1) & (df['rule_pred'] == 1)).sum())
    fp = int(((df['fraud_label'] == 0) & (df['rule_pred'] == 1)).sum())
    fn = int(((df['fraud_label'] == 1) & (df['rule_pred'] == 0)).sum())
    tn = int(((df['fraud_label'] == 0) & (df['rule_pred'] == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics = pd.DataFrame([{
        'model': 'Rule-based baseline',
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fraud_rate': float(df['fraud_label'].mean()),
        'flag_rate': float(df['rule_pred'].mean()),
    }])
    metrics.to_csv(output_dir / 'rule_metrics.csv', index=False)


def run_generation(
    n_claims: int = 1_000_000,
    n_members: int = 120_000,
    n_providers: int = 6_000,
    chunk_size: int = 250_000,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_parquet: bool = True,
    write_csv_sample: bool = True,
    smoke_test: bool = False,
) -> None:
    rng = np.random.default_rng(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    members = make_members(n_members, rng)
    providers = make_providers(n_providers, rng)
    members.to_csv(output_dir / 'members.csv', index=False)
    providers.to_csv(output_dir / 'providers.csv', index=False)

    all_chunks: list[pd.DataFrame] = []
    next_claim_id = 1
    n_chunks = (n_claims + chunk_size - 1) // chunk_size

    for chunk_id in range(n_chunks):
        n_this = min(chunk_size, n_claims - chunk_id * chunk_size)
        chunk_rng = np.random.default_rng(SEED + chunk_id + 1)
        chunk = generate_chunk(
            chunk_id=chunk_id,
            start_claim_id=next_claim_id,
            n_claims=n_this,
            members=members,
            providers=providers,
            rng=chunk_rng,
        )
        next_claim_id = int(chunk['claim_id'].max()) + 1

        if write_parquet:
            chunk.to_pickle(output_dir / f'claims_chunk_{chunk_id:03d}.pkl')

        all_chunks.append(chunk)

        if smoke_test:
            break

    claims = pd.concat(all_chunks, ignore_index=True)
    claims = add_detection_features(claims)
    claims = apply_rule_flags(claims)

    claims.to_pickle(output_dir / 'synthetic_claims_full.pkl')
    if write_csv_sample:
        claims.sample(min(100_000, len(claims)), random_state=SEED).to_csv(output_dir / 'synthetic_claims_sample_100k.csv', index=False)

    summarize_outputs(claims, output_dir)

    print(f'rows={len(claims):,}')
    print(f'fraud_rate={claims["fraud_label"].mean():.4f}')
    print(f'flag_rate={claims["rule_pred"].mean():.4f}')
    print('\nFraud type counts:')
    print(claims['fraud_type'].value_counts(dropna=False).head(10))
    print('\nTop rule reasons:')
    print(claims['rule_primary_reason'].value_counts(dropna=False).head(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate synthetic health claims with more realistic provider-level fraud patterns.')
    parser.add_argument('--n_claims', type=int, default=1_000_000)
    parser.add_argument('--n_members', type=int, default=120_000)
    parser.add_argument('--n_providers', type=int, default=6_000)
    parser.add_argument('--chunk_size', type=int, default=250_000)
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--smoke_test', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_generation(
        n_claims=args.n_claims,
        n_members=args.n_members,
        n_providers=args.n_providers,
        chunk_size=args.chunk_size,
        output_dir=Path(args.output_dir),
        smoke_test=args.smoke_test,
    )
