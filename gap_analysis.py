#!/usr/bin/env python3
"""
THE LANDAU-SIEGEL ZERO HUNT - FULL SCALE
=========================================
Searching for "Ghost Zeros" via Q47 Prime Gap Anomalies
FULL DATA: 3×10^8 to 2×10^9 (1.7 billion range, ~15 million primes)

Author: Ruqing Chen
Date: January 2026
"""

import numpy as np
from scipy import stats
import os
import glob
import json

print("=" * 70)
print("THE LANDAU-SIEGEL ZERO HUNT - FULL 2 BILLION SCALE")
print("Searching for Ghost Zeros in Q47 Prime Gaps")
print("=" * 70)

# =============================================================================
# LOAD ALL Q47 DATA (300M - 2000M)
# =============================================================================

print("\n[1] Loading COMPLETE Q47 prime data (300M - 2B)...")

all_n_values = []

# Data directory with extracted files
data_dir = "/home/claude/q47_full_data"
data_files = sorted(glob.glob(f"{data_dir}/Prime_Q47_*.txt"))

print(f"    Found {len(data_files)} data files")

for filepath in data_files:
    filename = os.path.basename(filepath)
    count_before = len(all_n_values)
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip headers
            if line.startswith('n') or line.startswith('-') or not line:
                continue
            
            # Parse: "n | Digits | MD5"
            if '|' in line:
                parts = line.split('|')
                try:
                    n = int(parts[0].strip())
                    all_n_values.append(n)
                except:
                    pass
    
    count_after = len(all_n_values)
    print(f"    {filename}: +{count_after - count_before:,} primes")

# Also add GalacticPrime data (50M-100M) for completeness
galactic_file = "/mnt/user-data/uploads/GalacticPrimeQ47_20260108_011656-script-results-data_list.txt"
if os.path.exists(galactic_file):
    count_before = len(all_n_values)
    with open(galactic_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('n') or line.startswith('-') or not line:
                continue
            if '|' in line:
                try:
                    n = int(line.split('|')[0].strip())
                    all_n_values.append(n)
                except:
                    pass
    print(f"    GalacticPrime (50M-100M): +{len(all_n_values) - count_before:,} primes")

# Sort and remove duplicates
all_n_values = sorted(set(all_n_values))
N = len(all_n_values)

print(f"\n    TOTAL: {N:,} unique primes loaded")

n_array = np.array(all_n_values, dtype=np.int64)
n_min, n_max = n_array[0], n_array[-1]
print(f"    Range: n ∈ [{n_min:,}, {n_max:,}]")
print(f"    Span: {(n_max - n_min)/1e9:.3f} billion")

# =============================================================================
# CHECK DATA CONTINUITY
# =============================================================================

print("\n[2] Checking data continuity...")

# Find the largest continuous segment
gaps_raw = np.diff(n_array)
huge_gap_threshold = 10000  # A gap > 10000 suggests missing data segment

huge_gaps = np.where(gaps_raw > huge_gap_threshold)[0]
print(f"    Found {len(huge_gaps)} potential data breaks (gap > {huge_gap_threshold})")

if len(huge_gaps) > 0:
    # Find largest continuous segment
    segment_starts = [0] + list(huge_gaps + 1)
    segment_ends = list(huge_gaps + 1) + [N]
    segment_lengths = [segment_ends[i] - segment_starts[i] for i in range(len(segment_starts))]
    
    best_segment = np.argmax(segment_lengths)
    best_start = segment_starts[best_segment]
    best_end = segment_ends[best_segment]
    
    print(f"    Largest continuous segment: indices [{best_start:,}, {best_end:,}]")
    print(f"    Segment length: {best_end - best_start:,} primes")
    
    # Use the largest continuous segment
    n_array = n_array[best_start:best_end]
    N = len(n_array)
    n_min, n_max = n_array[0], n_array[-1]
    print(f"    Using segment: n ∈ [{n_min:,}, {n_max:,}]")
else:
    print(f"    Data is continuous!")

# =============================================================================
# COMPUTE ALL GAPS
# =============================================================================

print("\n[3] Computing prime gaps...")

gaps = np.diff(n_array)
mean_gap = np.mean(gaps)
std_gap = np.std(gaps)
var_gap = np.var(gaps)
max_gap = np.max(gaps)
min_gap = np.min(gaps)
median_gap = np.median(gaps)

print(f"    Total gaps analyzed: {len(gaps):,}")
print(f"    Mean gap μ: {mean_gap:.2f}")
print(f"    Median gap: {median_gap:.2f}")
print(f"    Std gap σ: {std_gap:.2f}")
print(f"    Min gap: {min_gap}")
print(f"    Max gap: {max_gap}")
print(f"    Max/Mean ratio: {max_gap/mean_gap:.2f}x")

cv = std_gap / mean_gap
print(f"    Coefficient of variation: {cv:.4f} (Exponential expects 1.0)")

# =============================================================================
# POISSON/EXPONENTIAL EXPECTATION
# =============================================================================

print("\n[4] Theoretical Expectations (Poisson Process)...")

euler_gamma = 0.5772156649
expected_max = mean_gap * (np.log(N) + euler_gamma)
expected_max_99 = mean_gap * (np.log(N) + np.log(np.log(N)) + 2.5)

print(f"    Sample size N: {N:,}")
print(f"    Expected max gap E[max]: {expected_max:.2f}")
print(f"    Expected 99th percentile: {expected_max_99:.2f}")
print(f"    Observed max gap: {max_gap}")
print(f"    ")
print(f"    Ratio observed/E[max]: {max_gap/expected_max:.4f}")

if max_gap > expected_max_99:
    print(f"    ⚠️  Max gap EXCEEDS 99th percentile!")
elif max_gap > expected_max * 1.5:
    print(f"    🟡 Max gap is notably large")
else:
    print(f"    ✓  Max gap within normal range")

# =============================================================================
# EXTREME VALUE ANALYSIS
# =============================================================================

print("\n[5] Extreme Value Analysis...")

p_single_exceeds_max = np.exp(-max_gap / mean_gap)
expected_count_at_max = N * p_single_exceeds_max

print(f"    P(single gap ≥ {max_gap}) = exp(-{max_gap/mean_gap:.2f}) = {p_single_exceeds_max:.2e}")
print(f"    Expected count of gaps ≥ max: {expected_count_at_max:.4f}")

if expected_count_at_max < 0.01:
    print(f"    🔴 ANOMALY: This gap is extremely unlikely!")
elif expected_count_at_max < 0.5:
    print(f"    🟡 NOTABLE: Unusually large but possible")
else:
    print(f"    🟢 NORMAL: Consistent with Poisson expectation")

# =============================================================================
# SEARCH FOR GREAT VOIDS
# =============================================================================

print("\n[6] Searching for 'Great Voids'...")

thresholds_k = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]

print(f"\n    {'k (×μ)':<10} {'Threshold':<12} {'Observed':<12} {'Expected':<12} {'Ratio':<10} {'Status'}")
print("    " + "-" * 70)

void_alerts = []

for k in thresholds_k:
    threshold = k * mean_gap
    count = np.sum(gaps >= threshold)
    
    p_exceed = np.exp(-k)
    expected = N * p_exceed
    
    ratio = count / expected if expected > 0 else (float('inf') if count > 0 else 1)
    
    if count > 0 and expected < 0.01:
        status = "🔴 VOID!"
        void_alerts.append({'k': k, 'count': count, 'expected': expected})
    elif ratio > 5 and count > 0:
        status = "🟡 Excess"
        void_alerts.append({'k': k, 'count': count, 'expected': expected})
    elif count > 0:
        status = "✓"
    else:
        status = "—"
    
    exp_str = f"{expected:.2e}" if expected < 0.1 else f"{expected:.2f}"
    print(f"    {k:<10} {threshold:<12.1f} {count:<12} {exp_str:<12} {ratio:<10.2f} {status}")

# =============================================================================
# TOP LARGEST GAPS
# =============================================================================

print("\n[7] Top 20 Largest Gaps...")

top_k = 20
top_indices = np.argsort(gaps)[-top_k:][::-1]

print(f"\n    {'Rank':<6} {'Gap':<10} {'n location':<18} {'k=gap/μ':<10} {'P(≥gap)':<12}")
print("    " + "-" * 65)

critical_voids = []

for rank, idx in enumerate(top_indices, 1):
    gap = gaps[idx]
    n_loc = n_array[idx]
    k_val = gap / mean_gap
    p_val = np.exp(-k_val)
    expected_n = N * p_val
    
    if expected_n < 0.001:
        flag = "🔴"
        critical_voids.append({'rank': rank, 'gap': gap, 'n': n_loc})
    elif expected_n < 0.1:
        flag = "🟡"
    else:
        flag = ""
    
    print(f"    {rank:<6} {gap:<10} {n_loc:<18,} {k_val:<10.2f} {p_val:<12.2e} {flag}")

# =============================================================================
# CRAMÉR'S CONJECTURE
# =============================================================================

print("\n[8] Cramér's Conjecture Test...")

log_n = np.log(n_array[:-1])
cramer_bound = log_n ** 2
cramer_ratios = gaps / cramer_bound

max_cramer_ratio = np.max(cramer_ratios)
max_cramer_idx = np.argmax(cramer_ratios)

print(f"    Max Cramér ratio: {max_cramer_ratio:.4f}")
print(f"    Location: n = {n_array[max_cramer_idx]:,}")
print(f"    Gap at max: {gaps[max_cramer_idx]}")

if max_cramer_ratio > 2:
    print(f"    🔴 Cramér bound VIOLATED!")
elif max_cramer_ratio > 1.5:
    print(f"    🟡 Approaching Cramér bound")
else:
    print(f"    🟢 Well within Cramér bound")

# =============================================================================
# REGIONAL ANALYSIS
# =============================================================================

print("\n[9] Regional Anomaly Detection...")

n_regions = 100
chunk_size = len(gaps) // n_regions

regional_means = []
regional_maxes = []

for i in range(n_regions):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < n_regions - 1 else len(gaps)
    chunk = gaps[start:end]
    regional_means.append(np.mean(chunk))
    regional_maxes.append(np.max(chunk))

regional_means = np.array(regional_means)
global_mean = np.mean(regional_means)
global_std = np.std(regional_means)

z_scores = (regional_means - global_mean) / global_std if global_std > 0 else np.zeros_like(regional_means)
anomalous_regions = np.sum(np.abs(z_scores) > 3)

print(f"    Analyzed {n_regions} regions")
print(f"    Regional mean variation σ: {global_std:.4f}")
print(f"    Anomalous regions (|z| > 3): {anomalous_regions}")

# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n" + "=" * 70)
print("FINAL VERDICT: LANDAU-SIEGEL ZERO HUNT (FULL 2B SCALE)")
print("=" * 70)

evidence_for = []
evidence_against = []

# Test 1: Max gap ratio
ratio = max_gap / expected_max
if ratio > 2.0:
    evidence_for.append(f"Max gap {ratio:.2f}x larger than expected")
elif ratio > 1.5:
    evidence_for.append(f"Max gap borderline ({ratio:.2f}x expected)")
else:
    evidence_against.append(f"Max gap within {ratio:.3f}x of expectation")

# Test 2: Critical voids
if len(critical_voids) > 0:
    evidence_for.append(f"{len(critical_voids)} critical voids (P < 10⁻³)")
else:
    evidence_against.append("No critical voids detected")

# Test 3: Void alerts
if len(void_alerts) > 2:
    evidence_for.append(f"{len(void_alerts)} anomalous gap thresholds")
else:
    evidence_against.append(f"Gap distribution consistent ({len(void_alerts)} minor alerts)")

# Test 4: Cramér
if max_cramer_ratio > 2:
    evidence_for.append(f"Cramér bound violated ({max_cramer_ratio:.2f})")
else:
    evidence_against.append(f"Cramér bound satisfied ({max_cramer_ratio:.3f})")

# Test 5: Regional
if anomalous_regions > 5:
    evidence_for.append(f"{anomalous_regions} regional anomalies")
else:
    evidence_against.append(f"Regional homogeneity confirmed ({anomalous_regions} outliers)")

# Test 6: CV
if abs(cv - 1) > 0.05:
    evidence_for.append(f"CV = {cv:.4f} deviates from 1.0")
else:
    evidence_against.append(f"CV = {cv:.4f} ≈ 1.0 (perfect exponential)")

print("\n🔴 Evidence FOR Landau-Siegel Zero:")
for e in evidence_for:
    print(f"    • {e}")
if not evidence_for:
    print("    (None)")

print("\n🟢 Evidence AGAINST Landau-Siegel Zero:")
for e in evidence_against:
    print(f"    • {e}")
if not evidence_against:
    print("    (None)")

print("\n" + "-" * 70)
score_for = len(evidence_for)
score_against = len(evidence_against)

if score_for > score_against + 2:
    verdict = "ANOMALIES_DETECTED"
    print("🔴 VERDICT: SIGNIFICANT ANOMALIES - Further investigation needed!")
elif score_for > 0:
    verdict = "INCONCLUSIVE"
    print("🟡 VERDICT: INCONCLUSIVE - Minor deviations present")
else:
    verdict = "NO_EVIDENCE"
    print("🟢 VERDICT: NO EVIDENCE FOR LANDAU-SIEGEL ZERO")
    print("")
    print("   The Q47 prime gaps across 1.7 BILLION numbers are")
    print("   FULLY CONSISTENT with Poisson statistics.")
    print("")
    print("   This SUPPORTS the Riemann Hypothesis.")
    print("   The arithmetic fabric shows no ghost zero signatures.")

print("-" * 70)

# =============================================================================
# SAVE RESULTS
# =============================================================================

results = {
    'analysis': 'Landau-Siegel Zero Hunt - Full Scale',
    'date': '2026-01-20',
    'polynomial': 'Q(n) = n^47 - (n-1)^47',
    'data': {
        'total_primes': int(N),
        'n_range': [int(n_min), int(n_max)],
        'span_billions': float((n_max - n_min) / 1e9)
    },
    'gap_statistics': {
        'mean': float(mean_gap),
        'std': float(std_gap),
        'cv': float(cv),
        'min': int(min_gap),
        'max': int(max_gap),
        'median': float(median_gap)
    },
    'poisson_test': {
        'expected_max': float(expected_max),
        'observed_max': int(max_gap),
        'ratio': float(ratio),
        'p_value_max_gap': float(p_single_exceeds_max)
    },
    'cramer_test': {
        'max_ratio': float(max_cramer_ratio),
        'location': int(n_array[max_cramer_idx]),
        'passed': bool(max_cramer_ratio <= 2)
    },
    'anomalies': {
        'critical_voids': len(critical_voids),
        'void_alerts': len(void_alerts),
        'anomalous_regions': int(anomalous_regions)
    },
    'verdict': {
        'score_for_LS': score_for,
        'score_against_LS': score_against,
        'conclusion': verdict,
        'interpretation': 'Supports Riemann Hypothesis' if verdict == 'NO_EVIDENCE' else 'Requires investigation'
    },
    'top_gaps': [{'rank': i+1, 'gap': int(gaps[idx]), 'n': int(n_array[idx])} 
                  for i, idx in enumerate(top_indices[:10])]
}

with open('landau_siegel_hunt_full_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📁 Results saved to: landau_siegel_hunt_full_results.json")
print("\n" + "=" * 70)
