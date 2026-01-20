# Experimental Constraints on Landau-Siegel Zeros

**A 2-Billion Point Spectral Gap Analysis of Q₄₇**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

This repository contains the analysis code, data, and paper for experimental constraints on hypothetical Landau-Siegel zeros derived from spectral gap analysis of the polynomial prime sequence:

$$Q(n) = n^{47} - (n-1)^{47}$$

We analyzed **15.4 million primes** across the range $n \in [3 \times 10^8, 2 \times 10^9]$ using a **two-stage verification protocol**.

## Key Results

| Diagnostic | Observed | Poisson Prediction | Status |
|------------|----------|-------------------|--------|
| Coefficient of Variation | 0.995 | 1.000 | ✓ Consistent |
| Max Gap Ratio | 0.99 | 1.00 | ✓ Consistent |
| Cramér Ratio | < 1.5 | < 2.0 | ✓ Bounded |
| Regional Anomalies | 0/100 | 0 | ✓ Uniform |

**Conclusion:** No evidence for Landau-Siegel zeros. Gap distribution fully consistent with the Generalized Riemann Hypothesis.

## Two-Stage Verification Protocol

### Stage 1: Fast Scanning
- Process 1.7 billion integers using optimized algorithms
- Flag candidate anomalies exceeding statistical thresholds
- Initial scan identified apparent gap of 5147 at $n \approx 1.4 \times 10^9$

### Stage 2: Arbitrary-Precision Verification
- Rescan flagged regions using `gmpy2` Miller-Rabin primality testing
- Verification revealed **53 fine-structure primes** in the flagged region
- Corrected spacing: 95.3 (actually *denser* than average)

## Repository Structure

```
├── paper/
│   ├── Landau_Siegel_Exclusion_v2.pdf    # Main paper (6 pages)
│   └── Landau_Siegel_Exclusion_v2.tex    # LaTeX source
├── figures/
│   └── fig_verification_results.pdf       # 4-panel verification figure
├── data/
│   └── verification_results.json          # Final statistics
├── scripts/
│   ├── gap_analysis.py                    # Large-scale gap scanning
│   └── verify_region.py                   # High-precision verification
└── README.md
```

## Usage

### Gap Analysis (Stage 1)
```bash
python scripts/gap_analysis.py
```

### Region Verification (Stage 2)
```bash
# Install gmpy2 for high-precision arithmetic
pip install gmpy2

# Verify a specific region
python scripts/verify_region.py 1399874854 1399880001
```

## Related Work

This paper is part of the **Ouroboros Prime Condensate** research series:

| Paper | DOI | Description |
|-------|-----|-------------|
| Q47 Dataset | [10.5281/zenodo.18305185](https://doi.org/10.5281/zenodo.18305185) | Complete prime dataset |
| Ouroboros Phase Transition | [10.5281/zenodo.18306984](https://doi.org/10.5281/zenodo.18306984) | GUT theoretical framework |
| Spectral Statistics | [10.5281/zenodo.18259473](https://doi.org/10.5281/zenodo.18259473) | Poisson verification |
| **This Paper** | Pending | Landau-Siegel constraints |

## Citation

```bibtex
@misc{chen2026landau,
  author       = {Chen, Ruqing},
  title        = {{Experimental Constraints on Landau-Siegel Zeros: 
                   A 2-Billion Point Spectral Gap Analysis of Q47}},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXXX},
  url          = {https://github.com/Ruqing1963/Landau-Siegel-Q47-Constraints}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

**Ruqing Chen**  
GUT Geoservice Inc., Montreal, Canada  
ruqing@hotmail.com
