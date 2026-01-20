#!/usr/bin/env python3
"""
HIGH-PRECISION VERIFICATION SCRIPT
===================================
Verifies Q47 primality using gmpy2 arbitrary-precision arithmetic.
Used to resolve candidate anomalies identified by fast scanning.

Author: Ruqing Chen
Date: January 2026

Requirements:
    pip install gmpy2

Usage:
    python verify_region.py <start_n> <end_n>
    
Example:
    python verify_region.py 1399874854 1399880001
"""

import sys

try:
    import gmpy2
    from gmpy2 import mpz, is_prime
    GMPY2_AVAILABLE = True
    print("Running with gmpy2 (High Precision Mode)")
except ImportError:
    GMPY2_AVAILABLE = False
    print("WARNING: gmpy2 not available. Install with: pip install gmpy2")
    print("Falling back to basic primality test (less reliable for large numbers)")


def is_prime_basic(n):
    """Basic primality test (fallback if gmpy2 unavailable)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def Q47(n):
    """Compute Q(n) = n^47 - (n-1)^47."""
    if GMPY2_AVAILABLE:
        n = mpz(n)
        return n**47 - (n-1)**47
    else:
        return n**47 - (n-1)**47


def is_Q47_prime(n):
    """Check if Q(n) is prime."""
    q = Q47(n)
    if GMPY2_AVAILABLE:
        # gmpy2.is_prime uses Miller-Rabin with 25+ rounds by default
        return is_prime(q)
    else:
        return is_prime_basic(q)


def verify_region(start_n, end_n):
    """
    Verify all n values in [start_n, end_n] for Q47 primality.
    
    Returns:
        list: All n values where Q(n) is prime
    """
    print(f"\n{'='*60}")
    print("VERIFYING REGION FOR Q47 PRIMES")
    print(f"Range: n = {start_n:,} to {end_n:,}")
    print(f"Interval width: {end_n - start_n:,}")
    print(f"{'='*60}\n")
    
    primes_found = []
    total = end_n - start_n + 1
    
    # Check boundary points
    print("Checking boundaries...")
    if is_Q47_prime(start_n):
        print(f"  START n={start_n}: Q47 is PRIME")
        primes_found.append(start_n)
    else:
        print(f"  START n={start_n}: Q47 is composite")
    
    if is_Q47_prime(end_n):
        print(f"  END n={end_n}: Q47 is PRIME")
        if end_n not in primes_found:
            primes_found.append(end_n)
    else:
        print(f"  END n={end_n}: Q47 is composite")
    
    print(f"\nScanning interior ({total-2:,} integers)...")
    
    # Scan the interior
    progress_interval = max(1, (total - 2) // 10)
    
    for i, n in enumerate(range(start_n + 1, end_n)):
        if i > 0 and i % progress_interval == 0:
            pct = 100 * i / (total - 2)
            print(f"  Progress: {pct:.0f}% (n = {n:,})")
        
        if is_Q47_prime(n):
            primes_found.append(n)
            print(f"  FOUND PRIME at n = {n:,}")
    
    # Sort results
    primes_found.sort()
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Primes found: {len(primes_found)}")
    
    if len(primes_found) > 1:
        gaps = [primes_found[i+1] - primes_found[i] for i in range(len(primes_found)-1)]
        mean_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        print(f"Mean spacing: {mean_gap:.1f}")
        print(f"Max gap: {max_gap}")
    
    if len(primes_found) > 0:
        print(f"\nPrime n values:")
        for n in primes_found[:20]:  # Show first 20
            print(f"  {n:,}")
        if len(primes_found) > 20:
            print(f"  ... and {len(primes_found) - 20} more")
    
    return primes_found


def main():
    if len(sys.argv) != 3:
        print("Usage: python verify_region.py <start_n> <end_n>")
        print("Example: python verify_region.py 1399874854 1399880001")
        sys.exit(1)
    
    start_n = int(sys.argv[1])
    end_n = int(sys.argv[2])
    
    if start_n >= end_n:
        print("Error: start_n must be less than end_n")
        sys.exit(1)
    
    primes = verify_region(start_n, end_n)
    
    # Save results
    output_file = f"verified_primes_{start_n}_{end_n}.txt"
    with open(output_file, 'w') as f:
        f.write(f"# Q47 primes in range [{start_n}, {end_n}]\n")
        f.write(f"# Total: {len(primes)}\n")
        for n in primes:
            f.write(f"{n}\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
