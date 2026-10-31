# Positive-control report

Model: Kuttler & Maslovskaya (2022), re-implemented in `qs_model/` (paper mode, feedback OFF).

## 1. Deterministic control (one centred colony, R = 10 um, N = 1, 20 h)

| check | value | result |
|---|---|---|
| edges_zero | 0 | PASS |
| no_negative | 0 | PASS |
| mirror_symmetric | 2.66e-16 | PASS |
| xy_symmetric | 2.09e-09 | PASS |
| centre_AHL_over_threshold | 0.133 | INFO |
| picard_converged | 3 | PASS |

## 2. Time-step convergence (tau = 0.01 h vs 0.005 h, 5 h)

Relative change: AHL 3.18e-05, lactonase 8.82e-05 -> PASS (threshold 1e-2)

## 3. Perturbation test (paper Fig. 4)

| parameter | +20% | +30% | +40% | +50% | grows steadily |
|---|---|---|---|---|---|
| alpha_U xi (AHL) | 0.081 | 0.117 | 0.150 | 0.182 | PASS |
| alpha_U phi (lactonase) | 0.221 | 0.329 | 0.434 | 0.537 | |
| beta_U xi (AHL) | 0.004 | 0.006 | 0.008 | 0.010 | PASS |
| beta_U phi (lactonase) | 0.011 | 0.017 | 0.023 | 0.029 | |
| beta_L xi (AHL) | 0.042 | 0.060 | 0.077 | 0.092 | PASS |
| beta_L phi (lactonase) | 0.079 | 0.115 | 0.150 | 0.182 | |

## 4. Statistical control (single dose at 12 h, seeds 0-3, mean)

| quantity | ours | paper | result |
|---|---|---|---|
| A_peak (5%) | 4 | 4 | PASS |
| A_peak_delay_h | 4 | 3.5-4.5 | PASS |
| R_total_final (x10) | 3.79 | 2.02 | PASS |
| U_max_final (x10) | 4.88e-10 | 4.89e-10 | PASS |
| U_mean_final (x10) | 4.5e-11 | 1.36e-10 | PASS |
| L_max_final (x10) | 1.01e-12 | 6.76e-13 | PASS |
| L peaks before dose | 12.5 | < 12.0 | FAIL |
| L falls more than U | L 7.5e-05 vs U 4.4e-02 | L collapses, U lingers | PASS |
