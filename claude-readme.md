# claude-readme: what Claude built, and why

This file summarizes Claude's (Claude Code) contributions to the 02-120 QS final project. Michaela Xie directed the work. The full conversation is in `journal/`.

## Contributions, in order

1. **Build journal setup.** Following the course instructions, Claude added `.claude/hooks/save_journal.py` and the hooks in `.claude/settings.json`.
2. **Read the proposal and the paper.** Source: Kuttler & Maslovskaya 2022, *Front. Appl. Math. Stat.* 8:951783, [doi:10.3389/fams.2022.951783](https://doi.org/10.3389/fams.2022.951783). Claude pulled out every equation and every Table 1 value.
3. **Explained the model in 10 steps**, along with its pros and cons, its scope, and what it does not explain about QS. Then Claude set a 7-question quiz, graded Michaela's answers, and wrote a coding outline.
4. **Implemented the model** in the `qs_model/` package. Every function has a comment block with five parts:
   - Input
   - Output
   - Steps
   - Types
   - Role in model (2–3 sentences)
5. **Implemented the Q7 extension** (QS feeding back on growth) in `qs_model/feedback.py`. It is off by default, so the paper model stays an untouched positive control.
6. **Built two layers of positive controls** (`qs_model/controls.py`, `run_positive_control.py`):
   - a deterministic one, which matches the paper's Fig. 4 setup;
   - a statistical one, which runs several random seeds and compares against the paper's printed numbers.
7. **Wrote unit tests** in `tests/test_model.py`: 11 tests, each checking one building block against an answer known independently.
8. **Made visualizations** in `visualizations/`: Mermaid diagrams of the call chain, and 12 figures, each tied to a model step or quiz question.

## How to run

```
python -m pip install -r requirements.txt
python -m tests.test_model            # ~10 s, unit tests
python run_positive_control.py        # ~5 min, writes results/positive_control_report.md
python run_feedback_demo.py           # ~4 min, writes results/feedback_demo.md
python -m visualizations.make_figures # ~8 min, writes visualizations/figures/*.png
```

## The model in code: steps and the functions that do them

| # | Step (the 10-step explanation) | Paper | Functions |
|---|---|---|---|
| 1 | Organism and circuit (AHL = U, lactonase = L) | Sec. 1 | `params.Params` (Table 1) |
| 2 | Reaction–diffusion PDEs | Eqs. 1–2 | `solver.ahl_loss_rate`, `solver.implicit_x_half_step` (general form du/dt = D∇²u − qu + F) |
| 3 | Where molecules are produced: Hill switch × Gaussian footprint × N(t) | Eqs. 5–6 | `production.hill`, `ahl_production`, `lactonase_production`, `colony_sigma`, `colony_footprint`, `total_footprint`, `source_terms` |
| 4 | Zero initial values and zero edges | Eqs. 3–4 | `grid.zero_field`; boundaries forced to 0 in `implicit_x_half_step` |
| 5 | ADI + Thomas + fixed-point loop | Eqs. 7–11 | `solver.thomas_solve`, `second_difference_y`, `implicit_x/y_half_step`, `coupled_half_step`, `adi_step` |
| 6 | Monte Carlo colonies, logistic growth | Eq. 12 | `colonies.draw_initial_count`, `random_position`, `new_colony`, `seed_colonies`, `maybe_spawn`, `logistic_step`, `reached_limit`, `grow_colony` |
| 7 | Antibiotic pulse and shrinkage | Eqs. 13–14 | `antibiotic.rayleigh_pulse`, `antibiotic_concentration`, `kill_fraction`, `shrink_radius`, `treatment_started` |
| 8 | Prescribed density N(t) | Eqs. 15–16 | `density.growth_phase_density`, `decline_coefficients`, `decline_phase_density`, `density` |
| 9 | Main loop (Fig. 1 flowchart) | Fig. 1 | `simulation.run_simulation` → `advance_chemistry` → `adi_step`; `update_colonies`; `record` |
| 10 | Scheme check (perturbation test) | Fig. 4 | `controls.perturbation_test` |

### How the big functions chain the small ones

```
run_simulation
 ├─ make_grid, zero_field, seed_colonies
 └─ every 0.5 h:
     ├─ total_footprint ─ colony_footprint(colony_sigma)
     ├─ advance_chemistry ─ 50 × adi_step
     │    └─ coupled_half_step (x, then y)
     │         ├─ source_terms ─ ahl_production / lactonase_production ─ hill
     │         ├─ ahl_loss_rate
     │         └─ implicit_x/y_half_step ─ second_difference_y, thomas_solve
     ├─ update_colonies
     │    ├─ sense_colonies ─ qs_activation ─ hill            (Q7)
     │    ├─ apply_dna_uptake ─ dna_uptake_increment         (Q7 readout)
     │    ├─ maybe_spawn, grow_colony ─ effective_growth_rate, logistic_step
     │    └─ shrink_radius ─ kill_fraction, antibiotic_concentration ─ rayleigh_pulse
     └─ record ─ density, antibiotic_concentration, total_radius
```

## Where Q1–Q7 live in the code

| Q | Answer (short) | Code |
|---|---|---|
| Q1 | `γ_LU·L·U` is the only place L affects U. It is nonlinear, so L is held fixed while solving U (L → U → L). | `solver.ahl_loss_rate`, `solver.coupled_half_step`; figure 11 |
| Q2 | ε shifts the lactonase threshold. A bigger ε means the brake engages later. | `production.lactonase_production`; test `test_epsilon_delays_lactonase`; figure 01 |
| Q3 | ADI turns each row into a chain in which every point touches 2 neighbours. Thomas solves a chain in ~m steps. | `solver.thomas_solve`, `implicit_x_half_step`; test `test_thomas_matches_dense_solver`; figure 03 |
| Q4 | N(t) sets how much is produced. The radius (through σ = R/3) sets where, and also adds some amount. | `production.source_terms`, `colony_footprint`; figure 02 |
| Q5 | Decay rates, not diffusion: AHL half-life ~125 h, lactonase ~1.4 h. | `params.half_life`; figure 08 |
| Q6 | Runs are random. The strict control is one fixed colony at the centre (R = 10 µm, N = 1). | `controls.deterministic_control`, `statistical_control`; figures 09, 10 |
| Q7 | Local AHL → Hill activation → growth rate × (1 + k·activation), plus a DNA-uptake readout. | `feedback.py`, `colonies.grow_colony`; figure 12 |

## Assumptions Claude had to make (the paper does not give these)

- **`m = 4`** (antibiotic half-saturation). Calibrated so that one dose shrinks total colony size about 10×, matching Fig. 5B.
- **`a`, `b` in N(t).** Chosen so the curve is continuous at the dose and drops 30%, as the paper's text describes. Decline midpoint = dose time + 6 h, from the paper's t_d = 12 and b2 = 18.
- **Eq. 2 typo.** The paper prints `D_L ΔU`; the code uses `D_L ΔL`.
- **Gaussian denominator.** The code uses `σ²`, exactly as printed (not `2σ²`).
- **Perturbation reference.** The ξ, φ formula is ambiguous about which run to compare against. Every perturbed run is compared with the unperturbed run.
- **Multiple doses.** Each dose cuts the density by a further 30%.
- **Nearest-grid-point sensing** for colony activation (Q7 only).

## Results

### Positive control (`results/positive_control_report.md`)

**Deterministic (strict) control: all correctness checks pass.**
- Edges stay at exactly 0.
- No concentration goes negative.
- Mirror symmetry error is about 10⁻¹⁶. x↔y symmetry error is about 10⁻⁹, a tiny, expected ADI splitting error.
- The fixed-point loop converges in 3 or fewer iterations.
- Halving the time step changes the answer by less than 10⁻⁴.
- The paper's Fig. 4 perturbation test also behaves as it should: changes are small and grow steadily.

**Statistical control (4 seeds, single dose): 7 of 8 checks pass.**

| quantity | ours (mean) | paper |
|---|---|---|
| antibiotic peak / delay | 4.0 at 4 h | 4.0 at 3.5–4 h |
| final max AHL | **4.88·10⁻¹⁰** | **4.89·10⁻¹⁰** |
| final max lactonase | 1.0·10⁻¹² | 6.8·10⁻¹³ |
| final total radius | 3.8 µm | 2.0 µm |
| lactonase collapses, AHL lingers | yes | yes |

**One FAIL, kept visible:** the paper says lactonase peaks *before* the dose. Ours peaks 0–1 h *after* it, because the antibiotic starts at zero and takes time to act.

### Two findings worth discussing

1. **With the paper's own parameters, quorum sensing never switches on.**
   - AHL peaks at about 1·10⁻⁸ mol/L, roughly 7× below the threshold U_th = 7·10⁻⁸ (figure 07). At the centre of the deterministic control, AHL reaches only 13% of U_th.
   - Two things hold it down: lactonase (Q1, figure 11) and the absorbing edges of a 100 µm box.
   - So the paper's claim that "the QS level remains high" after treatment refers to AHL that is already far below the switch point.
2. **Q7 feedback only matters once QS can switch on.**
   - Paper strain: feedback ON and OFF are identical.
   - Sensitive strain (U_th lowered 10×, lactonase threshold kept at the paper's value): QS activation reaches 0.94.
   - With feedback, colonies are 33% larger by the time of the dose (68 vs 51 µm), and survivors stay about 30% larger through regrowth (4.2 vs 3.2 µm at 100 h).
   - QS still never turns back on after treatment. Post-treatment AHL is ~1000× below even the sensitive threshold, so the paper's "quorum memory" does not drive regrowth in this model.

## Files Claude created

```
qs_model/            the model package (10 modules)
tests/test_model.py  11 unit tests
run_positive_control.py, run_feedback_demo.py
results/             positive-control and feedback-demo reports
visualizations/      README.md (Mermaid logic diagrams), make_figures.py, figures/*.png
claude-readme.md, requirements.txt, .gitignore
```
