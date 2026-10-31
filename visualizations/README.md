# Visualizations of the core logic

The diagrams below are drawn in Mermaid, so GitHub renders them. The PNG plots in `figures/` are produced by:

```
python -m visualizations.make_figures
```

---

## 1. The whole program: the paper's Figure 1 flowchart as function calls

```mermaid
flowchart TD
    A[Params<br/>params.py] --> B[make_grid, zero_field<br/>U = L = 0 at t = 0]
    B --> C[seed_colonies<br/>1/2/3 colonies, p = .6/.3/.1]
    C --> D{{colony step loop<br/>every 0.5 h until t_obs}}
    D --> E[total_footprint<br/>sum of Gaussians, sigma = R/3]
    E --> F{{50 PDE steps<br/>tau = 0.01 h}}
    F --> G[density N t]
    G --> H[adi_step]
    H --> F
    F --> I[update_colonies]
    I --> J[record + snapshots]
    J --> D
    D --> K[Result<br/>series, snapshots, final U and L]
```

## 2. Inside one PDE step: `adi_step` → `coupled_half_step` → `thomas_solve`

```mermaid
flowchart LR
    subgraph X[half step 1: implicit in x]
      X1[L from OLD U<br/>source_terms -> implicit_x_half_step] --> X2[q_U = gamma_U + gamma_LU * L<br/>ahl_loss_rate - Q1]
      X2 --> X3[fixed-point loop:<br/>F1 from guess -> solve U -> repeat until stable]
      X3 --> X4[L again from NEW U]
    end
    subgraph Y[half step 2: implicit in y]
      Y1[same L -> U -> L order<br/>implicit_y_half_step = transpose + x solver]
    end
    X --> Y
    X1 -.each solve.-> T[thomas_solve<br/>all rows at once - Q3]
    X3 -.-> T
    Y1 -.-> T
```

## 3. Colony update: the dashed box of the paper's flowchart

```mermaid
flowchart TD
    S[sense_colonies<br/>local AHL -> activation, Hill switch] --> DNA[apply_dna_uptake<br/>readout only]
    DNA --> Q{antibiotic added?<br/>treatment_started}
    Q -- no --> SP[maybe_spawn<br/>p = 0.1]
    SP --> GR[grow_colony<br/>logistic_step with r_eff]
    GR --> LIM{reached_limit?}
    LIM -- yes --> STOP[stop growing]
    Q -- yes --> SH[shrink_radius<br/>R * 1 - theta A^2/A^2+m]
    SH --> RG{t >= regrow_time?}
    RG -- yes --> GR2[grow_colony with r_regrow]
    FB[[effective_growth_rate<br/>r * 1 + k_growth * activation<br/>only if feedback_on - Q7]] -.-> GR
    FB -.-> GR2
```

## 4. The Q7 feedback loop the paper does not have

```mermaid
flowchart LR
    U[AHL near colony] -->|Hill switch| ACT[QS activation]
    ACT -->|feedback_on| R[faster growth -> bigger R]
    R -->|sigma = R/3| FP[wider footprint]
    FP -->|F1 = N f1 U footprint| U
    ACT -.readout.-> D[DNA uptake]
    L[lactonase] -->|gamma_LU L U| U
    U -->|f2, threshold U_th + eps| L
```

In paper mode (`feedback_on = False`), the `ACT → R` arrow is cut. Bacteria still change the signal, but the signal never changes the bacteria.

---

## Figure index

| file | what it shows | step / quiz |
|---|---|---|
| `01_hill_switch_Q2.png` | f1 and f2 against AHL, and how eps delays the lactonase brake | production, **Q2** |
| `02_footprint_Q4.png` | Gaussian colony footprints: R sets where, N sets how much | production, **Q4** |
| `03_adi_vs_full_implicit_Q3.png` | Matrix shape for a fully implicit 2D step vs ADI chains | solver, **Q3** |
| `04_density_and_antibiotic.png` | N(t) and A(t) for one dose and three doses (paper Figs. 3, 5A, 8) | density, antibiotic |
| `05_colony_size.png` | Total colony radius over time (paper Figs. 5B, 9A) | colonies |
| `06_fields_snapshots.png` | AHL and lactonase maps at 5, 12 and 50 h (paper Fig. 6) | full model |
| `07_max_concentrations.png` | Max AHL and lactonase over time against the paper's values and U_th (Fig. 7) | full model |
| `08_half_lives_Q5.png` | Decay timescales: why lactonase collapses while AHL lingers | **Q5** |
| `09_positive_control_deterministic_Q6.png` | Deterministic control and the Fig. 4 perturbation test | **Q6** |
| `10_seed_variability_Q6.png` | Different seeds give different runs; paper values for reference | **Q6** |
| `11_coupling_Q1.png` | AHL with vs without lactonase destroying it | **Q1** |
| `12_feedback_Q7.png` | Feedback OFF vs ON, for the paper strain and the sensitive strain | **Q7** |
