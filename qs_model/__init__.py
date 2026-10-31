"""qs_model: a simple re-implementation of the Kuttler & Maslovskaya (2022) quorum-sensing model.

Paper: "Computer-assisted modeling of Quorum sensing in bacterial population exposed to
antibiotics", Front. Appl. Math. Stat. 8:951783, doi:10.3389/fams.2022.951783

Module map (in the order data flows through one simulation step):

    params.py      every number in the model (Table 1 + colony/antibiotic/feedback settings)
    grid.py        the 2D space grid the PDEs live on
    density.py     N(t): prescribed, normalized bacterial density            (paper Eqs. 15-16)
    antibiotic.py  A(t): antibiotic pulse and how it shrinks colonies        (paper Eqs. 13-14)
    colonies.py    Monte-Carlo colony birth + logistic growth of radius R   (paper Eq. 12)
    production.py  Hill switch + Gaussian footprint -> source terms F1, F2  (paper Eqs. 5-6)
    solver.py      Thomas algorithm + Peaceman-Rachford ADI + L->U->L loop  (paper Eqs. 1-2, 7-11)
    feedback.py    Q7 extension: AHL changes colony growth (OFF in paper mode)
    simulation.py  the main loop that chains all of the above (paper Figure 1 flowchart)
    controls.py    positive controls: deterministic paper setup + statistical paper comparison

Where each quiz question lives in the code:
    Q1 (coupling term)        solver.coupled_half_step, solver.ahl_loss_rate
    Q2 (epsilon shift)        production.lactonase_production
    Q3 (why Thomas / ADI)     solver.thomas_solve, solver.implicit_x_half_step
    Q4 (how much vs where)    production.source_terms, production.colony_footprint
    Q5 (AHL lingers)          params.half_life, visualizations figure 08
    Q6 (positive control)     controls.deterministic_control, controls.statistical_control
    Q7 (QS drives behavior)   feedback.py
"""
