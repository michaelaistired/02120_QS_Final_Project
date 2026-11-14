"""qs_emergence: individual-cell (agent) model for emergent behavior in quorum sensing.

Built on the chemistry of Kuttler & Maslovskaya (2022) (doi:10.3389/fams.2022.951783), with
new ingredients the paper does not have: a nutrient field, individual cells that eat, divide,
die and move, contact networks between cells, and two hypotheses for how food reaches a
starving colony centre. Output is a video of every field and every cell over time.

Module map (data flows top to bottom each step):

    params.py   every number (paper Table 1 + nutrient / agent / network / video settings)
    grid.py     2D grid; agents <-> grid (which cell is each agent in, counts per cell)
    solver.py   Thomas + Peaceman-Rachford ADI, generalized to fixed non-zero edge values
    fields.py   source/sink terms: QS production, nutrient uptake, starvation signal
    agents.py   the cells: sense, maintain/die, divide, move (forage + crowding)
    network.py  contact network between cells: food sharing along links, subnetwork labels
    engine.py   the main loop chaining everything, with a frame callback for video
    video.py    turns frames into MP4 (render while simulating)
    paper_control.py  Stage 0 positive control: the paper's own model, animated

Independence: nothing here imports qs_model EXCEPT paper_control.py, which deliberately uses
the original implementation's colony / density / antibiotic rules as the reference.
"""
