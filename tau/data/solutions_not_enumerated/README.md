# Not-enumerated route solutions (quarantined)

These CN states were never properly enumerated. Each carries a **single sequential
("caterpillar") route** — a placeholder produced by `dev/route_generation/build_seq_routes.py`,
not the full set of gain orderings. Timing a segment through the one caterpillar route
silently asserts a gain order that was never checked against the alternatives.

They were moved out of `tau/data/solutions/` so the timing path cannot pick them up.
Segments in these states now report `no_routes_for_state`, which is the truthful answer.

(8,0) and (8,1) are NOT here: they were fully enumerated (116 routes each) and stay in
`solutions/`. To restore any state here, enumerate it properly with
`dev/route_generation/gen_matrices.py` + the Sage array pipeline, then move it back.
