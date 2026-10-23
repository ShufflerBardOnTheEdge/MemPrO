# Design: per-leaflet protein footprint (`-footprint map`)

Status: agreed design, not yet implemented (branch `lipid-area-table`, 2026-10-23).
This is our own design. Do not copy or adapt code from COBY (its approach is an alpha-shape outline per leaflet; this one is not).

## Problem

Insane4MemPrO never measures how much of each leaflet the protein occupies.

How lipids are removed now (`place_lipid` in `Insane4MemPrO.py`):
- Each leaflet's grid is filled to (box area) / (area per lipid).
- Each lipid's tails are traced down from its head site, bending around protein beads within `-fudge` (0.3 nm).
- Sites whose tails cannot pass are dropped.

So the footprint depends implicitly on the grid spacing, `-fudge` and whether the tracing happens to succeed. In addition:
- head beads are never checked against the protein;
- leaflet planes come from lipid length, not from the membrane MemPrO fitted.

`mempro-leaflets` reported the QmcA build's lower leaflet as about 9% under-filled under its soluble domain. That figure depends on the footprint definition. The cause has not been confirmed yet; confirming it is the first step.

## Design

1. **Leaflet geometry from MemPrO.**
   - MemPrO fits the midplane and thickness for each rank. It writes them to `info_rank_N.txt`, e.g. "(Inner) Membrane Thickness: 38. A" for QmcA, plus the outer thickness with `-dm`.
   - `build_oriented` in `MemPrO.py` passes them to Insane through a new option (e.g. `-mt` for inner/outer, in nm).
   - Each leaflet spans from the midplane to its phosphate plane at ±T/2.
   - With `-def`, the surfaces follow MemPrOD's local deformation (`membrane_pos`).
   - Without MemPrO, fall back to lipid length, as now.
   - Curved membranes can keep the old method at first.
2. **Depth-resolved occupancy.**
   - Slice each leaflet at 0.2 nm, from the midplane to the phosphate plane, plus a headgroup layer of about 0.5–1 nm.
   - Rasterise the protein on a periodic 0.1 nm pixel grid in each slice.
   - A pixel is blocked if it lies within r_bead(protein) + r_bead(lipid) of a protein bead in that slice.
   - Radii depend on Martini 3 bead size (regular, small, tiny).
3. **Free area by volume conservation.** This is the core idea.
   - The hydrophobic core is roughly an incompressible fluid, so lipids fill whatever core volume the protein leaves free.
   - Per pixel: available fraction = fraction of core slices that are free.
   - The fraction is set to 0 if the protein occupies the headgroup layer there (amphipathic helices, or soluble domains resting on the interface).
   - Free area of the leaflet = sum of available fraction × pixel area.
   - N_lipids = free area / mixture area per lipid (`pick_area`; site areas for multi-site lipids).
   - Compared with the largest cross-section, this does not over-count tilted or funnel-shaped proteins. Compared with a single slice, it does not miss them.
4. **Placement to an exact count.**
   - Keep the ring or square grid, but drop sites on pixels that are mostly blocked.
   - Then thin, or top up from a fine fill of free pixels, to reach N exactly.
   - Keep the existing tail tracing.
   - A lipid that fails moves to a spare free site instead of being dropped.
5. **One shared function.**
   - Implement the map in a self-contained block, as is done with the area table, so that `mempro-leaflets` (`leaflet_check.py`) uses the same definition.
   - The build log reports each leaflet's footprint, free area and N.
   - Optionally write the map (`.npz` or `.png`) for inspection.

Option: `-footprint map|clash` (default `clash`, the current behaviour), so the two methods can be compared before deciding on a default.

## Validation plan

1. **Synthetic shapes:**
   - a bead cylinder gives a footprint of πR² (including bead radii);
   - a cone or hourglass gives the predicted difference between the upper and lower leaflets.
2. **MD, footprint-independent metric.**
   - Build, energy-minimise, then a few ns of semi-isotropic NPT (Martini 3, 310 K).
   - Count the lipid density beyond 3 nm from the protein in each leaflet.
   - Target: the upper/lower densities match to within 1–2%. Compare `clash` against `map`.
3. **Test proteins:**
   - 4G1U MemPrO build (about 42k beads, roughly 10 min/ns on an M4 Mac);
   - QmcA, 234k beads, best run on a GPU machine (inputs and commands in `tests/data/QmcA/`);
   - optionally a simple TM helix bundle.
   - Protein topologies come from martinize2 (available in the CCD2MD or compstrucbio conda envs on the Mac).

## Context

- Branch `lipid-area-table` (on top of v0.3.1) already contains:
  - the area-per-lipid table;
  - `-balance`, `-grid`, `-temp`, `-apl_ff`;
  - multi-site packing;
  - `mempro-leaflets`;
  - Martini 3 lipidome support (`lipidome_m3.dat`, `-lipid_itp`);
  - charge fixes;
  - 20 tests (`python -m unittest discover tests`).
- Key code (`mempro/Insane4MemPrO.py`):
  - `place_lipid`, `near_protein` and `place_bead`, plus the "Balance the leaflets" block;
  - `pick_area`, `assign_lipids` and `create_leaflet`.
- Key code (`mempro/MemPrO.py`): `build_oriented`, and the `optimise_mem_thickness` / `mem_structure` thickness values.
