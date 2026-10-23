# Design: per-leaflet protein footprint (`-footprint map`)

Status: implemented as `-footprint map` (branch `lipid-area-table`, 2026-10-23); the MD validation (step 2 of the plan) is still to do. See [Implementation](#implementation) for the QmcA diagnosis and the choices made where the design left them open.
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

## Implementation

### Why QmcA's lower leaflet was under-filled (confirmed)

The cause is the cavity fill, not tail tracing.
- Without `-ring`, Insane takes 25 slices from z = -4 to +4 nm (0.5 nm thick, every 0.33 nm), projects the protein onto a 20 x 20 grid over its extent, and fills every cell not reachable from one corner with pseudo-beads (`protein_lip`). Lipids are then traced past these as well as the protein.
- For QmcA (a ring of TM helices at r = 10-12 nm, z = -2.5 to +1.3 nm, under a soluble cap hanging from z = -2.5 nm) the grid cells are 1.25 nm, so the ring reads as closed in every slice from z = -4 to +1 nm and about 545 nm^2 is filled per slice: the ring interior plus three corners of the bounding box (the flood fill needs a corridor two cells wide to get round the ring, and the 1 nm margin is less than that).
- The lower head plane (z = -2 nm) lies inside the filled z range, and fill slices are 0.33 nm apart against a 0.3 nm clash tolerance, so head sites over the fill rarely find a free start. The upper head plane (+2 nm) is above the last filled slice.
- Tail tracing alone hardly removes anything: with `-ring` (no fill) only 0 and 11 of 1750 sites were dropped in the upper and lower leaflets, even over the ring wall.

| QmcA build (1750 grid sites per leaflet) | lipids fewer than 1750, upper | lower |
|---|---|---|
| default (`clash`) | 36 | 276 (275 of them next to fill beads; 172 with no free start) |
| `-ring` (no fill) | 0 | 11 |
| fill slices below z = -1.7 nm removed | 32 | 53 (45 at the ring wall) |
| `-footprint map -mt 3.8` | 33 (footprint 19.8 nm^2) | 202 (footprint 119.5 nm^2) |

With `map` the lipid density inside and outside the ring is the same in both leaflets (~1.75 heads/nm^2); only the annulus under the ring wall and cap rim is depleted. `mempro-leaflets` on the default build reports -5.7% differential strain with the new footprint (-8.5% with the old cross-section), and -0.1% on a `map` build with the same leaflet geometry. The lower leaflet's larger footprint is real: the cap rim (r = 10-11.7 nm) rests in its headgroup layer.

### Choices made in the implementation

- **Code.** The `# BEGIN FOOTPRINT` block in `Insane4MemPrO.py` (`footprint_maps`, `fp_select_sites`, bead radii), executed by `leaflet_check.py` as the area table is. Tests: `tests/test_footprint.py` (synthetic shapes, site selection, builds; QmcA with `MEMPRO_SLOW_TESTS=1`).
- **Blocked versus covered.** A pixel is blocked for lipid bead centres within r_protein + r_lipid, as designed. The free area counts the pixels lipid beads centred on unblocked pixels can cover, i.e. the blocked region shrunk back by r_lipid. Counting blocked pixels directly would also count the shell of lipid radius around the protein, which annular lipids fill, as protein. Crevices narrower than a lipid bead count as protein, so bead cylinders come out 2-4% above pi (R + r_bead)^2.
- **Headgroup layer.** 0.6 nm beyond the phosphate plane (3 slices). Where a protein widens into the headgroup layer (a cone or hourglass crossing it) that layer sets the footprint, as designed; volume averaging shows where the protein is narrower at the headgroups than in the core (bulge, funnel).
- **Enclosed regions.** Pixels that are not in the largest connected (periodic) free region of any core slice of either leaflet count as protein, unless `-ring`. A closed barrel is excluded; the QmcA ring is open in its upper slices and between helices, so its interior gets lipids.
- **Bead radii.** Martini 3 regular/small/tiny (0.264/0.230/0.191 nm) from residue and bead names (approximate for backbones, whose size varies with secondary structure); Go-model CA virtual sites and MemPrO dummies are ignored; atomistic input uses 0.17 nm per heavy atom. Lipid beads are regular.
- **Geometry.** `-mt` gives the phosphate-to-phosphate thickness (MemPrO's reported thickness: hydrophobic thickness + 10 A). `build_oriented` passes it, matching membranes by position: Insane's `-l`/`-u` membrane (at -ps) is MemPrO's outer membrane unless `-flip`. Without `-mt` each leaflet extends to its lipids' PO beads as built (+/-1.65 nm for POPE). With `-def` the midplane and thickness follow MemPrOD per pixel. Lipids are still built with heads at +/-2 nm either way.
- **Placement.** Grid sites with placement fraction < 0.5 are dropped, the least available removed first if there are too many, and missing sites added one at a time in the largest gap between free pixels (0.2 nm lattice). Spares (5%, at least 20) take lipids whose tails fail; tails are traced past the protein only, without fill beads.
- **Scope.** Flat membranes filling the box, including `-ps` and `-def`. Curved membranes, pores, micelles and `-radius` disks fall back to `clash`. The default stays `clash` until the MD comparison is done.

