# MemPrO
Membrane Protein Orientation in Lipid Bilayers. The paper associated with this code can be found [here](https://doi.org/10.1021/acs.jctc.5c01433).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pstansfeld/MemPrO/blob/main/MemPrO_Colab.ipynb)

Code and parameters from the following were used to help write this code:

**Insane:**
Tsjerk A. Wassenaar et al. "Computational lipidomics with insane: A versatile tool for generating custom membranes for molecular simulations". In: Journal of Chemical Theory and Computation 11 (5 May 2015), pp. 2144–2155. issn: 15499626. [GitHub](https://github.com/Tsjerk/Insane) and [paper](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.5b00209)

**Martini forcefield:**
Siewert J. Marrink et al. "The MARTINI Force Field: Coarse Grained Model for Biomolecular Simulations". In: (2007). [Paper](https://pubs.acs.org/doi/10.1021/jp071097f)

**Peptidoglycan parameters:**
Rakesh Vaiwala et al. "Developing a Coarse-Grained Model for Bacterial Cell Walls: Evaluating Mechanical Properties and Free Energy Barriers". In: Journal of Chemical Theory and Computation 16 (8 Aug. 2020), pp. 5369–5384. issn: 15499626. [Paper](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00539)

## Installation

### pip (recommended)

```
pip install mempro
```

For NVIDIA GPU support (Linux with CUDA 12 drivers), install the `gpu` extra instead:

```
pip install "mempro[gpu]"
```

MemPrO uses a GPU automatically when JAX can see one, and the CPU otherwise; use `-dev cpu` or `-dev gpu` to choose explicitly.

### From source

Clone the GitHub repository. Python 3.11 or later is required, along with the following packages:

* JAX 0.6.0 or later (CPU), tested up to 0.11.2. Recent versions are fastest; JAX 0.11 needs Python 3.12 or later:
```
pip install "jax[cpu]>=0.6.0,<0.12"
```
* Matplotlib 3.8.4 or later:
```
pip install "matplotlib>=3.8.4"
```

The Martini 3 force field file (`martini_v3.itp`) is included in `mempro/`. To use a different one, supply its path with the `-itp` flag; the standard file can be downloaded from [here](https://cgmartini.nl/docs/downloads/force-field-parameters/martini3/particle-definitions.html).

### Google Colab

MemPrO can also be run without installing anything locally, using the [MemPrO Colab notebook](https://colab.research.google.com/github/pstansfeld/MemPrO/blob/main/MemPrO_Colab.ipynb). It installs MemPrO from PyPI, orients a structure from the PDB, AlphaFold DB or your own upload, shows the results (including a 3D view of the top-ranked orientation), and can optionally predict membrane deformations with MemPrOD. It runs on both CPU and GPU runtimes.

## Running MemPrO

### From source

```
python PATH/TO/MemPrO.py -f input_file.pdb
```

### Installed via pip

```
mempro -f input_file.pdb
```

No environment variables are required. By default MemPrO will use a GPU if JAX can see one, otherwise one CPU device per performance core (see `-nc`), and will look for `Insane4MemPrO.py` and `martini_v3.itp` in the same directory as the script (or the installed package directory). These paths can be overridden with the `-insane` and `-itp` flags if needed.

Compiled code is cached in `~/.cache/mempro/jax`, so runs after the first one start about 15 s faster. Set `JAX_COMPILATION_CACHE_DIR` to put the cache somewhere else, or `MEMPRO_NO_JAX_CACHE=1` to turn it off.

## Outputs

MemPrO is intended to be run on PDB files, and will output the following files:

* **orientation.txt** — Contains general information about all of the final minima. In general rank 1 is the best orientation, however there are cases where this is not true. `orientation.txt` can then be used to determine the best orientation.

* **orientations.pdb** — All final orientations in a single PDB file, one model per rank, ordered by rank.

* **local_minima_orientation.png** — A representation of how spread out the local minima are. The more white the image, the more confidence can be placed in rank 1. This is only true in general and there are some cases where this can be safely ignored.

* **local_minima_potential.png** — A representation of the potential of the final configuration from a given starting configuration. This complements `local_minima_orientation.png` but in general does not provide much additional information.

* **Rank_{n}/** — A folder containing all relevant information for a particular rank:
  * `oriented_rank_{n}.pdb` — The nth ranked configuration. Contains the protein (atomistic or coarse grain depending on the input) and a dummy membrane. The rank is based on number of hits (or potential, see `-rank` flag).
  * `info_rank_{n}.txt` — Some information about the orientation.
  * `CG_System_rank_{n}/` — A CG system built using Insane4MemPrO for the rank N configuration, including a topology file. Only produced when using `-bd` and `-bd_args`.
  * `PG_potential_curve.png` — The potential curve for the position of the PG cell wall. Only produced with `-pg`.
  * `cross-section_area_curve.png` — The cross-sectional area of the segment of the protein passing through the PG layer. Only produced with `-pg`.
  * `Z_potential_curve.png` — Potential against Z position of the final orientation. Gives an idea of the nature of the minima.
  * `curv_potential_curve.png` — Potential against global membrane curvature for the final orientation. Can indicate whether curvature minimisation should be used. Not produced with `-dm`.

## Flags

**-h, --help** — Displays a help message with all possible flags.

**-f, --file_name** — Input PDB file for the protein to orient. The protein can be either atomistic or coarse grained; the code detects which automatically and ignores unknown atom/bead types. It is recommended that the PDB file has no missing atoms, as this can reduce orientation quality.

**-o, --output** — Name of the output directory. Defaults to `Orient/` in the current working directory.

**-ni, --iters** — Maximum number of minimisation iterations. The default of 150 is sufficient in most cases. Reducing this value increases speed; increasing it can sometimes improve results on difficult systems.

**-ng, --grid_size** — Number of initial starting configurations. Defaults to 36. For best efficiency this should be a multiple of the number of CPUs being used. More starting configurations sample the space of minima better, but there are diminishing returns and generally no more than 40 are needed.

**-nc, --num_cpus** — Number of CPU cores to use. Defaults to the number of performance cores where the system reports them (currently macOS, e.g. 4 on an Apple M4, whose 6 efficiency cores would slow each batch down), otherwise all available cores. For best performance with `-ng`, set `-nc` to a value that divides `-ng` evenly.

**-dev, --device** — Device to run on: `auto` (default; GPU if JAX can see one, otherwise CPU), `cpu` or `gpu`. On GPU, all starting configurations are minimised together in a single batch. GPU support needs `pip install "mempro[gpu]"` (Linux, NVIDIA).

**-sp, --single_precision** — Use 32-bit instead of 64-bit floats. Consumer GPUs (GeForce/RTX) have very little 64-bit throughput, so this makes minimisation much faster there; results can differ slightly from the default 64-bit run. Data-centre GPUs (A100/H100) and CPUs gain less.

**-rank, --rank** — Ranking method for minima: `h` ranks by percentage hits, `p` ranks by potential, `auto` (default) ranks by a value calculated from approximate minima depth and potential.

**-seed, --seed** — Seed for the random number generators. Runs with the same seed, input and flags give identical results, provided the number of devices (`-nc`, or the number of GPUs) is also the same. Default is unseeded.

**-dm, --dual_membrane** — Toggles dual membrane minimisation. Splits the membrane into an inner and outer membrane and minimises the distance between them. Only use this if your protein spans the periplasm or is a gap junction etc.

**-ch, --charge** — Partial charge of the (inner) membrane. The value corresponds to the average charge on a lipid divided by the average area per lipid, representing the average charge across the bilayer sheet. For an E. coli inner membrane this is around -0.008. Default is 0, which gives the best performance in most cases.

**-ch_o, --charge_outer** — Charge for the outer membrane only. Has no effect without `-dm`.

**-mt, --membrane_thickness** — Initial thickness of the (inner) membrane in Angstroms. Default is 28.

**-mt_o, --outer_membrane_thickness** — Initial thickness of the outer membrane in Angstroms. Has no effect without `-dm`. Default is 24.

**-mt_opt, --membrane_thickness_optimisation** — Toggles membrane thickness optimisation. Cannot be used with `-c`.

**-tm, --transmembrane_residues** — Indicates which residues are expected to be transmembrane. Format is a comma-separated list of inclusive-exclusive ranges e.g. `10-40,50-60`. Intended only for situations where the transmembrane region is known and MemPrO is consistently orienting incorrectly. Cannot be used with `-w`.

**-pg, --predict_pg_layer** — Toggles PG cell wall prediction for dual membrane minimisations. Outputs a dummy PG layer at the lowest potential position and two diagnostic graphs. Due to the nature of the PG layer the lowest potential is not necessarily the biologically correct placement; it is recommended to use the graphs for further interpretation. Requires `-dm`.

**-pg_guess, --pg_layer_guess** — An initial guess for the PG layer position, typically based on standard PG layer positions in a specific bacterium.

**-pr, --peripheral** — Use an alternative method for determining initial insertion depth. The usual method places the weighted mean of the hydrophobic residues at the membrane centre, which does not work well for peripheral membrane proteins. The alternative method scans all possible insertion depths within a range. Does not work with `-dm`.

**-w, --use_weights** — Toggles use of b-factors to weight the minimisation. Useful if part of your protein is particularly flexible or poorly predicted by AlphaFold. Do not use this if all b-factors are 0. Cannot be used with `-tm`.

**-wb, --write_bfactors** — Writes individual bead potentials to the b-factors of the output PDB. Useful for diagnosing orientation problems. Not currently compatible with `-c`.

**-c, --curvature** — Toggles global curvature minimisation. Not currently compatible with `-dm` or `-mt_opt`.

**-flip, --flip** — Flips the protein in the Z-axis on output.

**-itp, --itp_file** — Path to a Martini 3 `.itp` file. Defaults to `martini_v3.itp` in the same directory as `MemPrO.py` (or the installed package directory).

**-insane, --insane_path** — Path to `Insane4MemPrO.py`. Defaults to the same directory as `MemPrO.py` (or the installed package directory). Only required when using `-bd`.

**-bd, --build_system** — Number of final ranked configurations to build as CG MD-ready systems using Insane4MemPrO. Default is 0 (no systems built). Currently only works with coarse grain input files.

**-bd_args, --build_arguments** — Arguments to pass to Insane4MemPrO, primarily lipid composition, solvent type etc. Required when using `-bd`.

**-res, --additional_residues** — Comma-separated list of additional residues present in the input file (e.g. `POPE,POPG`).

**-res_itp, --additional_residues_itp_file** — An `.itp` file describing each additional residue. Required when using `-res`.

**-res_cg, --residue_cg_file** — A folder containing files named `RES.pdb` for each additional residue, providing coarse graining information. Examples can be found in the [CG2AT repository](https://github.com/owenvickery/cg2at/tree/master/database/fragments/martini_3-0_slipids/non_protein). Only required for atomistic inputs with additional residues.

## Examples

The examples below use `python PATH/TO/MemPrO.py`; if installed via pip, replace this with `mempro`.

Orient `input_file.pdb` on a grid of 40 starting configurations with 150 minimisation iterations:

```
python PATH/TO/MemPrO.py -f input_file.pdb -o Output_dir/ -ng 40 -ni 150
```

Orient with global curvature minimisation:

```
python PATH/TO/MemPrO.py -f input_file.pdb -o Output_dir/ -ng 40 -ni 150 -c
```

Orient and write per-bead potentials to b-factors (`-wb` is not compatible with `-c`):

```
python PATH/TO/MemPrO.py -f input_file.pdb -o Output_dir/ -ng 40 -ni 150 -wb
```

Orient with a dual membrane system, then build the top 2 ranked configurations as CG systems. The inner membrane contains POPE, POPG and CARD; the outer membrane contains LIPA:

```
python PATH/TO/MemPrO.py -f input_file.pdb -o Output_dir/ -ng 40 -ni 150 -dm -bd 2 -bd_args "-negi_c0 CL -posi_c0 NA -sol W -l POPE:7 -l POPG:2 -l CARD:1 -uo LIPA"
```

Orient using 8 CPU cores:

```
python PATH/TO/MemPrO.py -f input_file.pdb -o Output_dir/ -nc 8
```

Orient using a GPU (add `-sp` on consumer GeForce/RTX cards for a large speed-up):

```
python PATH/TO/MemPrO.py -f input_file.pdb -o Output_dir/ -dev gpu
```

A more detailed set of tutorials is available [here](MemPrO_tutorials.md).

## Insane4MemPrO

MemPrO comes with Insane4MemPrO, a CG system builder based on [Insane](https://github.com/Tsjerk/Insane). Insane4MemPrO allows the user to build more complex systems with up to 2 membranes, curvature, multiple proteins and more. When used with MemPrO directly via the `-bd` flag the system is built automatically.

### Flags

#### I/O related flags

**-f** — (Optional) Input protein to build a CG system around.

**-o** — (Required) Output file name.

**-p** — (Optional) Output topology file.

**-ct** — (Optional) Creates a template file. B-factors can be edited to indicate placement of multiple proteins. See examples below.

**-in_t** — (Optional) Input a template created by `-ct`. Builds a CG system with proteins (supplied via `-fs`) placed at the positions indicated in the template.

**-fs** — (Optional) A text file with an ordered list of proteins for multiple protein placements.

#### System size

**-x** — (Required) Box size in the x dimension.

**-y** — (Required) Box size in the y dimension.

**-z** — (Required) Box size in the z dimension.

#### Membrane/lipid options

**-l** — (Optional) Lipid type and relative abundance (`NAME[:N]`) in the membrane (or lower leaflet only if `-u` is used).

**-u** — (Optional) Lipid type and relative abundance (`NAME[:N]`) in the upper leaflet.

**-lo** — (Optional) Lipid type and relative abundance (`NAME[:N]`) in the outer membrane (or lower leaflet if `-uo` is used).

**-uo** — (Optional) Lipid type and relative abundance (`NAME[:N]`) in the outer membrane upper leaflet.

Lipid names follow the [Martini 3 lipidome](https://github.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters) (Pedersen et al. 2025, ACS Cent. Sci.): all its bilayer lipids can be built, bead for bead as in its `.itp` files, including ether (`XXEC`, `XXEE`, ...) and plasmalogen (`XXLC`, `XXLE`, ...) phospholipids, sphingomyelins, ceramides, phosphoinositides, cardiolipins (`TOCL`, `TMCL`), BMP (`DO2B`, `DO3B`), `DODG`, `DOTAP` and cholesterol (`CHOL`, or `CHL1`). Use the lipidome `.itp` files for the topology. Their definitions are in `lipidome_m3.dat`. Note that in the lipidome `DLPC`, `DLPE` are 18:2 (12:0 is `DUPC`, `DUPE`); the Martini 3.0 definitions are kept as `DLPC.o`, `DLPE.o` and `DAPC.o`. Insane4MemPrO also has bacterial lipids (LPS, cardiolipins, glycolipids, hopanoids, quinones) and peptidoglycan from other Martini 3 parameter sets.

**-lipid_itp** — (Optional, repeatable) A lipid topology (`.itp`). Lipids in it with an `;@INSANE alhead=... allink=... altail=...` line (as in the lipidome files and those made by its lipid generator) are built from that line, and the charges of all its molecules are used to neutralise the system. A warning is printed if a lipid in use has different bead names in the `.itp` than in Insane4MemPrO's template.

**-alname**, **-alhead**, **-allink**, **-altail** — (Optional, repeatable) Define a lipid from head, linker and tail strings, as in the lipidome lipid generator. Heads: `C` (NC3), `E` (NH3), `G` (GL0), `S` (CNO), `P`/`O` (PO4), `COH`, `PS1`, `PS2`, and inositols `PI`, `P1`-`P7`. Linkers: `G` (GLn), `E` (ETn, ether), `L` (PLn, plasmalogen), `A1 A2` (OH1 AM2, sphingosine). Tails: one string of bead letters per linker (e.g. `CDCC`), where `c`, `t` and `F` give beads named C, T and D.

**-a** — (Optional) Area per lipid (nm²) in the membrane (or lower leaflet if `-au` is used). By default the area per lipid of each leaflet is the abundance-weighted mean of the built-in per-lipid values (see `-apl`), so for example `-l POPE:7 -l POPG:2 -l CARD:1` gives about 0.67 nm² and `-uo LIPA` 1.56 nm². Built-in values are fluid-phase experimental areas (~303 K) where available, otherwise from simulation; sources are noted beside `lipid_apl` in `Insane4MemPrO.py`. The areas used are printed when building. In mixed leaflets, lipids much larger than the smallest one take several grid sites (area / smallest area, rounded), so for example cardiolipin among PE takes two sites and sits between them, and cholesterol among POPC takes half the space of a POPC. Giving `-a`/`-au`/`-ao`/`-auo` explicitly places one lipid per site, as before.

**-au** — (Optional) Area per lipid (nm²) in the upper leaflet.

**-ao** — (Optional) Area per lipid (nm²) in the outer membrane (or lower leaflet if `-auo` is used).

**-auo** — (Optional) Area per lipid (nm²) in the outer membrane upper leaflet.

**-apl** — (Optional, repeatable) Area per lipid (nm²) for one lipid type (`NAME:AREA`), overriding the built-in value. Built-in values are in `lipid_apl` in `Insane4MemPrO.py`: experimental areas from the Martini 3 lipidome benchmark where measured, otherwise the lipidome's Martini 3 single-lipid bilayer areas scaled by 1.04 (the mean ratio of experiment to those simulations, sd 0.03). Ether and plasmalogen lipids take the area of their diacyl analogue +1.5% and +5%. Lipids not listed use a value for their headgroup class (PC, PE, PG, PS, PA, SM, ceramide) or 0.6 nm².

**-grid** — (Default: circular) Lipid grid for flat membranes filling the box. `circular` places lipids on concentric rings; `square` uses a square lattice that is continuous across the periodic boundaries, avoiding close contacts where clipped rings meet at the box edges. Either way a flat leaflet gets (box area)/(area per lipid) lipids before any are removed around the protein. Curved membranes and `-radius` disks always use rings.

**-temp** — (Optional) Temperature (K) for the built-in phospholipid, cardiolipin and phosphoinositide areas, which otherwise refer to ~303 K (or the lowest temperature at which the lipid is fluid, e.g. DPPC 323 K, sphingomyelin 318 K; see `apl_ref_temp`). Areas are scaled by 0.25% per K, the typical fluid-phase expansion. LPS, sterol and hopanoid values are not scaled. Below a lipid's melting temperature (e.g. DPPC, sphingomyelin) the scaled value is not meaningful.

**-apl_ff** — (Default: exp) `exp` uses experimental areas; `m3` scales the same lipids to Martini 3 equilibrium areas (0.94 × experimental, as found for PO lipids in the Martini 3 lipidome benchmark), so the box starts closer to its relaxed size. Total box area relaxes in NPT either way; this mainly reduces the initial shrinkage. Cholesterol is not corrected for condensing: it keeps a steric 0.40 nm², and as it flips between leaflets in Martini any resulting imbalance relaxes.

**-balance** — (Default: off) Balance the two leaflets of each membrane after lipids clashing with the protein have been removed, by removing lipids at random from the fuller leaflet. `area` matches the total lipid area of the leaflets (number × area per lipid), `number` matches the lipid counts. The lipid area of each leaflet is always printed, with a warning if they differ by more than 5%, as a large mismatch can induce curvature. Some difference is expected where the protein occupies more of one leaflet than the other, in which case balancing would leave the fuller leaflet under-packed.

**-footprint** — (Default: clash) How lipids are removed around the protein. `clash` keeps each grid site whose lipid tails can be traced past the protein (and, without `-ring`, past beads filling the regions the protein encloses), so how much of each leaflet the protein takes depends on the grid, `-fudge` and whether the tracing succeeds. `map` measures it instead: each leaflet is sliced every 0.2 nm from the midplane to its phosphate plane, plus a 0.6 nm headgroup layer, and on a 0.1 nm grid lipids fill the core volume the protein leaves free (a domain resting on the headgroups excludes lipids beneath it). Each leaflet then gets exactly (free area)/(area per lipid) lipids, on grid sites the protein leaves free plus sites added in the largest gaps, and a lipid whose tails cannot be traced past the protein moves to a spare site. The footprint, free area and number of lipids of each leaflet are printed. Regions lipids cannot reach in any slice (the inside of a closed barrel) count as protein unless `-ring` is given. `map` is for flat membranes filling the box (also with `-ps` and `-def`); curved membranes, pores, micelles and `-radius` disks use `clash`. `mempro-leaflets` uses the same footprint, so it can check a `map` build directly. See `docs/footprint-design.md`.

**-mt** — (Optional) Membrane thickness (nm) between the phosphate planes for `-footprint map`, as fitted by MemPrO; one value, or two comma-separated for the `-l`/`-u` and `-lo`/`-uo` membranes. MemPrO passes it when building with `-bd`. By default each leaflet extends to the phosphate beads of its lipids as built (about 1.65 nm from the midplane for POPE).

**-fp_map** — (Optional) Write each leaflet's `-footprint map` (available area fraction and placement map, 0.1 nm pixels) to this `.npz` file.

**-ps** — (Default: 0) Distance between inner and outer membrane.

**-curv** — (Default: 0,0,1) Membrane curvature: three comma-separated values for curvature at the peak, curvature at the base, and direction of curvature.

**-curv_o** — (Default: 0,0,1) Curvature of the outer membrane. See `-curv`.

**-curv_ext** — (Default: 3) Extent of the curved region in the absence of a protein.

**-micelle** — (Optional) Builds a micelle around the protein instead of a bilayer.

**-radius** — (Optional) Builds a membrane disk with the given radius, useful for simulating nanodiscs.

**-def** — (Optional) Takes as input the `Membrane_data` folder generated by [MemPrOD](https://github.com/ShufflerBardOnTheEdge/MemPrOD). Builds the deformations predicted by MemPrOD. Cannot be used with `-curv` or `-ps`.

#### Peptidoglycan layer options

**-pgl** — (Default: 0) Number of PG layers to place at `-pgl_z`.

**-pgl_z** — (Default: 0) Z position of the PG layer relative to the centre of the periplasmic space.

**-cper** — (Default: 0.4) Percentage of cross-links.

**-lper** — (Default: 0.1) Percentage of cross-links that are between layers.

**-per33** — (Default: 0.03) Percentage of 3-3 cross-links; all other cross-links will be 3-4.

**-oper** — (Default: 1) Percentage chance of a monomer linking with an oligomer. (Actual chance of link is `cper * oper`.)

**-gdist** — (Default: 0.75,4,8.9,0.25,10,45) Distribution of glycan strand lengths. Format as `weight1,stddev1,mean1,weight2,...` where each triple describes a Gaussian. The sum of these Gaussians forms the distribution.

#### Protein options

**-fudge** — (Default: 0.3) Fudge factor for allowing lipid-protein overlap. A value of 0.15 is recommended when using `-def`.

#### Solvent options

**-sol** — (Required) Solvent type.

**-sold** — (Default: 0.5) Solvent diameter.

**-solr** — (Default: 0.1) Magnitude of random deviations to solvent positions.

#### Charge options

**-posi_c0** — (Required) Positive ion type and relative abundance (`NAME[:N]`) in the system (or compartment 0 if `-posi_c1`/`-posi_c2` are also used). When using multiple membranes, disjoint water compartments may form; these flags allow each compartment to be neutralised independently with different ions.

**-negi_c0** — (Required) Negative ion type and relative abundance (`NAME[:N]`) in the system (or compartment 0).

**-posi_c1** — (Optional) Positive ion type and relative abundance in compartment 1.

**-negi_c1** — (Optional) Negative ion type and relative abundance in compartment 1.

**-posi_c2** — (Optional) Positive ion type and relative abundance in compartment 2.

**-negi_c2** — (Optional) Negative ion type and relative abundance in compartment 2.

**-ion_conc** — (Default: 0.15,0.15,0.15) Ion concentration in each compartment.

**-charge** — (Default: `auto`) Charge of the system. `auto` detects charge automatically.

**-charge_ratio** — (Optional) How to split the charge across compartments. If not supplied, each compartment is neutralised separately.

**-zpbc** — (Optional) Determines whether Z periodicity is used when calculating compartments.

### Examples

These instructions walk through building a CG system with a curved membrane containing multiple proteins.

First, create a template:

```
python PATH/TO/Insane4MemPrO.py -l POPE -sol W -x 10 -y 10 -z 50 -o test.gro -p topol.top -curv 0.1,0.15,1 -fudge 0.3 -curv_ext 6 -ct template.pdb -negi_c0 CL -posi_c0 NA
```

This creates a POPE-only membrane in a (10,10,50) box (automatically enlarged if needed to accommodate the curvature). The curvature `0.1,0.15,1` specifies a peak curvature of 0.1, a base curvature of 0.15, and the curved region pointing in the positive Z direction, maintained for 6 Å before returning to planar.

Next, open `template.pdb` in PyMOL. Protein positions are indicated by b-factors. Select a single bead at each desired protein location and run `alter sele,b=N` where N increments by 1 for each protein (the Nth protein will be placed at the bead with b-factor N). Once all b-factors are set, run `save template.pdb`.

Then build the final system:

```
python PATH/TO/Insane4MemPrO.py -l POPE -sol W -x 10 -y 10 -z 50 -o test.gro -p topol.top -curv 0.1,0.15,1 -fudge 0.3 -curv_ext 6 -in_t template.pdb -fs prots.txt -negi_c0 CL -posi_c0 NA
```

Ensure all membrane-relevant values match the template command. `-in_t` supplies the edited template and `-fs` supplies a text file listing the same number of proteins as edited b-factors. The `-p` flag produces a rudimentary `.top` file; protein-related entries will need to be updated but all other values are correct.

More detailed tutorials are available [here](Insane4MemPrO_tutorials.md).

### Matching leaflets after equilibration

Lipids don't flip between leaflets in Martini (except sterols), so if one leaflet starts more crowded than the other the mismatch persists and can bend the membrane. This matters most for asymmetric membranes (e.g. LPS over phospholipids) and around proteins that occupy more of one leaflet than the other. After a short NPT equilibration, check the leaflets with:

```
mempro-leaflets eq.gro
```

(or `python PATH/TO/leaflet_check.py eq.gro`; several frames or a multi-frame `.gro` from `gmx trjconv` can be given and are averaged). For each membrane it reports each leaflet's available area (box area minus the protein's footprint in that leaflet, as for Insane4MemPrO `-footprint map`, with the leaflets extending to their mean phosphate planes in the frame), the area its lipids should occupy (from the Insane4MemPrO table, with the same `-apl`, `-temp` and `-apl_ff` options), how stretched or compressed each leaflet is, and a suggested `-asym` to rebuild with so both are equally strained. This is area matching, an approximation to zero differential stress between the leaflets, so treat the suggestion as a guide, rebuild and check again. `-footprint section` uses the earlier definition instead, the protein's largest cross-section in 0.5 nm slices with bead radius `-radius` (default 0.26 nm). `-asym` applies to every membrane in the system.

## My protein didn't orient correctly

If your protein hasn't oriented correctly there are a number of things to try:

* Check all orientations by loading `orientations.pdb` and reviewing `orientation.txt`. The correct orientation may not be rank 1.
* Check `curv_potential_curve.png` to see if your protein prefers a curved membrane. If so, using `-c` will help.
* Check the PDB file for missing atoms or residues, which can cause the surface to be incorrectly evaluated. Missing atoms should be fixed before orientation.
* Using `-wb` can be helpful for diagnosing missing atoms — it writes the potential contribution of each residue as a b-factor, viewable in PyMOL.
* If your protein is a peripheral membrane protein, using `-pr` will help greatly. In some cases `-pr` will also help integral membrane proteins.
* MemPrO runs with a membrane charge of 0 by default. Some proteins that associate with the membrane through charge interactions will not orient correctly at 0 charge. For these, using `-ch` (or `-ch_o` for the outer membrane) to set the charge will help. Values around `-0.005` are a good starting point.

## FAQ

There are currently no frequently asked questions. If you have any questions or encounter errors you cannot fix, please contact phillip.stansfeld@warwick.ac.uk.
