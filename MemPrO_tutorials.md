# Tutorials

Note that these tutorials assume a Linux OS, however as MemPrO is a Python script it should also work on any other OS as long as you can run Python. Additionally, PyMOL will be used as the molecular visualisation program throughout, however VMD or any other such program can be used.

Commands below use `python PATH/TO/MemPrO.py`; if MemPrO is installed via pip, replace this with `mempro`.

## Tutorial 1 - A Basic Example

This first tutorial will run through how to use MemPrO for the most basic use-case: the orientation of an integral membrane protein in a planar membrane. The first step is to download an example integral membrane protein. Let us choose 4G1U from the Protein Data Bank. To download this, use the fetch command in PyMOL followed by saving as a `.pdb` — further details can be found [here](https://pymolwiki.org/index.php/Fetch). Otherwise go to the [following page on the PDB website](https://www.rcsb.org/structure/4g1u) and download in PDB format.

![Alt text](Tutorial_pics/Fig1.svg)

Now create a folder called "Tutorial1" to contain all the files for this tutorial and place the downloaded PDB file in it. This structure will include some ligands; these can either be removed or, as they do not affect orientation, ignored. In a terminal navigate to the folder you just created.

We will now run MemPrO:

```
python PATH/TO/MemPrO.py -f 4g1u.pdb -ng 16 -ni 150
```

Here we are using an initial grid of 16 starting configurations (the more the better, though 16 is sufficient) and 150 minimisation iterations (again the more the better, but 150 is enough for most cases). By default MemPrO will use all available CPU cores. To limit this, add `-nc N` where N is the number of cores you want to use.

Once the code has finished running, which should take around 90 seconds on a modern machine, you should find a folder called "Orient" in Tutorial1. Opening this folder you should find several files and folders. First look at `orientations.txt`, which contains all the orientations found by MemPrO. MemPrO can be a little variable between runs so your results may differ slightly from what is shown here, but they should be very similar.

![Alt text](Tutorial_pics/Fig2.svg)

In this case MemPrO found two possible orientations. Each orientation has 6 numbers associated with it. The first is its rank. Let us look at the rank 1 orientation.

![Alt text](Tutorial_pics/Fig3.svg)

The second number represents the relative potential of the orientation, where 0 would be a protein fully in solvent. The rank 1 orientation here has a lower potential than if the protein were not in the membrane, which is a good sign. The third number is the percentage of configurations that minimised to this particular orientation; a value of 93.75% indicates high confidence, though a low value does not always mean low confidence. The fourth number should be very close to the second and can otherwise be ignored. The fifth number indicates the calculated depth of the minima — the higher the value the more stable the orientation. The sixth and final number is calculated from the other values and is used to rank orientations; it can only be used to compare orientations within the same run.

Knowing what these numbers mean, one can see that the rank 1 orientation is a very stable and deep minima while any others are clearly not. We can now look at `orientations.pdb`.

`orientations.pdb` contains all orientations in a single file, ordered by rank. Looking at each orientation in turn, one should be able to verify that the rank 1 orientation looks sensible while the others are more questionable. MemPrO outputs all minima found, regardless of quality, to provide as much information as possible.

Within the Orient folder there will be a folder for each rank. Let us look in "Rank_1". In this folder we should find `info_rank_1.txt`, which contains some additional information on the orientation, two images, and the oriented protein PDB file containing a dummy membrane for visualisation.

`Z_potential_curve.png` shows the potential as the protein moves through the membrane. There should be a clear minima at around 20 Å with large peaks either side, indicating a stable orientation. `curv_potential_curve.png` shows the potential as curvature is varied. A curved membrane is only predicted when using `-c`, however this graph is always calculated. In this case the minima is at around 0 curvature, confirming that a planar membrane is sufficient. We will see cases later where this graph indicates the need for curvature orientation.

Hopefully, one can see that MemPrO is straightforward to use and also provides a lot of information about the orientation. The next tutorials focus on a few of the more advanced features.

## Tutorial 2 - Double Membrane Systems

We will now look at double membrane orientation in MemPrO. Download the PDB for a double membrane protein — let us choose 5NIK. Use the fetch command in PyMOL or go to the [following page on the PDB website](https://www.rcsb.org/structure/5nik) and download in PDB format.

Create a folder called "Tutorial2", place the downloaded PDB file in it, and navigate there in a terminal. Run the following:

```
python PATH/TO/MemPrO.py -f 5nik.pdb -ng 16 -ni 150 -dm
```

The only difference from Tutorial 1 is the addition of `-dm`, which tells MemPrO to use two membranes.

Once the code has finished running (around 140 seconds) you should find a folder called "Orient" in Tutorial2. The structure is very similar to before with a few differences. The ranking method is different for double membrane proteins, which is reflected in `orientations.txt` — the final 3 values for each rank will be 0. Double membrane orientations are ranked on relative potential only, due to the different nature of such orientations.

![Alt text](Tutorial_pics/Fig4.svg)

Within "Rank_1" there are no longer two images, but `info_rank_1.txt` and `oriented_rank_1.pdb` are still present. Looking at `info_rank_1.txt`, the first line shows "Inter-Membrane distance" with a value of around 272 Å, indicating the distance between the inner and outer membranes.

## Tutorial 3 - Predicting the PG Layer

5NIK is a periplasm-spanning protein, so the peptidoglycan (PG) layer would reside between the inner and outer membranes. MemPrO can predict the placement of this. Copy `5nik.pdb` to a new folder called "Tutorial3" and run the following:

```
python PATH/TO/MemPrO.py -f 5nik.pdb -ng 16 -ni 150 -dm -pg
```

After the code has finished running (around 150 seconds) look in the "Orient" folder.

In "Orient/Rank_1" there will now be two additional graphs. Focus on `PG_potential_curve.png`, which shows the potential associated with placing the PG layer at each position. The potential should be lowest around 0–30 Å from the centre, with two reasonably deep minima at around ~10 and ~30 Å. The first of these, at the lowest potential, is added to `oriented_rank_1.pdb` as a set of dummy beads. The second is also a valid placement, as external factors can easily tip the balance between two close minima. In `info_rank_1.txt` you will find the exact position and cross-sectional area of the protein at that position.

![Alt text](Tutorial_pics/Fig5.svg)

If additional information about the PG layer position is known — such as the length of LPP in the particular bacterium — then `-pg_guess` can be used to bias the potential. The value is the distance from the outer membrane to the PG layer. For E. coli, LPP is about 75 Å long (corresponding to ~30 Å from the centre, our second minima), so we can run:

```
python PATH/TO/MemPrO.py -f 5nik.pdb -ng 16 -ni 150 -dm -pg -pg_guess 75 -o Orient_PG_Guess/
```

This will output to a folder called "Orient_PG_Guess" as specified by the `-o` flag.

![Alt text](Tutorial_pics/Fig6.svg)

Looking at `Rank_1/oriented_rank_1.pdb` in Orient_PG_Guess, the PG layer should now be placed higher up the protein.

## Tutorial 4 - Global Curvature Predictions

We will now look at predicting global membrane curvature. Download a PDB for a protein that causes membrane curvature — let us choose 6BPZ. Use the fetch command in PyMOL or go to the [following page on the PDB website](https://www.rcsb.org/structure/6bpz) and download in PDB format.

Create a folder called "Tutorial4" and place `6bpz.pdb` in it. We will start with a standard planar orientation to illustrate why global curvature predictions are sometimes needed.

```
python PATH/TO/MemPrO.py -f 6bpz.pdb -ng 16 -ni 150 -o Orient_NoCurv/
```

Looking in "Orient_NoCurv" there are many more ranks than in previous examples. Looking at `orientations.txt` we see none of the orientations have a negative potential, which can be indicative of a problematic orientation. The first few ranks have a final ranking value much higher than the others. Looking at those orientations in `orientations.pdb`, the transmembrane regions are placed roughly within the membrane but with a lot of variance in position, indicating either a highly mobile or unstable orientation.

![Alt text](Tutorial_pics/Fig7.svg)

Looking in "Rank_1", `Z_potential_curve.png` shows an extremely deep minima suggesting the orientation may not be mobile in Z. `curv_potential_curve.png` reveals the issue — it shows a minima at around -0.005, meaning a fairly significant negative curvature has lower potential than a planar membrane. Let us now run with global curvature prediction:

![Alt text](Tutorial_pics/Fig8.svg)

```
python PATH/TO/MemPrO.py -f 6bpz.pdb -ng 16 -ni 150 -o Orient_Curv/ -c
```

Looking in "Orient_Curv" we can immediately see fewer final orientations. `orientations.txt` shows only one orientation with negative potential and a deep minima, indicating a much more stable prediction. `oriented_rank_1.pdb` shows the protein placed in a highly curved membrane. `curv_potential_curve.png` shows the minima has actually shifted further — this is because without curvature prediction there is greater error in the placement which affects the curvature calculations.

![Alt text](Tutorial_pics/Fig9.svg)

![Alt text](Tutorial_pics/Fig10.svg)

It is not always possible to see whether a membrane should be curved from `curv_potential_curve.png` in a planar orientation, but in many cases it can give an indication of whether the protein prefers curved environments.

## Tutorial 5 - Building CG Systems from Orientations

In this tutorial we will use MemPrO to orient 5NIK and automatically build a CG system. For the purpose of this tutorial both membranes will be made up of POPG and POPE, even though this is not biologically accurate. Further tutorials for Insane4MemPrO are available [here](Insane4MemPrO_tutorials.md). Start by creating a folder called "Tutorial5" and copying or downloading `5nik.pdb` as in [Tutorial 2](#tutorial-2---double-membrane-systems).

To build a CG system, 5NIK must first be coarse grained. For this we will use Martinize2 — for install instructions and usage refer to the [GitHub repo](https://github.com/marrink-lab/vermouth-martinize).

Run the following to coarse grain 5NIK:

```
martinize2 -f 5nik.pdb -ff martini3001 -x 5nik-cg.pdb -o 5nik-cg.top -dssp PATH/TO/mkdssp -scfix -elastic -ef 500 -eu 0.9 -el 0.5 -ea 0 -ep 0 -merge A -maxwarn 1000
```

The details of this command are not important for this tutorial. For running CG simulations these values will need to be set correctly for the kind of simulation you wish to run.

Once we have `5nik-cg.pdb`, run the following:

```
python PATH/TO/MemPrO.py -f 5nik-cg.pdb -ng 16 -ni 150 -dm -bd 1 -bd_args "-l POPE:8 -l POPG:2 -sol W -negi_c0 CL -posi_c0 NA"
```

The `-bd` flag indicates we want to build a CG system for the top n (here n=1) ranked configurations. The `-bd_args` flag is a string of additional arguments to pass to Insane4MemPrO. MemPrO handles naming of all output files, the type of system to build (in this case a double membrane system), and the simulation cell size. Everything else must go in the `-bd_args` string. Here `-l` indicates the lipid type and relative abundance (POPE and POPG in a ratio of 8:2) in the lower leaflet of the inner membrane; without `-u`, `-uo`, or `-lo` this composition is used for all leaflets. `-negi_c0` and `-posi_c0` specify the negative and positive ions (Cl⁻ and Na⁺), and `-sol` specifies the solvent (water). For a full explanation of all flags, refer to the [Insane4MemPrO documentation](README.md#insane4mempro).

Looking in "Orient", the familiar output files from MemPrO are present. Checking `orientations.txt` and `orientations.pdb` should confirm that orientation has gone well and the rank 1 orientation looks sensible. In "Rank_1" there is an additional folder "CG_System_rank_1" containing three files: `CG_system.gro`, `protein-cg.pdb`, and `topol.top`. `topol.top` is a basic topology file that may need minor editing but will contain the correct molecule counts. `protein-cg.pdb` is a copy of the oriented protein without dummy membranes. `CG_system.gro` is the full CG system built according to the arguments passed to MemPrO.

![Alt text](Tutorial_pics/Fig11.svg)

These files can be used together with the output from Martinize2 and the appropriate `.itp` files to run simulation.

## Tutorial 6 - Orienting Protein-Lipid Complexes

In this tutorial we will orient a simple protein-lipid complex. First create the complex by following [Insane4MemPrO — Tutorial 1](https://github.com/ShufflerBardOnTheEdge/MemPrO/blob/main/Insane4MemPrO_tutorials.md#tutorial-1---a-basic-example) to generate a CG system with POPE and the protein 4G1U.

Create a folder called "Tutorial6" and copy the final CG system across from Insane4MemPrO Tutorial 1. For more interesting results you may energy minimise the system before continuing. To create a protein-lipid complex, load `CG-System.gro` in PyMOL, then hide everything except the protein and lipids:

```
hide
show spheres,pol
show spheres,resn POPE
```

Select a few of the lipids around the protein by clicking on each one, then run:

```
save 4G1U_Lipid.pdb,pol or sele
```

This creates `4G1U_Lipid.pdb`, our protein-lipid complex. Make sure PyMOL is in the correct directory when saving. We can now orient this complex using:

```
python PATH/TO/MemPrO.py -f 4G1U_Lipid.pdb -res POPE -res_itp PATH/TO/martini_v3.0.0_phospholipids_v1.itp -ng 16 -ni 150
```

Martini 3 should include `martini_v3.0.0_phospholipids_v1.itp`, but any `.itp` describing a Martini 3 coarse grained POPE will work. Once the code has finished, look at `Orient/Rank_1/oriented_rank_1.pdb` to check the orientation looks correct.

![Alt text](Tutorial_pics/Fig20.svg)

For an atomistic input, the flag `-res_cg` will also be needed. This takes a folder containing files named `RES.pdb` for each residue being added (in this case the folder would contain `POPE.pdb`) with the coarse graining information. The format is based on CG2AT; an example for POPE can be found on the CG2AT GitHub repo [here](https://github.com/owenvickery/cg2at/tree/master/database/fragments/martini_3-0_slipids/non_protein/POPE).

## Final Comments

Hopefully with the above 6 tutorials you should now be set to use MemPrO to orient proteins and build CG systems ready for simulation. MemPrO has a few more advanced features, and may have more in future, that are not covered here. Some advanced tutorials may become available in the future. For now, if you run into difficulties or find errors, please let me know by emailing phillip.stansfeld@warwick.ac.uk and I'll do my best to help.
