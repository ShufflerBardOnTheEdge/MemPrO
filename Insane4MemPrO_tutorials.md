# Tutorials

Note that these tutorials assume a Linux OS, however as MemPrO is a Python script it should also work on any other OS as long as you can run Python. PyMOL will be used as the molecular visualisation program throughout, however VMD or any other such program can be used. It is recommended to look at the [first MemPrO tutorial](MemPrO_tutorials.md#tutorial-1---a-basic-example) before starting these.

Commands below use `python PATH/TO/MemPrO.py` and `python PATH/TO/Insane4MemPrO.py`; if MemPrO is installed via pip, replace these with `mempro` and `insane4mempro` respectively.

## Tutorial 1 - A Basic Example

In this tutorial we will build a CG system with a simple POPE lipid bilayer with an integral membrane protein embedded within. For all of the following tutorials proteins will need to be in coarse grained format. We will go over coarse graining using the protein 4G1U, which we will then use to build the full CG system. To download 4G1U, use the fetch command in PyMOL followed by saving as a `.pdb` — further details can be found [here](https://pymolwiki.org/index.php/Fetch). Otherwise go to the [following page on the PDB website](https://www.rcsb.org/structure/4g1u) and download in PDB format. Create a folder called "Tutorial1" and place the downloaded file into it.

We will be using Martinize2 for coarse graining — for install instructions and usage refer to the [GitHub repo](https://github.com/marrink-lab/vermouth-martinize). Navigate to "Tutorial1" and run the following:

```
martinize2 -f 4g1u.pdb -ff martini3001 -x 4g1u-cg.pdb -o 4g1u-cg.top -dssp PATH/TO/mkdssp -scfix -elastic -ef 500 -eu 0.9 -el 0.5 -ea 0 -ep 0 -merge A -maxwarn 1000
```

This will generate several files. `4g1u-cg.pdb` is the coarse grained structure, which can be inspected in PyMOL by running `show spheres`. `4g1u-cg.top` contains topology information for the coarse grained protein, needed when amending topology files after building the CG system. Finally, the `.itp` files will be used in the topology file later.

![Alt text](Tutorial_pics/Fig12.svg)

We now want to orient the protein so that it will be in the correct position when building the CG system. Note that we can build the system automatically after orientation using MemPrO, but as this tutorial is for Insane4MemPrO we will build it manually. To orient the protein:

```
python PATH/TO/MemPrO.py -f 4g1u-cg.pdb -ng 16 -ni 150
```

![Alt text](Tutorial_pics/Fig13.svg)

For more details on MemPrO refer to the [MemPrO tutorials](MemPrO_tutorials.md). Once finished, look at `Orient/Rank_1/oriented_rank_1.pdb` in PyMOL to check orientation has proceeded correctly. Create a copy of the oriented protein without the dummy membrane:

```
sed '/DUM/d' ./Orient/Rank_1/oriented_rank_1.pdb > 4g1u-oriented.pdb
```

Now run the following to create the full CG system:

```
python PATH/TO/Insane4MemPrO.py -f 4g1u-oriented.pdb -p topol.top -o CG-System.gro -x 20 -y 20 -z 20 -sol W -l POPE -negi_c0 CL -posi_c0 NA
```

`-p` and `-o` set the output file names. `CG-System.gro` is the CG system and `topol.top` is the topology file needed for running simulations. Load `CG-System.gro` in PyMOL — type `show spheres` to show all beads and `show cell` to display the simulation cell. `-x`, `-y`, `-z` control the cell dimensions (here 20 nm each). `-sol W` sets the solvent to water. `-l POPE` defines a membrane composed entirely of POPE. `-negi_c0` and `-posi_c0` define the negative and positive ions (Cl⁻ and Na⁺ respectively).

![Alt text](Tutorial_pics/Fig14.svg)

There is one final step before simulation. Open `topol.top` and look at the `[molecules]` section. The first entry, "Protein", is a placeholder. Open `4g1u-cg.top`, copy everything under `[molecules]`, and replace the "Protein" line in `topol.top` with this. Additionally, replace `include protein-cg.top` with `include molecule_{n}.itp` for each `.itp` file generated during coarse graining (n = 0, 1, 2...). This process may be automated in a future version of Insane4MemPrO.

![Alt text](Tutorial_pics/Fig15.svg)

The system is now ready for simulation. This is where each tutorial ends — there are already many good tutorials for running coarse grained molecular dynamics, but remember to energy minimise first!

## Tutorial 2 - Building a Curved System

In this tutorial we will build a curved membrane with a more complex lipid composition using POPE and POPG. To simulate with these lipids they must be defined in the appropriate `.itp` files.

Download the protein 7N5E from the PDB. Use the fetch command in PyMOL or go to the [following page on the PDB website](https://www.rcsb.org/structure/7n5e) and download in PDB format. Create a folder called "Tutorial2" and place the file into it.

Coarse grain the protein as in [Tutorial 1](#tutorial-1---a-basic-example), naming the output `7n5e-cg.pdb`. Next we will orient the protein in a curved membrane. We will build the system manually first and then show how MemPrO can do it automatically.

Run the following to orient 7N5E in a curved membrane:

```
python PATH/TO/MemPrO.py -f 7n5e-cg.pdb -ng 16 -ni 150 -c
```

![Alt text](Tutorial_pics/Fig16.svg)

`Orient/orientation.txt` should show a clear rank 1. In `Rank_1/info_rank_1.txt` you will see a curvature of roughly -0.013 Å⁻¹ — we will need this value to build the CG system. Follow the procedure in [Tutorial 1](#tutorial-1---a-basic-example) to make a copy of the oriented protein without dummy membranes, called `7n5e-oriented.pdb`. To build the system:

```
python PATH/TO/Insane4MemPrO.py -f 7n5e-oriented.pdb -p topol.top -o CG-System.gro -x 20 -y 20 -z 30 -curv 0.13,0.1,-1 -sol W -l POPE:4 -l POPG:1 -negi_c0 CL -posi_c0 NA
```

A few changes from Tutorial 1: the box in Z has been increased to accommodate the curved region. `-curv 0.13,0.1,-1` describes the curvature — the first value (0.13) is the peak curvature in nm⁻¹ from MemPrO, the second (0.1) is the curvature as the membrane returns to planar, and the third (-1) is the sign of the curvature. `-l POPE:4 -l POPG:1` creates a membrane with POPE and POPG in a ratio of 4:1.

![Alt text](Tutorial_pics/Fig17.svg)

Loading `CG-System.gro` we can inspect the CG system. The simulation box will be fairly large to allow the curvature to relax back to planar and to leave enough space between periodic images. Update `topol.top` as described in [Tutorial 1](#tutorial-1---a-basic-example).

We can also build this automatically with MemPrO:

```
python PATH/TO/MemPrO.py -f 7n5e-cg.pdb -o Orient_build/ -ng 16 -ni 150 -c -bd 1 -bd_args "-sol W -l POPE:4 -l POPG:1 -negi_c0 CL -posi_c0 NA"
```

Only the composition-related flags need to be specified as MemPrO handles all structural elements. Looking in "Orient_build/Rank_1/CG_System_rank_1/" you will find `CG-system.gro` and `topol.top`. Update `topol.top` as always before simulation.

## Tutorial 3 - Double Membrane Systems

In [MemPrO Tutorial 5](MemPrO_tutorials.md#tutorial-5---building-cg-systems-from-orientations) we look at building a double membrane CG system automatically. In this tutorial we will build a more complex version of that system, controlling the ions in each compartment and the lipid composition of the inner and outer membranes — both manually and automatically using MemPrO.

Make a new folder called "Tutorial3". If you have already done [MemPrO Tutorial 5](MemPrO_tutorials.md#tutorial-5---building-cg-systems-from-orientations) copy over `5nik-cg.pdb`, otherwise follow the download instructions in [MemPrO Tutorial 2](MemPrO_tutorials.md#tutorial-2---double-membrane-systems).

Start by orienting the protein:

```
python PATH/TO/MemPrO.py -f 5nik-cg.pdb -ng 16 -ni 150 -dm
```

As in [Tutorial 1](#tutorial-1---a-basic-example), create a copy without the dummy membrane called `5nik-oriented.pdb`. Look at `Orient/Rank_1/info_rank_1.txt` to find the inter-membrane distance, which should be around 272 Å. We can now build the CG system:

```
python PATH/TO/Insane4MemPrO.py -f 5nik-oriented.pdb -p topol.top -o CG-System.gro -x 20 -y 20 -z 50 -ps 13.6 -sol W -l POPE:8 -l POPG:1 -l CARD:1 -uo LIPA -negi_c0 CL -posi_c0 NA -negi_c2 CL -posi_c2 NA:1 -posi_c2 CA:4 -auo 1.8
```

`-ps 13.6` places the two membranes at ±13.6 nm, corresponding to the inter-membrane distance from MemPrO. `-uo LIPA` specifies that the upper leaflet of the outer membrane should be composed entirely of LIPA. `-auo 1.8` sets the area per lipid in the outer membrane upper leaflet to 1.8 nm².

The flags `-negi_c2 CL`, `-posi_c2 NA:1`, and `-posi_c2 CA:4` require a brief explanation of disjoint water compartments. Each membrane in a system is a barrier to water. With one membrane, water and ion beads can freely traverse the periodic boundary — every bead can reach any position in the cell. With a second membrane, beads between the two membranes are trapped, creating two distinct solution compartments. These flags specify that compartment "c2" should contain Cl⁻, Na⁺, and Ca²⁺ ions in the ratio 1:4. There are three possible compartments: c0, c1, and c2.

![Alt text](Tutorial_pics/Fig18.svg)

Loading `CG-System.gro` we can inspect the system and check everything is as expected. Update `topol.top` before simulation.

We can also build this automatically with MemPrO:

```
python PATH/TO/MemPrO.py -f 5nik-cg.pdb -o Orient_build/ -ng 16 -ni 150 -dm -bd 1 -bd_args "-sol W -l POPE:8 -l POPG:1 -l CARD:1 -uo LIPA -negi_c0 CL -posi_c0 NA -negi_c2 CL -posi_c2 NA:1 -posi_c2 CA:4 -auo 1.8"
```

MemPrO handles the double membrane setup; we only need to define the composition. Looking in "Orient_build/Rank_1/CG_System_rank_1/" you will find `CG-system.gro` and `topol.top`. Update `topol.top` as always. It is recommended to use position restraints during equilibration to allow water to fill the interior of the protein correctly.

## Tutorial 4 - Building with Peptidoglycan Layers

In Tutorial 3 we built a more accurate double membrane system. We can further improve the biological accuracy by adding a peptidoglycan (PG) cell wall. Create a new folder called "Tutorial4" and copy over `5nik-cg.pdb` from Tutorial 3 (or follow the download instructions in [MemPrO Tutorial 2](MemPrO_tutorials.md#tutorial-2---double-membrane-systems)). You will also need the files `NAM.itp`, `NAG.itp`, `SNPEP.itp`, `UNPEP.itp`, and `UUNPEP.itp` — the components used to build the PG layer, found in the "PG_Components" folder.

Start by orienting the protein:

```
python PATH/TO/MemPrO.py -f 5nik-cg.pdb -ng 16 -ni 150 -dm -pg
```

As in [Tutorial 1](#tutorial-1---a-basic-example), create a copy without the dummy membrane called `5nik-oriented.pdb`. Look at `Orient/Rank_1/info_rank_1.txt` to find the inter-membrane distance and the predicted PG layer position (around 272 Å and ~30 Å from the centre respectively). We can now build the CG system:

```
python PATH/TO/Insane4MemPrO.py -f 5nik-oriented.pdb -p topol.top -o CG-System.gro -x 20 -y 20 -z 50 -ps 13.6 -sol W -l POPE:8 -l POPG:1 -l CARD:1 -uo LIPA -negi_c0 CL -posi_c0 NA -negi_c2 CL -posi_c2 NA:1 -posi_c2 CA:4 -auo 1.8 -pgl 3 -pgl_z 30 -oper 0 -lper 0.2
```

The new flags here are: `-pgl 3`, which specifies 3 PG layers; `-pgl_z 30`, which places the PG layer 30 Å from the centre (from the MemPrO prediction); and `-oper` and `-lper`, which control the fine detail of the layer. All PG-related flags are explained in detail [here](README.md#peptidoglycan-layer-related-options).

![Alt text](Tutorial_pics/Fig19.svg)

This command additionally outputs `PGL.itp`, an `.itp` file containing bond and angle information for the entire PG layer. Loading `CG-System.gro` you should see the PG layer placed at the position predicted by MemPrO.

We can also build this automatically with MemPrO:

```
python PATH/TO/MemPrO.py -f 5nik-cg.pdb -o Orient_build/ -ng 16 -ni 150 -dm -bd 1 -pg -pg_guess 75 -bd_args "-sol W -l POPE:8 -l POPG:1 -l CARD:1 -uo LIPA -negi_c0 CL -posi_c0 NA -negi_c2 CL -posi_c2 NA:1 -posi_c2 CA:4 -auo 1.8 -pgl 3 -oper 0 -lper 0.2"
```

The `-pgl_z` flag is no longer needed as MemPrO handles the PG layer position automatically. Looking in "Orient_build/Rank_1/CG_System_rank_1/" you will find `CG-system.gro` and `topol.top`. As with Tutorial 3, it is recommended to use position restraints during equilibration.

## Tutorial 5 - Building with Deformations

In this tutorial we will look at how to use Insane4MemPrO to build membrane deformation predictions from MemPrOD. If you haven't used MemPrOD before, refer to the [MemPrOD GitHub](https://github.com/ShufflerBardOnTheEdge/MemPrOD). Create a new folder called "Tutorial5". Download 4BWZ from [here](https://memprotmd.bioch.ox.ac.uk/_ref/PDB/4bwz/_sim/4bwz_default_dppc/), download `4bwz_default_dppc-head-contacts.pdb` and rename it to `4bwz.pdb`. Place `4bwz.pdb` in "Tutorial5".

Coarse grain the protein using Martinize2 as in [Tutorial 1](#tutorial-1---a-basic-example) to produce `4bwz-cg.pdb`. Then orient the protein:

```
python PATH/TO/MemPrO.py -f 4bwz-cg.pdb
```

Create `4bwz-oriented.pdb` without the dummy membrane, as in [Tutorial 1](#tutorial-1---a-basic-example).

Next, predict the membrane deformations using MemPrOD. Installation and usage instructions can be found [here](https://github.com/ShufflerBardOnTheEdge/MemPrOD). Run:

```
python PATH/TO/MemPrOD.py -f 4bwz-oriented.pdb
```

This takes around 100 seconds. A folder called "Deformations" will be created containing several files and folders. We are interested in `Deformations.pdb` and `Membrane_Data/`. Load `Deformations.pdb` and `4bwz-oriented.pdb` together in PyMOL to visualise the deformation prediction. We can now build the CG system with deformations:

```
python PATH/TO/Insane4MemPrO.py -f 4bwz-oriented.pdb -p topol.top -o CG-System.gro -x 20 -y 20 -z 20 -sol W -l POPE -negi_c0 CL -posi_c0 NA -def Deformations/Membrane_Data/
```

The only addition compared to Tutorial 1 is `-def`, which tells Insane4MemPrO where the deformation information is stored. Load `CG-system.gro` in PyMOL and run the following to see the deformations clearly:

```
hide
show spheres,name nh3
show spheres,pol
```

The NH3 beads in the POPE lipids should be deformed in the same way as predicted by MemPrOD. The system can now be energy minimised and equilibrated as usual.

![Alt text](Tutorial_pics/Fig21.svg)

## Final Comments

Hopefully you now feel confident using Insane4MemPrO to build complex systems with and without MemPrO's automatic build features. There are some more advanced features of Insane4MemPrO not covered here, and more tutorials on these may become available in the future. For now, if you run into any problems or bugs please let me know at phillip.stansfeld@warwick.ac.uk and I will do my best to help.
