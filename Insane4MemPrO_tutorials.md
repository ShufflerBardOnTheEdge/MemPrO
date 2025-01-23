# Tutorials

Note that these tutorials assume a Linux OS, however as MemPrO is a python script it should also work on any other OS so long as you can run python. Additionally, PyMOL will be used as the molecular visualisation programm throughout this tutorial, however VMD or any other such program can be used. It is reccomended to have look at the [first MemPrO tutorial](MemPrO_tutorials.md#a_basic_example) before looking at these.

## A Basic Example

In this tutorial we will build a CG system with a simple POPE lipid bilayer with an integral membrane protein emmbedded within. For all of the following tutorials proteins will need to be in a coarse grained format. Each tutorial will have links/instructions for downloading an atomisitic protein from the PDB, here we will quickly go over how to go about coarse graining these proteins. For this tutorial we will go over coarse graining the protein 4G1U, which we will be using to build the full CG system, to download this one can use the fetch commmand in PyMOL followed by saving as a .pdb, further details on this method can be found [here](https://pymolwiki.org/index.php/Fetch). Otherwise go to the [following page on the PDB website](https://www.rcsb.org/structure/4g1u) and download in PDB format. Create a folder called "Tutorial1" and place the downloaded file into it.

We will be using Martinize2 for coarse graining, for install instructions and usage refer to the [GitHub repo](https://github.com/marrink-lab/vermouth-martinize). Navigate to "Tutorial1" and run the following to CG 4g1u:
>martinize2 -f 4g1u.pdb -ff martini3001 -x 4g1u-cg.pdb -o 4g1u-cg.top -dssp PATH/TO/mkdssp -scfix -elastic -ef 500 -eu 0.9 -el 0.5 -ea 0 -ep 0 -merge A -maxwarn 1000

Provided this runs without errors several files will be generated. The first of these will be "4g1u-cg.pdb", which will be the coarse grained copy of 4g1u. One can look at this in PyMOL by first loading the file, then running "show spheres" to check everything has run correctly. Next we have "4g1u-cg.top" which containes the topology inofrmation for the corase grained protein. We will need this when ammending the topology files after the creation of the CG system. Finally, we have the ".itp" files, these will be used in the topolgy file later.

We now want to orient the protein so that it will be in the correct place when building the CG system. Note that we can build this system automatically after orientation using MemPrO, but as this tutorial is for the use of Insane4MemPrO we will not be doing this. To orient to protein run the following:

>python PATH/TO/MemPrO_Script.py -f 4g1u-cg.pdb -ng 16 -ni 150

For more details on the use of MemPrO refer to these [tutorials](MemPrO_tutorials.md#). Once the code has finished running look at the file "Oreint/Rank_1/oriented_rank_1.pdb" in PyMOL to check orientation has procceded correctly. We will now create a copy of this without the dummy membrane by running to following:

>sed '/DUM/d' ./Orient/Rank_1/oriented_rank_1.pdb > 4g1u-oriented.pdb

We now run the following to create the full CG system:

>python PATH/TO/Insane4MemPrO.py -f 4g1u-oriented.pdb -p topol.top -o CG-System.gro -x 20 -y 20 -z 20 -sol W -l POPE -negi_c0 CL -posi_c0 NA

Here -p and -o indicate the name of the output files. "CG-System.gro" is the CG system as the name may suggest. "topol.top" is the topolgy file which will be need for running simulations. Load "CG-System.gro" in PyMOL. Make sure all sphere are visible by typing "show spheres", also type "show cell" to show the simulation cell. -x,-y and -z control the size of the system cell, here we inputted 20 nm, and we can verify this by looking at the bottom of "CG-System.gro". -sol defines the solvent used, in this case "W" indicates water. -l is used to define the composition of the bilayer, in this case we are just defining a membrane composed of only POPE. More detail on -l will be in following tutorials. -negi_c0 and -posi_c0 define the (NEG)ative (I)ons and (POS)itive (I)ons. In this case we are setting negative ions to be CL- and positive ions to be NA+. These work in a similar way to -l and more detail will be in further tutorials. All this can be verified by looking at CG beads in the CG system using PyMOL.

There is one final step before we can begin simulating the CG system. Open "topol.top" and look at the \[molecules\] section. The first molecule is "Protein", this is a placeholder and will need to be replaced. Open "4g1u-cg.top", copy everything under \[molecules\] and replace the line containing "Protein" in "topol.top" with this. Additionally, replace "include protein-cg.top" with "include molecule_{n}.itp" where {n} is 0,1,2... for each .itp created during the coarse graining process. This process may be automated in future versions of Insane4MemPrO.

And we are ready for simulation! This is where will end each tutorial as there are already many good tutorials for running coaose grained molecular dynamics simulation, all I will say is remeber to energy minimise!

## Building a Curved System

In this tutorial we will build a curved membrane with a more complex lipid composition. We will be building a membrane made up of POPC, DOPS and cholesterol, to be able to build using these lipids, they must be defined in Insane4MemPrO, this is identical to the original Insane. Additionally, to run simulations with these lipids they must be defined in itp files. As this tutorial is for using Insane4MemPrO I will not go over in detail how to add lipids to Insane4MemPrO or how to simulate. For help with this refere to ~this and this~.

We start as usual by downloading an appropriate protein from the PDB, in this case we will be downloading a piezo protein with code 7WLT. To download this one can use the fetch commmand in PyMOL followed by saving as a .pdb, further details on this method can be found [here](https://pymolwiki.org/index.php/Fetch). Otherwise go to the [following page on the PDB website](https://www.rcsb.org/structure/7wlt) and download in PDB format. Create a folder called "Tutorial2" and place the downloaded file into it.

Next coarse grain the protein as in [Tutorial 1](#a_basic_example) calling the output 7wlt-cg.pdb. Next we will need to orient the protein in a curved membrane. We can use MemPrO to do this, note that we can also automatically build using MemPrO. We will first build it ourselves after orientation and then build it automatically with MemPrO. 

Run the following to orient 7WLT in a curved membrane.
>python PATH/TO/MemPrO_Script.py -f 7wlt.pdb -ng 16 -ni 150 -c

