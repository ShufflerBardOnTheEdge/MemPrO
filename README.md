# MemPrO
Membrane Protein Orientation in Lipid Bilayers

## Installation
TBD

>pip install MemPrO

There may be additional steps after this to deal with paths. 

MemPrO is also accessible on Google Colab via the link [ADD LINK]. The Colab will contain instructions for use. 

## Outputs

MemPrO is intended to be run on PDB files, and will output the following files:

  * orientation.txt : This is a file that contains general information about all of the final minima. In general rank 1 is the best orientation, however there are cases where this is not true. orientation.txt can then be used to determine the best orientaion.
  * local_minima_orientation.png : This is a representation of how spread out the local minima are. The more white the more confidence can be placed in the rank 1. This is only true in general and there are some cases where this can be safely ignored.

  * local_minima_potential.png : This is a representation of the potential of the final configuration from a given starting configuration. This complements local_minima_orientation.png but in general does not provide much additional information.

  * Rank_{n}: This is a folder that contains all the relevant information for a particular rank. It contains the following files/folders:
    * oriented_rank_{n}.pdb : This is the nth ranked configuration. It will contain the protein (atomistic or coarse grain depending on the input) and a dummy membrane. The rank is based on number of hits (or potential, see flags) .

    * info_rank_{n}.txt : This contains some information about the orientation.

    * CG_System_rank_{n} : This folder contains a CG-system built using insane and the rank N config. It also contains a topology file. This is ouputted when using flags -bg and -bg_args, see the section of flags for syntax.
	
    * PG_potential_curve.png : This is the potential curve for the position of the PG cell wall. This is only outputted when you use flag -pg.
   
    * cross-section_area_curve.png: This shows the cross-sectional area of the segment of the protein that passes through the PG layer. This is only outputted with flag -pg.
	
    * Z_potential_curve.png : This is a graph of potential against Z position of the final orientation. This can give an idea of the kind of minima.
	
    * curv_potential_curve.png : This is a graph of potential against global curvature of membrane with the final orientation. This can indicate of curvature minimisation should be used.
	 
   
MemPrO will also output information about the progress of the oreination to the command line. Additionally MemPrO will output WARNINGs generated by tensorflow, this is due to the version of JAX being used and will hopefully in future be removed. These warnings can in general safely be ingonred.

## Flags
MemPro will takes the following flags as input:


-h, --help : This will display a help message with all possible flags and some informtation on their use.

-f, --file_name : This is the name of the input file for the protein that you wish to orient. It must be a .pdb. In future support for .gro files may be added. The protein can be either atomistic or coarse grained. The code should detect which it is and ingnore unknown atom/bead types, however in the case of an error please let me know and send me the file which caused it. It is reccomended that the PDB files had no missing atoms as this can cause the orientation to be of lower quaility.

-o, --output : This is the name of the ouput directory. The code will deafult to a folder called Orient in the directory where the code is run. The syntax for this flag is -o "Output_dir/". Note that a backslash is required.

-ni, --iters : This indicates the maximum number of minimisation iteration the code will carry out. The default value of 150 should be enough in most cases. For some systems a smaller value can also work, so if greater speed is needed this can be reduced. Equally if the final output does not seem satisfactory then increasing this value can sometimes fix this, or at least provide information.

-ng, --grid_size : This is the number of initial starting configurations from which to minimise. It defaults to 20. For maximum efficiency this should be a multiple of the number of CPUs being used. Having a greater number of starting configurations will sample the space of minima better, however there is diminishing returns and generally no more than 40 will be needed.

-rank, --rank : This can be either "h", "p" or "auto". "h" ranks the minima by % hits, "p" ranks by potential and "auto" ranks by a value depending on approximate minima depth and potential.

-dm, --dual_membrane : This toggles dual membrane minimisations. This allows the code to split the membrane into a inner and outer membrane and minimise the distance between them. Only use this is you know your protein will span the periplasm or is a gap junction etc.

-ch, --charge : This value corresponds to the partial charge of the (inner) membrane. For -ch and -ch_o below the value corresponds to the average charge on a lipid divided by the average area per lipid. This will give a average charge across a sheet that will represent the bilayer. The value for an E-coli membrane is around -0.008. In general bet preformance is obtained by using a charge of 0.

-ch_o, --charge_outer : This is the charge for the outer membrane only. This will do nothing without -dm.

-mt, --membrane_thickness : This is the inital thickness of the (inner) membrane.

-mt_o, --outer_membrane_thickness : This is the initial thickness of the outer membrane. This will do nothing without -dm.

-mt_opt, --membrane_thickness_optimisation : This toggles membrane thickness optimisation. This cannot be used with curvature minimisation (currently).

-tm, --transmembrane_residues : This inidicates which residues are expected to be transmembrane. THe format for this is a comma seperated list of ranges e.g. 10-40,50-60. This is inteneded only for situations where the transmembrane region is known and MemPrO is consistently orienting incorrectly.

-pg --predict_pg_layer : This toggels PG cell wall prediction for dual membrane minimisations. This will output a dummy PG layer where the lowest potential is found and teo graphs. The graphs show the potentail associated with the placemnt of the PG layer at a particular Z and the cross-sectional area of the segment of protein going through the PG layer. Due to the nature of the PG layer the lowest potential in not neccassarialy the biologically correct placemnt, it is therefore reccomeneded use these graphs for further information.

-pr, --peripheral : This tells the code to use an alternative method for determining inital insertion depth. The usual method is to place a weighted mean of the hydrophobic residues at the center of the membrane, this clearly does not work for a peripheral membrane protein or for a protein that does not fully span the membrane. The alternative method scans all possible insertion depths within a range and selects the lowest. This does not work with -dm.

-w, --use_weights : This toggles the use of b-factors to weight the minimsation. This is useful if part of your protein is particularily flexible or poorly predicted by alphafold. Do not use this if all your b-factors are 0 as currently this will just break with not indication of why (This will be changed soon).

-wb, --write_bfactors : This toggles writing of individual bead potentials into b factors. Not currently working with -c.

-c, --curvature : This toggles global curvature minimisation. This is currently not compatible with -dm. 

-itp, --itp_file : The code makes heavy use of the martini parameters. The path to a martini 3 itp file should go here. The default loaction is TBD.

-bd, --build_system : This should be a number that indicates how many of the final configurations should be build using Insane. This will currently only work if your input file is coarse grain.  

-bd_args, --build_arguments: These are the arguments to pass to Insane4MemPrO. This mostly includes padding and lipid composition etc. There are some additional build args for periplasm spanning systems. These are as follows.

## Examples

This should be run before any other commands

>export NUM_CPU=N

Where N is replaced by the number of CPU you wish to use, for the examples below 20 were used.

The below will orient input_file.pdb on a grid of 40 starting configurations with 150 minimisation iterations

>python MemPrO_Script.py -f input_file.pdb -o "Output_dir/" -ng 40 -ni 150 -itp "PATH/TO/MARTINI"



The below will orient input_file.pdb on a grid of 40 starting configurations. It will preform global curvature minimisation alongside usual for 150 iterations. It will write the potential contibutions of each bead to the b-factors of the final output.

>python MemPrO_Script.py -f input_file.pdb -o "Output_dir/" -ng 40 -ni 150 -c -wb -itp "PATH/TO/MARTINI"




The below will orient input_file.pdb on a grid of 40 starting configurations with 150 minimisation interations. It will be minimised with a double membrane system. Once minimised the code will build the top 2 ranks using Insane4MemPrO. The CG system built with insane will have asymentric membranes, one with POPE,POPG and CARD the other with only LIPA. The CG system will have the same charge and salt concentration in each of the seperate compartments created by the presence of two membranes.

>python MemPrO_Script.py -f input_file.pdb -o "Output_dir/" -ng 40 -ni 150 -dm -bd 2 -bd_args "-salt 0.15,0.15,0.15 -charge_ratio 1,1,1 -sol W -l POPE:7 -l POPG:2 -l CARD:1 -lo LIPA" -itp "PATH/TO/MARTINI"
## Insane4MemPrO
MemPrO comes with Insane4MemPrO which is a heavily modifed version of Insane. Insane4MemPrO allows the user to build more complex systems with upto 2 membranes, curvature and multiple proteins. When used with MemPrO directly via the -bd flag the system will be automatically built.


### Additional/edited flags:

-uo/-lo: These are used to define lipid composition of the outer membrane if building a double membrane system. They have the same syntax as the original -u,-l

-ps: This should be a float value equal the half the distance between the two membranes. For flat systems and pores this is the distance between planar regions. For curved double membrane systems it is the distance between the center of each membrane.

-ct: This creates a "template" file along with a txt. The template file is a .pdb which contains a representation of the membrane, this can then be edited in pymol to indicate positions of multiple proteins. The .txt will have the same name as the template and contain the membrane relevant options used to create the membrane.

-in_t: This is the input template for multiple protein placements.

-fs: This is .txt that contains a ordered list of proteins to be placed in the membrane. Each line should be "file_name direction ring". Here file_name is the path to the protein, direction is either 1 or -1 and can be used to flip individual proteins by 180 and ring is either 0 or 1 and determins if lipids inside the protein are removed or not. 0 indicating the lipids are removed.

-curv/-curv_o: These should be 3 comma seperated values "c,rc,dc" where c is the curvature at the center of the membrane, rc is the curvature of the membrane as it returns to planar (Return curvature) and dc is the direction  of the curvature.

-curv_ext: This is the extent of the curved region. When not using templates this will be ignored as the extent is calculated from the protein. This can be used to control the size of the pore.

-pore: This flags if a pore should be created or not. When this is used the first value of -curv will be ignored as will the last. The return curvature cannot be zero and will be set to 0.1 by default.

-fudge: This indicates how close lipids can be the to beads in the protein in nm. A value of 0 means complete overlap. Generally 0.3 works well, but if there are errors with overlapping atoms then try increasing this. 

-salt: This has been changed to be a comma seperated list of upto 3 values. As a double membrane system will create disjoint compartments the salt concentration can be set for each compartment. if there is no z periodicity then there are 3 compartents for a double membrane system.

-charge_ratio: This is similar to salt but instead indicates how to split the charge across the compartments. Only the first n values will be read where n is the number of compartments in the system. The values need not add to 1 as they will be normalised. 

-zpbc: This indicates if there is z periodicty when determining the number of solvent compartments in the system.

## FAQ
There are currently no frequently asked questions. If you do have any questions or encounter errors that you cannot fix please contact me via my email m.parrag@warwick.ac.uk and I will do my best to provide help.




