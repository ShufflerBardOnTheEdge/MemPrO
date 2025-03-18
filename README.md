# MemPrO
Membrane Protein Orientation in Lipid Bilayers. The paper associated with this code can be found here \[link to be added soon\].

## Installation
To install MemPrO run the following (Not yet on PyPi, this will work once the code has been published):
>pip install MemPrO

Otherwise, clone the GitHub repository. Python 3.11.5 or better is required, and the following packages need to be installed
* Jax 0.4.30 (As JAX is in constant development at the time of writing do not use any other versions, as MemPrO may no longer work as intended)
>pip install jax\["cpu"\]==0.4.30
* Matplotlib 3.8.4
>pip install matplotlib==3.8.4

A Martini3 forcefield is also required, this can be downloaded from [here](https://cgmartini.nl/docs/downloads/force-field-parameters/martini3/particle-definitions.html) 

Before running MemPrO some environment variables must be set, for Linux users the following lines can be run.

>export NUM_CPU=N

>export PATH_TO_INSANE=PATH/TO/INSANE4MEMPRO

>export PATH_TO_MARTINI=PATH/TO/MARTINI3

Where N is replaced by the number of CPU you wish to use.

MemPrO is also accessible on Google Colab via the link [ADD LINK]. The Colab will contain instructions for use. (Currently the Colab in WIP and this will be updated)



## Outputs

MemPrO is intended to be run on PDB files, and will output the following files:

  * orientation.txt : This is a file that contains general information about all of the final minima. In general rank 1 is the best orientation, however there are cases where this is not true. orientation.txt can then be used to determine the best orientation.
  * local_minima_orientation.png : This is a representation of how spread out the local minima are. The more white the more confidence can be placed in the rank 1. This is only true in general and there are some cases where this can be safely ignored.

  * local_minima_potential.png : This is a representation of the potential of the final configuration from a given starting configuration. This complements local_minima_orientation.png but in general does not provide much additional information.

  * Rank_{n}: This is a folder that contains all the relevant information for a particular rank. It contains the following files/folders:
    * oriented_rank_{n}.pdb : This is the nth ranked configuration. It will contain the protein (atomistic or coarse grain depending on the input) and a dummy membrane. The rank is based on number of hits (or potential, see flags) .

    * info_rank_{n}.txt : This contains some information about the orientation.

    * CG_System_rank_{n} : This folder contains a CG-system built using Insane4MemPrO and the rank N config. It also contains a topology file. This is outputted when using flags -bg and -bg_args, see the section of flags for syntax.
	
    * PG_potential_curve.png : This is the potential curve for the position of the PG cell wall. This is only outputted when you use flag -pg.
   
    * cross-section_area_curve.png: This shows the cross-sectional area of the segment of the protein that passes through the PG layer. This is only outputted with flag -pg.
	
    * Z_potential_curve.png : This is a graph of potential against Z position of the final orientation. This can give an idea of the kind of minima.
	
    * curv_potential_curve.png : This is a graph of potential against global curvature of membrane with the final orientation. This can indicate of curvature minimisation should be used.
	 
   
MemPrO will also output information about the progress of the orientation to the command line. Additionally MemPrO will output WARNINGs generated by tensorflow, this is due to the version of JAX being used and will hopefully in future be removed. These warnings can in general safely be ignored.

## Flags
MemPro will takes the following flags as input:


-h, --help : This will display a help message with all possible flags and some information on their use.

-f, --file_name : This is the name of the input file for the protein that you wish to orient. It must be a .pdb. In future support for .gro files may be added. The protein can be either atomistic or coarse grained. The code should detect which it is and ignore unknown atom/bead types, however in the case of an error please let me know and send me the file which caused it. It is recommended that the PDB files had no missing atoms as this can cause the orientation to be of lower quality.

-o, --output : This is the name of the ouput directory. The code will default to a folder called Orient in the directory where the code is run. The syntax for this flag is -o "Output_dir/". Note that a backslash is required.

-ni, --iters : This indicates the maximum number of minimisation iteration the code will carry out. The default value of 150 should be enough in most cases. For some systems a smaller value can also work, so if greater speed is needed this can be reduced. Equally if the final output does not seem satisfactory then increasing this value can sometimes fix this, or at least provide information.

-ng, --grid_size : This is the number of initial starting configurations from which to minimise. It defaults to 20. For maximum efficiency this should be a multiple of the number of CPUs being used. Having a greater number of starting configurations will sample the space of minima better, however there is diminishing returns and generally no more than 40 will be needed.

-rank, --rank : This can be either "h", "p" or "auto". "h" ranks the minima by % hits, "p" ranks by potential and "auto" ranks by a value depending on approximate minima depth and potential.

-dm, --dual_membrane : This toggles dual membrane minimisations. This allows the code to split the membrane into a inner and outer membrane and minimise the distance between them. Only use this is you know your protein will span the periplasm or is a gap junction etc.

-ch, --charge : This value corresponds to the partial charge of the (inner) membrane. For -ch and -ch_o below the value corresponds to the average charge on a lipid divided by the average area per lipid. This will give a average charge across a sheet that will represent the bilayer. The value for an E-coli membrane is around -0.008. In general bet preformance is obtained by using a charge of 0.

-ch_o, --charge_outer : This is the charge for the outer membrane only. This will do nothing without -dm.

-mt, --membrane_thickness : This is the initial thickness of the (inner) membrane.

-mt_o, --outer_membrane_thickness : This is the initial thickness of the outer membrane. This will do nothing without -dm.

-mt_opt, --membrane_thickness_optimisation : This toggles membrane thickness optimisation. This cannot be used with curvature minimisation (currently).

-tm, --transmembrane_residues : This indicates which residues are expected to be transmembrane. THe format for this is a comma separated list of ranges e.g. 10-40,50-60. This is intended only for situations where the transmembrane region is known and MemPrO is consistently orienting incorrectly.

-pg --predict_pg_layer : This toggles PG cell wall prediction for dual membrane minimisations. This will output a dummy PG layer where the lowest potential is found and two graphs. The graphs show the potential associated with the placement of the PG layer at a particular Z and the cross-sectional area of the segment of protein going through the PG layer. Due to the nature of the PG layer the lowest potential in not necessarily the biologically correct placement, it is therefore recommended to use these graphs for further information.

-pr, --peripheral : This tells the code to use an alternative method for determining initial insertion depth. The usual method is to place a weighted mean of the hydrophobic residues at the centre of the membrane, this clearly does not work for a peripheral membrane protein or for a protein that does not fully span the membrane. The alternative method scans all possible insertion depths within a range and selects the lowest. This does not work with -dm.

-w, --use_weights : This toggles the use of b-factors to weight the minimisation. This is useful if part of your protein is particularly flexible or poorly predicted by Alphafold. Do not use this if all your b-factors are 0 as currently this will just break with not indication of why (This will be changed soon).

-wb, --write_bfactors : This toggles writing of individual bead potentials into b factors. Not currently working with -c.

-c, --curvature : This toggles global curvature minimisation. This is currently not compatible with -dm. 

-itp, --itp_file : The code makes heavy use of the Martini3 parameters. The path to a Martini 3 itp file should go here. The default location is read from the environment variable PATH_TO_MARTINI.

-bd, --build_system : This should be a number that indicates how many of the final configurations should be build using Insane4MemPrO. This will currently only work if your input file is coarse grain.  

-bd_args, --build_arguments: These are the arguments to pass to Insane4MemPrO. This mostly includes lipid composition, solvent type etc.

## Examples

Before running MemPrO some environment variables must be set, for Linux users the following lines can be run.

>export NUM_CPU=N

>export PATH_TO_INSANE=PATH/TO/INSANE4MEMPRO

>export PATH_TO_MARTINI=PATH/TO/MARTINI3

Where N is replaced by the number of CPU you wish to use, for the examples below 20 were used.

The below will orient input_file.pdb on a grid of 40 starting configurations with 150 minimisation iterations

>python PATH/TO/MemPrO_Script.py -f input_file.pdb -o "Output_dir/" -ng 40 -ni 150


The below will orient input_file.pdb on a grid of 40 starting configurations. It will preform global curvature minimisation alongside usual for 150 iterations. It will write the potential contributions of each bead to the b-factors of the final output.

>python PATH/TO/MemPrO_Script.py -f input_file.pdb -o "Output_dir/" -ng 40 -ni 150 -c -wb 


The below will orient input_file.pdb on a grid of 40 starting configurations with 150 minimisation iterations. It will be minimised with a double membrane system. Once minimised the code will build the top 2 ranks using Insane4MemPrO. The CG system built with insane will have asymmetric membranes, one with POPE,POPG and CARD the other with only LIPA. The CG system will have the same charge and salt concentration in each of the separate compartments created by the presence of two membranes.

>python PATH/TO/MemPrO_Script.py -f input_file.pdb -o "Output_dir/" -ng 40 -ni 150 -dm -bd 2 -bd_args "-negi_c0 CL -posi_c0 NA -sol W -l POPE:7 -l POPG:2 -l CARD:1 -uo LIPA"

A more detailed set of tutorials is available [here](MemPrO_tutorials.md)

## Insane4MemPrO
MemPrO comes with Insane4MemPrO a CG system builder based on Insane. A link to the original Insane code can be found [here](https://github.com/Tsjerk/Insane). Insane4MemPrO allows the user to build more complex systems with up to 2 membranes, curvature, multiple proteins and more. When used with MemPrO directly via the -bd flag the system will be automatically built.


### Flags:

#### I/O related flags:

-f : (Optional) Input protein to build CG system around.

-o : (Required) Output file name.

-p : (Optional) Output topology file

-ct : (Optional) If used this will create a template file. The B factors can be edited to indicate placement of multiple proteins. More detail in examples section.

-in_t: (Optional) If a template created by -ct is being used it is inputted with this flag. This will build a CG system with proteins indicated in -fs being placed where indicated in the template.

-fs : (Optional) A text file with an ordered list of proteins for use in multiple protein placements.

#### Options relating to system size:

-x : (Required) Indicates size of box in x dimension

-y : (Required) Indicates size of box in y dimension

-z : (Required) Indicates size of box in z dimension

#### Membrane/lipid related options:

-l : (Optional) Lipid type and relative abundance (NAME\[:N\]) in membrane (or just lower leaflet if -u used)

-u : (Optional) Lipid type and relative abundance (NAME\[:N\]) in upper leaflet

-lo : (Optional) Lipid type and relative abundance (NAME\[:N\]) in outer membrane (or just lower leaflet if -uo used)

-uo : (Optional) Lipid type and relative abundance (NAME\[:N\]) in outer membrane upper leaflet

-a : (Default: 0.6) Area per lipid (nm\*nm) in membrane (or just lower leaflet if -au used)

-au : (Optional) Area per lipid (nm\*nm) in upper leaflet

-ao : (Optional) Area per lipid (nm\*nm) in outer membrane (or just lower leaflet if -au used)

-auo : (Optional) Area per lipid (nm\*nm) in outer membrane upper leaflet

-ps : (Default 0) Specifies distance between inner and outer membrane. 

-curv : (Default 0,0,1) Curvature of the membrane, consists of 3 comma separated values. The curvature at the middle the curvature as it relaxes back to planar and the direction of the curvature.

-curv_o: (Default 0,0,1) Curvature of outer membrane. see -curv.

-curv_ext: (Default 3) Extent of curved region in the absence of a protein, this also controls the size of the pore if -pore is used.

-micelle: (Optional) Builds a micelle around a protein instead of a bilayer.

-radius: (Optional) Builds a membrane disk with given radius, this can be usefull for simulating nano-disks.

#### Peptidoglycan layer related options.

-pgl: (Optional) Number of PG layers to place at -pgl_z.

-pgl_z: (Optional) Z position of PG layer relative to center of periplasmic space.

-cper: (Optional) Percentage of crosslinks.

-lper: (Optional) Percentage of crosslinks that are between layers.

-per33: (Optional) Percentage of 3-3 crosslinks, all other crosslinks will be 3-4.

-oper: (Optional) Percentage chance of a monomer linking with a oligomer. (Actual change of link is cper*oper)

-gdist: (Default 0.75,4,8.9,0.25,10,45) Distribution of glycan strand lengths. Format as weight 1,standard deviation 1,mean 1,weight 2..., were each triple describes a gaussian. The sum of these forms the distribution

#### Protein related options:

-fudge : (Default 0.3) Fudge factor for allowing lipid-protein overlap

#### Solvent related options:

-sol : (Required) indicates the solvent used.

-sold : (Default 0.5) Indicates how tightly packed the solvent is.

-solr : (Default 0.1) Magnitude of random deviations to solvent positions.

#### Charge related options:

-posi_c0 : (Required) Positive ion type and relative abundance (NAME\[:N\]) in system (Or compartment 0 if -posi_c1/-posi_c2 used) When using multiple membranes disjoint compartments of water may form, these flags allow for each compartment to be neutralised independently and with different ions.

-negi_c0 : (Required) Negative ion type and relative abundance (NAME\[:N\]) in system (Or compartment 0 if -negi_c1/-negi_c2 used).

-posi_c1 : (Optional) Positive ion type and relative abundance (NAME\[:N\]) in compartment 1.

-negi_c1 : (Optional) Negative ion type and relative abundance (NAME\[:N\]) in compartment 1.

-posi_c2 : (Optional) Positive ion type and relative abundance (NAME\[:N\]) in compartment 2.

-negi_c2 : (Optional) Negative ion type and relative abundance (NAME\[:N\]) in compartment 2.

-ion_conc: (Default 0.15,0.15,0.15) Concentration of ions in each compartment.

-charge : (Default "auto") Charge of system. "auto" detects charge automatically.

-charge_ratio : (Optional) When supplied indicates how to split the charge across compartments, otherwise each compartment is neutralised separately.

-zpbc : (Optional) Determines if Z periodicity is used when calculating compartments.





### Examples
These instructions will run through building a CG system with a curved membrane with multiple proteins placed.

Initially a template needs to be created. To do this a command similar to the following should be run:
    
>python PATH/TO/Insane4MemPrO.py -l POPE -sol W -x 10 -y 10 -z 50 -o test.gro -p topol.top -curv 0.1,0.15,1 -fudge 0.3 -curv_ext 6 -ct template.pdb -negi_c0 CL -posi_c0 NA

This will create a membrane with only POPE (-l POPE). The box size if set to be (10,10,50) although as this too small to contain the curvature it is automatically increased. The curvature is described by 0.1,0.15,1 indicating a curvature of 0.1 at the peak, a curvature 0.15 at the base, and the curved region points in positive Z direction. The curved region will maintain the curvature of 0.1 for 6 Angstroms before returning to planar. The name of the template file is template.pdb (-ct template.pdb)
    
After this open template.pdb in PyMOL. The position of the proteins will be indicated by the b-factors in this pdb. Select a single bead where you would like to place a protein and then type the command alter sele,b=1 After this deselect the bead and repeat the process increasing the b-fac value by 1 each time for each protein you would like to add.
    
It is important to increase the b-factors by 1 as this will indicated the order in which the proteins are placed. i.e. The n-th protein will be placed at the bead with b-factor n. Once all b-factors have been changed type the command save template.pdb.
    
Next run the following command in terminal:
    
>python PATH/TO/Insane4MemPrO.py -l POPE -sol W -x 10 -y 10 -z 50 -o test.gro -p topol.top -curv 0.1,0.15,1 -fudge 0.3 -curv_ext 6 -in_t template.pdb -fs prots.txt -negi_c0 CL -posi_c0 NA
    
Ensure all the membrane relevant values are the same. The template file (-in_t template.pdb) should be the one with altered b-factors. -fs takes the .txt mentioned earlier, this should contain the same number of proteins as altered b-facs. This will then build the system.
    
The flag -p will create a rudimentary .top file. The Protein related entries will need to be changed but the other values are correct.

More detailed tutorials are available [here](Insane4MemPrO_tutorials.md)

## My protein didn't orient correctly
If your protein hasn't oriented correctly then there are a number of things you can try:
* Check all orientations, by loading "orientations.pdb" and looking at "orientations.txt". The correct orientation may not have been rank 1.
* Check "curv_potential_curve.png" to see if your protein prefers a curved membrane. If so using the flag "-c" will help.
* Check the pdb file is correct, missing atoms or residues can cause the surface to be incorrectly evaluated. Missing atoms should be fixed before orientation.
* Using the flag "-wb" can sometimes be helpful in determining if atoms are missing. "-wb" will add the potential contribution of each residue as a b-factor which can be viewed in PyMOL.
* If your protein is a peripheral membrane protein using the flag "-pr" will help greatly. In some cases "-pr" will help integral membrane proteins as well.
* MemPrO runs with a membrane charge of 0 by default. Some proteins which associate with the membrane through charge interactions will not orient correctly with 0 charge, for these using the flag "-ch" ("-ch_o" for outer membrane) will set the charge. Values around "-0.005" are a good starting point.


## FAQ
There are currently no frequently asked questions. If you do have any questions or encounter errors that you cannot fix please contact me via my email m.parrag@warwick.ac.uk and I will do my best to provide help.
