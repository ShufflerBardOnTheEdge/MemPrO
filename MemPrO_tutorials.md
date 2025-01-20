# Tutorials

Note that these tutorials assume a Linux OS, however as MemPrO is a python script it should also work on any other OS so long as you can run python. Additionally, PyMOL will be used as the molecular visualisation programm throughout this tutorial, however VMD or any other such program can be used.

## A Basic Example


This first tutorial will run through how to use MemPrO for the most basic usecase, the orientation of a integral membrane protein in a planar membrane. The first step will be to download an exmaple integral membrane protein, let us choose 4G1U from the protein data bank. To download this one can use the fetch commmand in PyMOL followed by saving as a .pdb, further details on this method can be found [here](https://pymolwiki.org/index.php/Fetch). Otherwise go to the [following page on the PDB website](https://www.rcsb.org/structure/4g1u) and download in PDB format.

Now create a folder called "Tutorial1" to contain all the files for this tutorial, and place the downloaded pdb file in there. This structure will include some ligands, these can either be removed or, as it does not affect orientation, ignored. Now in a termail navigate to the folder you just created.

We will now run MemPrO:

>python PATH/TO/MemPrO_Script.py -f 4g1u.pdb -ng 16 -ni 150

Here we are using an initial grid of 16 starting configurations (The higher to better, though 16 is more than enough), and 150 minimisation iterations (Again the higher the better, but 150 is more than enough). 

Once the code has finished running, which with ~16 CPUs should take about 90 seconds, you should find a folder called "Orient" in the Tutorial1 folder. Opening this folder you should find several files and folders. First we will look at orientations.txt, in here we will find all the orientations found by MemPrO. MemPrO can be a little random so your results may differ from what is shown here, but it should be very similar. 

In my case MemPrO found two possible orientations. Each orientation has 6 numbers associated with it. The first is its rank. Let us look at the rank 1 orientation. 
> 1    -71.372    93.75    -71.372    173.346    155273.357

The second number represents the relative potential of the orientation, where 0 would be a protein fully in solvent. So here we can see that rank 1 orientation has lower potential than if the protein were not in the membrane which is a good sign. The third number is the percent of configurations that minimised to this particular orientation, a value of 93.75% indicates a high confidence, though a low value doesn't always mean low confidence. The fourth number should be very close to the second but otherwise can be ignored. The fith number inidcates the calculated depth of the minima, the higher the more stable the orientation is. The sixth and final number is a value calculated from the other values shown by which the orientations are ranked, this value can only be used to compare orientations from the same run of		for i in add_reses:
			ori.add_Reses(i,args.additional_residues_itp_file)
			ori.add_AtomToBeads(args.residue_cg_file.lstrip("/")+"/"+i+".pdb") MemPrO.

Knowing now what these numbers mean hopefully one can see, in this case, the rank 1 orientation is a very stable and deep minima, while any others are very much not. We can now look at orientations.pdb.

orientations.pdb contains all orientations in a single file, ordered by their rank. Looking at each orientation in turn hopefully one can verify that the rank 1 orientation look sensible while the others are indeed more questionable. MemPrO outputs all minima found no matter how bad to provide as much information as possible.

One should be able to a folder for each rank within the Orient folder. Let us look in the folder "Rank_1". In this folder we should find info_rank_1.txt which contains some additional information on the orientation, two images, and the oriented protein PDB file which contains a dummy membrane for visualisation.

The first of the images Z_potential_curve.png shows the potential as the protein moves through the membrane. One should see a clear minima at around 20 angstroms with large peaks either side. This graph also shows this orientation is stable. In cases where orientation is more difficult this graph can be very helpful in identifying problems. The second image curv_potential_curve.png shows potrntial as curvature is varied. A curved membrane is only predicted when the flag -c is used, however this graph will always be calculated, in this case we can see the minima is around 0 curvature, so we can be confident that a planar membrane is indeed sufficient for orientation. We will see cases later where this graph will indicate the need for the use of curvature orientation.

Hopefully, one can see that MemPrO is very easy to use and also provides lots of information about the orientation. The next few tutorials will focus on a few of the more advanced features.

## Double membrane systems

