# QmcA test case

QmcA is a large SPFH-family oligomer: a ring of transmembrane helices topped by a big soluble cap. It is the test case for the per-leaflet protein footprint (`docs/footprint-design.md`). Built with the current method, its lower leaflet comes out about 9% under-filled beneath the soluble domain.

| File | What it is |
|---|---|
| `QmcA.pdb.gz` | Atomistic model of the oligomer (predicted structure; pLDDT in the B-factor column), 55,978 atoms. Use as martinize2 input. |
| `QmcA-cg.pdb.gz` | Martini 3 coarse-grained model (23,062 beads), used as the MemPrO input below. |
| `protein-cg_rank_1.pdb.gz` | Rank 1 orientation from MemPrO, as passed to Insane4MemPrO (membrane centre at z = 0). |
| `info_rank_1.txt` | MemPrO's information for rank 1: membrane thickness 38 Å, no curvature, and which residues are transmembrane. |

Unzip the files first (`gunzip -k *.gz`).

Reproduce the orientation and build (about 3 min to orient with `-sp -ng 16` on an M4 Mac, plus 30 s to build):

    mempro -f QmcA-cg.pdb -o QmcA_Orient -bd 1 -bd_args "-l POPE:8 -l POPG:2 -sol W -negi_c0 CL -posi_c0 NA"

Build directly from the oriented protein (the box size MemPrO used):

    python mempro/Insane4MemPrO.py -f protein-cg_rank_1.pdb -x 32.11502 -y 32.262005 -z 24.711554 \
        -l POPE:8 -l POPG:2 -sol W -negi_c0 CL -posi_c0 NA -o CG-system.gro -p topol.top

The built system has about 234k beads, so MD validation is best run on a GPU machine.
