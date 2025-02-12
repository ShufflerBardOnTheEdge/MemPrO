#!/usr/bin/env python

import sys,math,random
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import jax
import jax.numpy as jnp

#This is a heavily modified version of Insane, called Insane4MemPrO, written to work with the orientation software MemPrO. 
#Modifications were made by Matyas Parrag.


# Modify insane to take in arbitary lipid definition strings and use them as a template for lipids
# Also take in lipid name 
# Edits: by Helgi I. Ingolfsson (all edits are marked with: # HII edit - lipid definition )





# PROTOLIPID (diacylglycerol), 18 beads
#
# 1-3-4-6-7--9-10-11-12-13-14
#  \| |/  |
#   2 5   8-15-16-17-18-19-20
#

lipidsx = {}
lipidsy = {}
lipidsz = {}
lipidsa = {}



#Detergent

moltype = "type_DDM"
lipidsx[moltype] = (0.27, -0.18, 0.43, 0.17, -0.48, -0.03, 0.42, -0.03, 0.09, 0.03, 0.01)
lipidsy[moltype] = (-0.32, -0.53, -0.29, -0.34, -0.04, 0.29, 0.46, 0.21, 0.21, 0.06, 0.02)
lipidsz[moltype] = (3.58, 3.07, 2.68, 3.10, 3.68, 3.05, 3.78, 3.50, 2.04, 1.06, 0.06)
lipidsa.update({ "DDM":(moltype, "A1 B1 C1 VS1 A2 B2 C2 VS2 L1 L2 L3")})

## Diacyl glycerols
moltype = "lipid"
lipidsx[moltype] = (    0, .5,  0,  0, .5,  0,  0, .5,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1)
lipidsy[moltype] = (    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0)
lipidsz[moltype] = (   10,  9,  9,  8,  8,  7,  6,  6,  5,  4,  3,  2,  1,  0,  5,  4,  3,  2,  1,  0)
lipidsa.update({      # 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
## Phospholipids
    "DTPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A  -   -   -   -  C1B C2B  -   -   -   - "),
    "DLPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A C3A  -   -   -  C1B C2B C3B  -   -   - "),
    "DPPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  C1B C2B C3B C4B  -   - "),
    "DBPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A C3A C4A C5A  -  C1B C2B C3B C4B C5B  - "),
    "POPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B C2B C3B C4B  -   - "),
    "DOPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B D2B C3B C4B  -   - "),
    "DAPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 D1A D2A D3A D4A C5A  -  D1B D2B D3B D4B C5B  - "),
    "DIPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A D2A D3A C4A  -   -  C1B D2B D3B C4B  -   - "),
    "DGPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A D3A C4A C5A  -  C1B C2B D3B C4B C5B  - "),
    "DNPC": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A C3A D4A C5A C6A C1B C2B C3B D4B C5B C6B"),
    "DTPE": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A  -   -   -   -  C1B C2B  -   -   -   - "),
    "DLPE": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A C3A  -   -   -  C1B C2B C3B  -   -   - "),
    "DPPE": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  C1B C2B C3B C4B  -   - "),
    "DBPE": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A C3A C4A C5A  -  C1B C2B C3B C4B C5B  - "),
    "POPE": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B C2B C3B C4B  -   - "),
    "DOPE": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B D2B C3B C4B  -   - "),
    "POPG": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B C2B C3B C4B  -   - "),
    "DOPG": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B D2B C3B C4B  -   - "),
    "POPS": (moltype, " -   -   -  CN0  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B C2B C3B C4B  -   - "),
    "DOPS": (moltype, " -   -   -  CN0  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B D2B C3B C4B  -   - "),
    "DPSM": (moltype, " -   -   -  NC3  -  PO4 AM1 AM2 T1A C2A C3A  -   -   -  C1B C2B C3B C4B  -   - "),
    "DBSM": (moltype, " -   -   -  NC3  -  PO4 AM1 AM2 T1A C2A C3A C4A  -   -  C1B C2B C3B C4B C5B  - "),
    "BNSM": (moltype, " -   -   -  NC3  -  PO4 AM1 AM2 T1A C2A C3A C4A  -   -  C1B C2B C3B C4B C5B C6B"),
    "APG": (moltype,  " -   -  ALA GLX  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B C2B C3B C4B  -   - "),
    "KPG": (moltype,  "LY1 LY2 LY3 GLX  -  PO4 GL1 GL2 C1A D2A C3A C4A  -   -  C1B C2B C3B C4B  -   - "),
# PG for thylakoid membrane   
    "OPPG": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  C1B D2B C3B C4B  -   - "),
# PG for thylakoid membrane of spinach (PPT with a trans-unsaturated bond at sn1 and a triple-unsaturated bond at sn2, 
# and PPG  with a transunsaturated bond at sn1 and a palmitoyl tail at sn2)
    "JPPG": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  D1B C2B C3B C4B  -   - "),
    "JFPG": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A D2A D3A D4A  -   -  D1B C2B C3B C4B  -   - "),
## Monoacylglycerol
    "GMO":  (moltype, " -   -   -   -   -   -  GL1 GL2 C1A C2A D3A C4A C5A  -   -   -   -   -   -   - "),
## Templates using the old lipid names and definitions
  "DHPC.o": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A  -   -   -   -  C1B C2B  -   -   -   - "),
  "DMPC.o": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A C3A  -   -   -  C1B C2B C3B  -   -   - "),
  "DSPC.o": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A C3A C4A C5A  -  C1B C2B C3B C4B C5B  - "),
  "POPC.o": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  C1B C2B D3B C4B C5B  - "),
  "DOPC.o": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A D3A C4A C5A  -  C1B C2B D3B C4B C5B  - "),
  "DUPC.o": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A D2A D3A C4A  -   -  C1B D2B D3B C4B  -   - "),
  "DEPC.o": (moltype, " -   -   -  NC3  -  PO4 GL1 GL2 C1A C2A C3A D4A C5A C6A C1B C2B C3B D4B C5B C6B"),
  "DHPE.o": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A  -   -   -   -  C1B C2B  -   -   -   - "),
  "DLPE.o": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A C3A  -   -   -  C1B C2B C3B  -   -   - "),
  "DMPE.o": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A C3A  -   -   -  C1B C2B C3B  -   -   - "),
  "DSPE.o": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A C3A C4A C5A  -  C1B C2B C3B C4B C5B  - "),
  "POPE.o": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  C1B C2B D3B C4B C5B  - "),
  "DOPE.o": (moltype, " -   -   -  NH3  -  PO4 GL1 GL2 C1A C2A D3A C4A C5A  -  C1B C2B D3B C4B C5B  - "),
  "PPCS.o": (moltype, " -   -   -  NC3  -  PO4 AM1 AM2 C1A C2A C3A C4A  -   -  D1B C2B C3B C4B  -   - "),
  "DOPG.o": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A C2A D3A C4A C5A  -  C1B C2B D3B C4B C5B  - "),
  "POPG.o": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  C1B C2B D3B C4B C5B  - "),
  "DOPS.o": (moltype, " -   -   -  CN0  -  PO4 GL1 GL2 C1A C2A D3A C4A C5A  -  C1B C2B D3B C4B C5B  - "),
  "POPS.o": (moltype, " -   -   -  CN0  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  C1B C2B D3B C4B C5B  - "),
   "CPG.o": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  C1B C2B D3B C4B  -   - "),
   "PPG.o": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A C2A C3A C4A  -   -  D1B C2B C3B C4B  -   - "),
   "PPT.o": (moltype, " -   -   -  GL0  -  PO4 GL1 GL2 C1A D2A D3A D4A  -   -  D1B C2B C3B C4B  -   - "),
  "DSMG.o": (moltype, " -   -   -  C6   C4 C1  GL1 GL2 C1A C2A C3A C4A C5A  -  C1B C2B C3B C4B C5B  - "),
  "DSDG.o": (moltype, "C61 C41 C11 C62 C42 C12 GL1 GL2 C1A C2A C3A C4A C5A  -  C1B C2B C3B C4B C5B  - "),
  "DSSQ.o": (moltype, " -   -   S6 C6   C4 C1  GL1 GL2 C1A C2A C3A C4A C5A  -  C1B C2B C3B C4B C5B  - "),
})


# HII fix for PI templates and new templates PI(s) with diffrent tails, PO-PIP1(3) and POPIP2(4,5)  
#Prototopology for phosphatidylinositol type lipids 5,6,7 are potentail phosphates (PIP1,PIP2 and PIP3)
# 1,2,3 - is the inositol and 4 is the phosphate that links to the tail part.
#  5
#   \
#  6-2-1-4-8--10-11-12-13-14-15
#    |/    |
#  7-3     9--16-17-18-19-20-21 
moltype = "INOSITOLLIPIDS"
lipidsx[moltype] = (   .5,  .5,   0,   0,   1, .5,  0,  0,   .5,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1)
lipidsy[moltype] = (    0,   0,   0,   0,   0,  0,  0,  0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0)
lipidsz[moltype] = (    8,   9,   9,   7,  10, 10, 10,  6,    6,   5,   4,   3,   2,   1,   0,   5,   4,   3,   2,   1,   0)
lipidsa.update({      # 1     2    3    4    5   6   7   8    9    10    11    12    13    14   15    16    17    18    19   20 
    "DPPI": (moltype, " C1   C2   C3    CP   -   -   -  GL1  GL2  C1A  C2A  C3A  C4A   -    -   C1B  C2B  C3B  C4B   -    - "),
    "POPI": (moltype, " C1   C2   C3    CP   -   -   -  GL1  GL2  C1A  D2A  C3A  C4A   -    -   C1B  C2B  C3B  C4B   -    - "),
    "PIPI": (moltype, " C1   C2   C3    CP   -   -   -  GL1  GL2  C1A  D2A  D3A  C4A   -    -   C1B  C2B  C3B  C4B   -    - "),
    "PAPI": (moltype, " C1   C2   C3    CP   -   -   -  GL1  GL2  D1A  D2A  D3A  D4A  C5A   -   C1B  C2B  C3B  C4B   -    - "),
    "PUPI": (moltype, " C1   C2   C3    CP   -   -   -  GL1  GL2  D1A  D2A  D3A  D4A  D5A   -   C1B  C2B  C3B  C4B   -    - "),
    "POP1": (moltype, " C1   C2   C3    CP  P1   -   -  GL1  GL2  C1A  C2A  D3A  C4A   -    -   C1B  C2B  C3B  C4B   -    - "),
    "POP2": (moltype, " C1   C2   C3    CP  P1  P2   -  GL1  GL2  C1A  C2A  D3A  C4A   -    -   C1B  C2B  C3B  C4B   -    - "),
    "POP3": (moltype, " C1   C2   C3    CP  P1  P2  P3  GL1  GL2  C1A  C2A  D3A  C4A   -    -   C1B  C2B  C3B  C4B   -    - "),
## Templates using the old lipid names and definitions
  "PI.o"  : (moltype, " C1   C2   C3    CP   -   -   -  GL1  GL2  C1A  C2A  C3A  C4A   -    -   CU1  CU2  CU3  CU4  CU5   - "),
  "PI34.o": (moltype, " C1   C2   C3    CP PO1 PO2   -  GL1  GL2  C1A  C2A  C3A  C4A   -    -   CU1  CU2  CU3  CU4  CU5   - "),
})


#Prototopology for longer and branched glycosil and ceramide based glycolipids
#
#     17-15-14-16
#         |/
#        13
#         |
# 12-10-9-7-6-4-3-1--18--20-21-22-23-24
#  |/   |/  |/  |/    |
#  11   8   5   2    19--25-26-27-28-29 
moltype = "GLYCOLIPIDS"
lipidsx[moltype] = (    0,  .5,   0,   0,  .5,  0,  0, .5,  0,    0,   .5,    0,    0,    0,   0,    0,    0,    0,   .5,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1)
lipidsy[moltype] = (    0,   0,   0,   0,   0,  0,  0,  0,  0,    0,    0,    0,   .5,    1,   1,    1,    1,    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0)
lipidsz[moltype] = (    6,   7,   7,   8,   9,  9, 10, 11, 11,   12,   13,   13,   10,    9,  10,    8,   11,    5,    5,   4,   3,   2,   1,   0,   4,   3,   2,   1,   0)
lipidsa.update({      # 1     2    3    4    5   6   7   8   9    10    11    12    13    14   15    16    17    18    19   20   21   22   23   24   25   26   27   28   29
    "DPG1": (moltype, "GM1  GM2  GM3  GM4  GM5 GM6 GM7 GM8 GM9  GM10  GM11  GM12  GM13  GM14 GM15  GM16  GM17   AM1   AM2  T1A  C2A  C3A   -    -    -   C1B  C2B  C3B  C4B   -    - "),
    "DXG1": (moltype, "GM1  GM2  GM3  GM4  GM5 GM6 GM7 GM8 GM9  GM10  GM11  GM12  GM13  GM14 GM15  GM16  GM17   AM1   AM2  T1A  C2A  C3A  C4A  C5A   -   C1B  C2B  C3B  C4B  C5B  C6B"),
    "PNG1": (moltype, "GM1  GM2  GM3  GM4  GM5 GM6 GM7 GM8 GM9  GM10  GM11  GM12  GM13  GM14 GM15  GM16  GM17   AM1   AM2  T1A  C2A  C3A   -    -    -   C1B  C2B  C3B  D4B  C5B  C6B"),
    "XNG1": (moltype, "GM1  GM2  GM3  GM4  GM5 GM6 GM7 GM8 GM9  GM10  GM11  GM12  GM13  GM14 GM15  GM16  GM17   AM1   AM2  T1A  C2A  C3A  C4A  C5A   -   C1B  C2B  C3B  D4B  C5B  C6B"),
    "DPG3": (moltype, "GM1  GM2  GM3  GM4  GM5 GM6  -   -   -    -     -     -    GM13  GM14 GM15  GM16  GM17   AM1   AM2  T1A  C2A  C3A   -    -    -   C1B  C2B  C3B  C4B   -    - "),
    "DXG3": (moltype, "GM1  GM2  GM3  GM4  GM5 GM6  -   -   -    -     -     -    GM13  GM14 GM15  GM16  GM17   AM1   AM2  T1A  C2A  C3A  C4A  C5A   -   C1B  C2B  C3B  C4B  C5B  C6B"),
    "PNG3": (moltype, "GM1  GM2  GM3  GM4  GM5 GM6  -   -   -    -     -     -    GM13  GM14 GM15  GM16  GM17   AM1   AM2  T1A  C2A  C3A   -    -    -   C1B  C2B  C3B  D4B  C5B  C6B"),
    "XNG3": (moltype, "GM1  GM2  GM3  GM4  GM5 GM6  -   -   -    -     -     -    GM13  GM14 GM15  GM16  GM17   AM1   AM2  T1A  C2A  C3A  C4A  C5A   -   C1B  C2B  C3B  D4B  C5B  C6B"),
    "DPCE": (moltype, "  -    -    -    -    -   -   -   -   -     -     -     -     -     -    -     -     -   AM1   AM2  T1A  C2A  C3A   -    -   C1B  C2B  C3B  C4B   - "),
    "DPGS": (moltype, " C1   C2   C3    -    -   -   -   -   -     -     -     -     -     -    -     -     -   AM1   AM2  T1A  C2A  C3A   -    -   C1B  C2B  C3B  C4B   - "),
    "DPMG": (moltype, " C1   C2   C3    -    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
    "DPSG": (moltype, " S1   C1   C2   C3    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
    "DPGG": (moltype, "GB2  GB3  GB1  GA1  GA2 GA3   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
#lipids for thylakoid membrane of cyanobacteria: oleoyl tail at sn1 and palmiotyl chain at sn2. SQDG no double bonds
    "OPMG": (moltype, " C1   C2   C3    -    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  D2B  C3B  C4B   - "),
    "OPSG": (moltype, " S1   C1   C2   C3    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  D2B  C3B  C4B   - "),
    "OPGG": (moltype, "GB2  GB3  GB1  GA1  GA2 GA3   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  D2B  C3B  C4B   - "),
#lipids for thylakoid membrane of spinach: for the *T both chains are triple unsaturated and the *G have a triple unsaturated chain at sn1 and a palmitoyl chain at sn2. 
    "FPMG": (moltype, " C1   C2   C3    -    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  D2B  D3B  D4B   - "),
    "DFMG": (moltype, " C1   C2   C3    -    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  D2A  D3A  D4A   -   C1B  D2B  D3B  D4B   - "),
    "FPSG": (moltype, " S1   C1   C2   C3    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  D2B  D3B  D4B   - "),
    "FPGG": (moltype, "GB2  GB3  GB1  GA1  GA2 GA3   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  D2B  D3B  D4B   - "),
    "DFGG": (moltype, "GB2  GB3  GB1  GA1  GA2 GA3   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  D2A  D3A  D4A   -   C1B  D2B  D3B  D4B   - "),
## Templates using the old lipid names and definitions
  "GM1.o" : (moltype, "GM1  GM2  GM3  GM4  GM5 GM6 GM7 GM8 GM9  GM10  GM11  GM12  GM13  GM14 GM15  GM16  GM17   AM1   AM2  C1A  C2A  C3A  C4A  C5A  C1B  C2B  C3B  C4B   - "), 
  "DGDG.o": (moltype, "GB2  GB3  GB1  GA1  GA2 GA3   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
  "MGDG.o": (moltype, " C1   C2   C3    -    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
  "SQDG.o": (moltype, " S1   C1   C2   C3    -   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
  "CER.o" : (moltype, "  -    -    -    -    -   -   -   -   -     -     -     -     -     -    -     -     -   AM1   AM2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
  "GCER.o": (moltype, " C1   C2   C3    -    -   -   -   -   -     -     -     -     -     -    -     -     -   AM1   AM2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
  "DPPI.o": (moltype, " C1   C2   C3    -   CP   -   -   -   -     -     -     -     -     -    -     -     -   GL1   GL2  C1A  C2A  C3A  C4A   -   C1B  C2B  C3B  C4B   - "),
})


moltype = "QUINONES"
lipidsx[moltype] = (    0,  .5,   0,    0,   0,   0,   0,   0,   0,    0,    0,    0)
lipidsy[moltype] = (    0,   0,   0,    0,   0,   0,   0,   0,   0,    0,    0,    0)
lipidsz[moltype] = (    6,   7,   7,   5.5,  5,  4.5,  4,  3.5, 2.5,   2,  1.5,    1)
lipidsa.update({      # 1     2    3    4    5    6    7    8    9    10    11    12
    "PLQ": (moltype, " PLQ3 PLQ2 PLQ1 PLQ4 PLQ5 PLQ6 PLQ7 PLQ8 PLQ9 PLQ10 PLQ11 PLQ12"),
})

moltype = "CYSTEINES"
lipidsx[moltype] = (     1,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1)
lipidsy[moltype] = (     1,   0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,   0,  0,  1,  0,  0,  0,  0,  0,  0)
lipidsz[moltype] = (     6,   7,  6,  6,  5,  4,  3,  2,  1,  0,  5,  4,  3,  2,  1,  0,   7,  6,  6,  5,  4,  3,  2,  1,  0)
lipidsa.update({      # 1    2   3   4   5   6   7   8     9  10  11  12  13  14  15  16   17  18  19  20  21  22  23  24  25
    "CYST": (moltype, " BB  SC1 GL1 GL2 C1A D2A C3A C4A   -   - C1B C2B C3B C4B   -   -   -   -   -  C1C C2C C3C C4C  -   -"),
    "CYSD": (moltype, " BB  SC1 GL1 GL2 C1A D2A C3A C4A   -   - C1B C2B C3B C4B   -   -   -   -   -   -   -   -   -   -   -"),
})

# Prototopology for cardiolipins
#  
#       4-11-12-13-14-15-16
#       |
#   2---3--5--6--7--8--9-10
#  / 
# 1
#  \
#   17-18-20-21-22-23-24-25
#       |
#      19-26-27-28-29-30-31
#
moltype = "CARDIOLIPINS"
lipidsx[moltype] = (   0.5,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1)
lipidsy[moltype] = (     1,   0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,   0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1)
lipidsz[moltype] = (     8,   7,  6,  6,  5,  4,  3,  2,  1,  0,  5,  4,  3,  2,  1,  0,   7,  6,  6,  5,  4,  3,  2,  1,  0,  5,  4,  3,  2,  1,  0)
lipidsa.update({      #  1    2   3   4   5   6   7   8   9  10  11  12  13  14  15  16   17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
    "CDL0": (moltype, "GL5 PO41 GL1 GL2 C1A D2A C3A C4A   -   - C1B C2B C3B C4B   -   - PO42 GL3 GL4 C1C D2C D3C C4C   -   - C1D C2D C3D C4D   -   -"), # Warning not the same names is in .itp
    "CDL1": (moltype, "GL5 PO41 GL1 GL2 C1A D2A C3A C4A   -   - C1B C2B C3B C4B   -   - PO42 GL3 GL4 C1C D2C D3C C4C   -   - C1D C2D C3D C4D   -   -"), # Warning not the same names is in .itp
    "CDL2": (moltype, "GL5 PO41 GL1 GL2 C1A D2A C3A C4A   -   - C1B C2B C3B C4B   -   - PO42 GL3 GL4 C1C D2C D3C C4C   -   - C1D C2D C3D C4D   -   -"), # Warning not the same names is in .itp 
    "CL4P": (moltype, "GL5 PO41 GL1 GL2 C1A D2A C3A C4A   -   - C1B C2B C3B C4B   -   - PO42 GL3 GL4 C1C D2C C3C C4C   -   - C1D C2D C3D C4D   -   -"), 
    "CL4M": (moltype, "GL5 PO41 GL1 GL2 C1A D2A C3A   -   -   - C1B C2B C3B   -   -   - PO42 GL3 GL4 C1C D2C C3C   -   -   - C1D C2D C3D   -   -   -"), 
    "CARD": (moltype, "GL0 PO1  GL1 GL2 C1A D2A C3A C4A   -   - C1B C2B C3B C4B   -   - PO2 GL3 GL4 C1C D2C C3C C4C   -   - C1D C2D C3D C4D   -   -"),
## Templates using the old lipid names and definitions
  "CL4.o" : (moltype, "GL5 PO41 GL1 GL2 C1A C2A D3A C4A C5A   - C1B C2B D3B C4B C5B   - PO42 GL3 GL4 C1C C2C D3C C4C C5C   - C1D C2D D3D C4D C5D   -"), 
  "CL4O.o": (moltype, "GL5 PO41 GL1 GL2 C1A C2A D3A C4A C5A   - C1B C2B D3B C4B C5B   - PO42 GL3 GL4 C1C C2C D3C C4C C5C   - C1D C2D D3D C4D C5D   -"),
})


moltype = "LPS"
lipidsx[moltype] = (-7.97, -8.52, -7.94, -8.14, -8.52, -7.87, -8.75, -8.77, -8.61, -8.58, -8.58, -7.69, -7.53, -7.47, -7.58, -7.45, -7.46, -7.37, -7.42, -6.88, -7.56, -7.11, -7.53, -7.26, -7.32, -7.42, -7.54, -8.07, -8.19, -8.44, -7.19, -6.72, -6.82, -6.81, -7.53, -7.74, -7.88, -7.67, -7.41, -7.86, -7.65, -7.51, -7.25, -7.68, -7.39, -7.84, -7.63, -7.18, -7.29)
lipidsy[moltype] = (0.73, 0.52, 0.60, 0.62, 0.07, 0.51, 0.27, 0.54, 0.70, 0.96, 1.25, 0.51, 0.52, 0.58, 0.76, 0.49, 0.22, 0.10, 0.27, -0.31, -0.04, 0.72, -0.20, -0.29, -0.41, -0.51, -0.60, -0.35, -0.32, -0.13, 1.00, 1.26, 1.42, 1.75, 1.32, 1.56, 1.78, 0.33, 0.52, 0.52, 0.45, 0.85, 0.02, 0.31, -0.05, 0.19, 0.15, 0.58, -0.14)
lipidsz[moltype] = (5.76, 5.32, 5.25, 5.44, 5.95, 4.51, 4.75, 4.25, 3.59, 2.78, 1.97, 4.00, 3.31, 2.47, 1.63, 6.12, 6.82, 6.30, 6.41, 6.10, 5.61, 5.60, 5.04, 4.41, 3.69, 2.97, 2.11, 4.54, 3.72, 2.80, 5.03, 4.58, 3.76, 2.85, 4.46, 3.68, 2.75, 7.59, 8.18, 8.03, 7.93, 7.36, 8.26, 8.81, 9.30, 9.27, 9.13, 8.95, 9.81)
lipidsa.update({ "KLA":(moltype, "ZM1 ZM2 ZM3 VS1 PO1 ZM4 ZM5 CF C1F C2F C3F CE C1E C2E C3E GM1 GM2 GM3 VS2 PO2 GM4 GM5 CAB C1A C2A C3A C4A C1B C2B C3B CCD C1C C2C C3C C1D C2D C3D KR1 KR2 KR3 VS3 KS1 KS2 KR4 KR5 KR6 VS4 KS3 KS4")})

moltype = "LPS2"
lipidsx[moltype] = (     0,   0,  1,  0,  0,  0,  0,  1,  2,  2,  1,  2,  1,  1,  0,  0,  0,  0,  1,  1,  1,   1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  2,  2,  2,  3,  3,  3,  1,  2,  1,  0,  0,  0, 0,  0,   0,  1,  0,  0,  1,  2,  2,  2,  2,  0,  1,  0,  0,  1,  1,  1,  1,  2,  1,  1,  1,  0,  0,  0  )
lipidsy[moltype] = (     0,   1,  0,  1,  0,  0,  0,  1,  0,  1,  1,  2,  1,  1,  1,  1,  1,  1,  0,  0,  0,   0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  2,  1,  1,  1,  1,  2,  3,  2,  2,  2,  3,  3,  3,  1,  1,  1,  1,  0,  1, 0,  1,   1,  0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  1,  1,  1,  0  )
lipidsz[moltype] = (     5,   5,  5,  6,  5,  6,  7,  5,  5,  5,  5,  5,  7,  8,  4,  3,  2,  1,  4,  3,  2,  1,   0,  4,  3,  2,  1,  0,  3,  2,  1,  0,  4,  4,  3,  2,  1,  0,  3,  3,  2,  1,  0,  3,  2,  1,  9,  9,  8, 10, 11, 11, 12, 11, 12, 12, 13, 13, 13, 13, 13, 12, 14, 14, 14, 15, 16, 16, 15, 15, 16, 16, 17, 18, 17, 17, 18, 17  )
lipidsa.update({      #  1    2   3   4   5   6   7   8   9  10  11  12  13  14  15  16   17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  )
    "PGIN": (moltype, " GM1  GM2 GM3 GM4 PO1  -   -  GM6 GM7 GM8 GM9  -   -   -  GL1 C1A C2A C3A GL2 C1B C2B C3B C4B GL3  C1C C2C C3C C4C C1D C2D C3D C4D GL4  -  C1E C2E C3E  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "), 
    "BPER": (moltype, " GM1  GM2 GM3 GM4 PO1  -   -  GM6 GM7 GM8 GM9 PO2  -   -  GL1 C1A C2A  -  GL2 C1B C2B  -   -  GL3  C1C C2C C3C  -  C1D C2D C3D  -  GL4  -  C1E C2E C3E  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "), 
    "BFRA": (moltype, " GM1  GM2 GM3 GM4 PO1  -   -  GM6 GM7 GM8 GM9  -   -   -  GL1 C1A C2A C3A GL2 C1B C2B C3B  -  GL3  C1C C2C C3C C4C C1D C2D C3D  -  GL4  -  C1E C2E C3E C4E  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "), 
    "CTRA": (moltype, " GM1  GM2 GM3 GM4 PO1  -   -  GM6 GM7 GM8 GM9 PO2  -   -  GL1 C1A C2A C3A GL2 C1B C2B C3B  -  GL3  C1C C2C C3C C4C C1D C2D C3D C4D GL4  -  C1E C2E C3E  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "), 
    "CJEJ": (moltype, " GM1  GM2 GM3 GM4 PO1  -   -  GM6 GM7 GM8 GM9 PO2  -   -  GL1 C1A C2A  -  GL2 C1B C2B C3B  -  GL3  C1C C2C C3C  -  C1D C2D C3D C4D GL4 GL5 C1E C2E C3E  -  C1F  -  C2F C3F C4F  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "), 
    "HPYL": (moltype, " GM1  GM2 GM3 GM4 PO1  -   -  GM6 GM7 GM8 GM9  -   -   -  GL1 C1A C2A C3A GL2 C1B C2B C3B C4B GL3  C1C C2C C3C C4C C1D C2D C3D C4D  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "), 
    "NMEN": (moltype, " GM1  GM2 GM3 GM4 PO1 PO2 QD1 GM6 GM7 GM8 GM9 PO3 PO4 QD2 GL1 C1A C2A C3A  -  C1B C2B C3B  -  GL2  C1C C2C C3C  -  GL3 C1D C2D C3D  -   -  C1E C2E C3E  -  GL4  -  C1F C2F C3F  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "), 
    "SMIN": (moltype, " GM1  GM2 GM3 GM4 PO1  -   -  GM6 GM7 GM8 GM9 PO2  -   -  GL1 GL2 C1A C2A  -  C1B C2B C3B C4B GL3  C1C C2C C3C  -  GL4 C1D C2D C3D  -   -  C1E C2E C3E  -  GL5 GL6 C1F C2F C3F C1G C2G C3G  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "), 
})

# Prototopology for mycolic acid(s)
#
#  1--2--3--4--5--6--7--8
#                       |
# 16-15-14-13-12-11-10--9
# |
# 17-18-19-20-21-22-23-24
#                     /
# 32-31-30-29-28-27-25-26
#

moltype = "MYCOLIC ACIDS"
lipidsx[moltype] = (      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,    1,    1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  1,   1,   1)
lipidsy[moltype] = (      0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,    1,    1,    1,    1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,  0,   0,   0)
lipidsz[moltype] = (      7,   6,   5,   4,   3,   2,   1,   0,   0,   1,   2,   3,   4,   5,   6,    7,    7,    6,    5,   4,   3,   2,   1,   0,   1,   0,   2,   3,   4,  5,   6,   7)
lipidsa.update({        # 1    2    3    4    5    6    7    8    9   10   11   12   13   14   15    16    17    18    19   20   21   22   23   24   25   26   27   28   29   30   31   32
    "AMA":   (moltype, "  -    -    -  C1A  C2A  C3A  C4A  C5A  M1A  C1B  C2B  C3B  C4B    -    -     -     -     -   M1B  C1C  C2C  C3C    -    -  COH  OOH  C1D  C2D  C3D  C4D  C5D  C6D"),
    "AMA.w": (moltype, "  -    -    -  C1A  C2A  C3A  C4A  C5A  M1A  C1B  C2B  C3B  C4B    -    -     -     -     -   M1B  C1C  C2C  C3C    -    -  COH  OOH  C1D  C2D  C3D  C4D  C5D  C6D"),
    "KMA":   (moltype, "  -    -    -  C1A  C2A  C3A  C4A  C5A  M1A  C1B  C2B  C3B  C4B    -    -     -     -     -   M1B  C1C  C2C  C3C    -    -  COH  OOH  C1D  C2D  C3D  C4D  C5D  C6D"),
    "MMA":   (moltype, "  -    -    -  C1A  C2A  C3A  C4A  C5A  M1A  C1B  C2B  C3B  C4B    -    -     -     -     -   M1B  C1C  C2C  C3C    -    -  COH  OOH  C1D  C2D  C3D  C4D  C5D  C6D"),
})


# Sterols
moltype = "sterol"
lipidsx[moltype] = (     0,  0,  0,  0,  0, 0,   0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0)
lipidsy[moltype] = (     0,  0,  0,  0,  0, 0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0)
lipidsz[moltype] = (     0,  0,  0,  0,  0, 0, 5.3,4.5,3.9,3.3, 3 ,2.6,1.4,  0,  0,  0,  0,  0)
lipidsa.update({
    "CHOL": (moltype, " -   -   -   -   -   -  ROH  R1  R2  R3  R4  R5  C1  C2  -   -   -   - "),
    "ERGO": (moltype, " -   -   -   -   -   -  ROH  R1  R2  R3  R4  R5  C1  C2  -   -   -   - "),
})


# Hopanoids
moltype = "Hopanoids"
lipidsx[moltype] = (     0,  0,  0,  0, 0.5,-0.5,   0,   0, 0.5, 0.5,   0,   0,   0,   0,  0,  0,  0,  0)
lipidsy[moltype] = (     0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0)
lipidsz[moltype] = (     0,  0,  0,  0, 0.5, 1.4, 2.6,   3, 3.3, 3.9, 4.5, 5.0, 5.5, 6.0,  0,  0,  0,  0) 
lipidsa.update({
    "HOPR": (moltype, " -   -   -   R1   R2   R3   R4   R5   R6   R7   R8   -    -    -    -   -   -   - "),
    "HHOP": (moltype, " -   -   -   R1   R2   R3   R4   R5   R6   R7   R8   C1   -    -    -   -   -   - "),
    "HDPT": (moltype, " -   -   -   R1   R2   R3   R4   R5   R6   R7   R8   C1   -    -    -   -   -   - "),
    "HBHT": (moltype, " -   -   -   R1   R2   R3   R4   R5   R6   R7   R8   C1   C2   C3   -   -   -   - "),
})


#Higher lipids in a different order
moltype = "Isoprenyls-m3-further"
lipidsx[moltype] = (  0, 0.5,  0,  0,  0,  0,  0,0.5,  1,   1,  1,  1,  1,   1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5, 1,  1,  0,  0,  0,  1,  0,  1,  2,  1, 1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,1.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5,  1,  1,  0,  0,  0,  1,  0,  1,  2,  1,1.5,  2,  2,  2,  1,  1,  2,2.5,  2,2.5)
lipidsy[moltype] = (  0,   0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,   1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5, 1,  0,  1,  0,  0,  0,  1,  1,  1,  2, 1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5,  1,  0,  1,  0,  0,  0,  1,  1,  1,  2,1.5,  2,  2,  1,  2,  2,  1,1.5,  2,2.5)
lipidsz[moltype] = (  8,   7,  6,  5,  4,  3,  2,  1,  2,   3,  4,  5,  6,  10, 10, 10, 10,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 11, 11, 11, 11, 11, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 17, 17, 17, 17, 17, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 23, 23, 23, 23, 23, 25, 25, 25, 25, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 26, 26, 26, 26, 26, 28, 28, 28, 28, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 29, 29, 29, 29, 29, 31, 31, 31, 31, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 32, 32, 32, 32, 32, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 35, 35, 35, 35, 35, 37, 37, 37, 37, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 38, 38, 38, 38, 38, 40, 40, 40, 40, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 41, 41, 41, 41, 41)
lipidsa.update({    # 1    2   3   4   5   6   7   8   9   10  11  12  13   14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30 31  32  33   34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
#    "LIP2": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20  -  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LIP4": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LIP6": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LIP8": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LI10": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LI12": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LI14": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LI16": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LI18": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LI20": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   - "),
    "LI22": (moltype,"PO1 PO2 CP1 CP2 CP3 CP4 CP5 CP6 CP7 CP8 CP9 CP10 CP11 B1  B2  B3  B4  B5  B6  B7  B8  B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31 B32 B33 B34 B35 B36 B37 B38 B39 B40 B41 B42 B43 B44 B45 B46 B47 B48 B49 B50 B51 B52 B53 B54 B55 B56 B57 B58 B59 B60 B61 B62 B63 B64 B65 B66 B67 B68 B69 B70 B71 B72 B73 B74 B75 B76 B77 B78 B79 B80 B81 B82 B83 B84 B85 B86 B87 B88 B89 B90 B91 B92 B93 B94 B95 B96 B97 B98 B99 B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20"),
})


# Lists for automatic charge determination
charges = {"HPPA":-1,"4HPA":-1,"UDP2":-2,"UDP1":-1, "GLYP":-1, "GLYM":-1, "ARG":1, "LYS":1, "ASP":-1, "GLU":-1, "DOPG":-1, "POPG":-1, "DOPS":-1, "POPS":-1, "DSSQ":-1, "KPG":+1, "LIPA":-2, "PGIN":-1, "REMP":-6, "RAMP":-10, "OANT":-10, "CARD":-2, "A2P2":-1, "APM2":-1, "APM6":-1, "LIP2":-4, "LIP4":-6, "LIP6":-8, "LIP8":-10, "LI10":-12, "LI12":-14, "LI14":-16, "LI16":-18, "LI18":-20, "LI20":-22, "LI22":-24,"UPEP":-1,"SPEP":-2,"UUPEP":-1,"KLA":-6}
ion_charges = {"NA" :1,"CL":-1,"CA":2}


a,  b  = math.sqrt(2)/20, math.sqrt(2)/60
ct, st = math.cos(math.pi*109.47/180), math.sin(math.pi*109.47/180) # Tetrahedral

# Get a set of coordinates for a solvent particle with a given name
# Dictionary of solvents; First only those with multiple atoms
solventParticles = {
    "PW":       (("W",(-0.07,0,0)),                          # Polarizable water
                 ("WP",(0.07,0,0)),
                 ("WM",(0.07,0,0))),
    "BMW":      (("C",(0,0,0)),
                 ("Q1",(0.12,0,0)),
                 ("Q2",(-0.06,math.cos(math.pi/6)*0.12,0))), # BMW water
    "SPC":      (("OW",(0,0,0)),                             # SPC
                 ("HW1",(0.01,0,0)),
                 ("HW2",(0.01*ct,0.01*st,0))),
    "SPCM":     (("OW",(0,0,0)),                             # Multiscale/Martini SPC 
                 ("HW1",(0.01,0,0)),
                 ("HW2",(0.01*ct,0.01*st,0)),
                 ("vW",(0,0,0))),
    "FG4W":     (("OW1",(a,a,a)),                            # Bundled water
                 ("HW11",(a,a-b,a-b)),
                 ("HW12",(a,a+b,a+b)),
                 ("OW2",(a,-a,-a)),
                 ("HW21",(a-b,-a,-a+b)),
                 ("HW22",(a+b,-a,-a-b)),
                 ("OW3",(-a,-a,a)),
                 ("HW31",(-a,-a+b,a-b)),
                 ("HW32",(-a,-a-b,a+b)),
                 ("OW4",(-a,a,-a)),
                 ("HW41",(-a+b,a,-a+b)),
                 ("HW42",(-a-b,a,-a-b))),
    "FG4W-MS":  (("OW1",(a,a,a)),                            # Bundled water, multiscaled
                 ("HW11",(a,a-b,a-b)),
                 ("HW12",(a,a+b,a+b)),
                 ("OW2",(a,-a,-a)),
                 ("HW21",(a-b,-a,-a+b)),
                 ("HW22",(a+b,-a,-a-b)),
                 ("OW3",(-a,-a,a)),
                 ("HW31",(-a,-a+b,a-b)),
                 ("HW32",(-a,-a-b,a+b)),
                 ("OW4",(-a,a,-a)),
                 ("HW41",(-a+b,a,-a+b)),
                 ("HW42",(-a-b,a,-a-b)),
                 ("VZ",(0,0,0))),
    "GLUC":     (("B1",(-0.11, 0,   0)),
                 ("B2",( 0.05, 0.16,0)),
                 ("B3",( 0.05,-0.16,0))),
    "FRUC":     (("B1",(-0.11, 0,   0)),
                 ("B2",( 0.05, 0.16,0)),
                 ("B3",( 0.05,-0.16,0))),
    "SUCR":     (("B1",(-0.25, 0.25,0)),
                 ("B2",(-0.25, 0,   0)),
                 ("B3",(-0.25,-0.25,0)),
                 ("B4",( 0.25, 0,   0)),
                 ("B5",( 0.25, 0.25,0)),
                 ("B6",( 0.25,-0.25,0))),
    "MALT":     (("B1",(-0.25, 0.25,0)),
                 ("B2",(-0.25, 0,   0)),
                 ("B3",(-0.25,-0.25,0)),
                 ("B4",( 0.25, 0,   0)),
                 ("B5",( 0.25, 0.25,0)),
                 ("B6",( 0.25,-0.25,0))),
    "CELL":     (("B1",(-0.25, 0.25,0)),
                 ("B2",(-0.25, 0,   0)),
                 ("B3",(-0.25,-0.25,0)),
                 ("B4",( 0.25, 0,   0)),
                 ("B5",( 0.25, 0.25,0)),
                 ("B6",( 0.25,-0.25,0))),
    "KOJI":     (("B1",(-0.25, 0.25,0)),
                 ("B2",(-0.25, 0,   0)),
                 ("B3",(-0.25,-0.25,0)),
                 ("B4",( 0.25, 0,   0)),
                 ("B5",( 0.25, 0.25,0)),
                 ("B6",( 0.25,-0.25,0))),
    "SOPH":     (("B1",(-0.25, 0.25,0)),
                 ("B2",(-0.25, 0,   0)),
                 ("B3",(-0.25,-0.25,0)),
                 ("B4",( 0.25, 0,   0)),
                 ("B5",( 0.25, 0.25,0)),
                 ("B6",( 0.25,-0.25,0))),
    "NIGE":     (("B1",(-0.25, 0.25,0)),
                 ("B2",(-0.25, 0,   0)),
                 ("B3",(-0.25,-0.25,0)),
                 ("B4",( 0.25, 0,   0)),
                 ("B5",( 0.25, 0.25,0)),
                 ("B6",( 0.25,-0.25,0))),
    "LAMI":     (("B1",(-0.25, 0.25,0)),
                 ("B2",(-0.25, 0,   0)),
                 ("B3",(-0.25,-0.25,0)),
                 ("B4",( 0.25, 0,   0)),
                 ("B5",( 0.25, 0.25,0)),
                 ("B6",( 0.25,-0.25,0))),
    "TREH":     (("B1",(-0.25, 0.25,0)),
                 ("B2",(-0.25, 0,   0)),
                 ("B3",(-0.25,-0.25,0)),
                 ("B4",( 0.25, 0,   0)),
                 ("B5",( 0.25, 0.25,0)),
                 ("B6",( 0.25,-0.25,0))),
# Loose aminoacids
    "GLY":      (("BB", ( 0,    0,   0)),),
    "ALA":      (("BB", ( 0,    0,   0)),),
    "ASN":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))), 
    "ASP":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "GLU":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "GLN":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "LEU":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "ILE":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "VAL":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "SER":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "THR":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "CYS":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "MET":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "LYS":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "PRO":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "HYP":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",(-0.25, 0,   0))),
    "ARG":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",( 0,    0,   0)),
                 ("SC2",(-0.25, 0.125, 0))),
    "PHE":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",( 0,    0,   0)),
                 ("SC2",(-0.25,-0.125, 0)),
                 ("SC3",(-0.25, 0.125, 0))),
    "TYR":      (("BB", ( 0.25, 0,   0)),
                 ("SC1",( 0,    0,   0)),
                 ("SC2",(-0.25,-0.125, 0)),
                 ("SC3",(-0.25, 0.125, 0))),
    "TRP":      (("BB", ( 0.25, 0.125, 0)),
                 ("SC1",( 0.25, 0,   0)),
                 ("SC2",( 0,   -0.125, 0)),
                 ("SC3",( 0,    0.125, 0)),
                 ("SC4",(-0.25, 0,     0))),
    }

# Update the solvents dictionary with single atom ones
for s in ["W","NA","CL","Mg","K","BUT"]:
    solventParticles[s] = ((s,(0,0,0)),)

# Apolar amino acids nd stuff for orienting proteins in membrane 
apolar = "ALA CYS PHE ILE LEU MET VAL TRP PLM CLR".split()

## PRIVATE PARTS FROM THIS POINT ON ##

S = str
F = float
I = int
R = random.random

def vector(v):
    if type(v) == str and "," in v:
        return [float(i) for i in v.split(",")]
    return float(v)

def vvadd(a,b):    
    if type(b) in (int,float):
        return [i+b for i in a]
    return [i+j for i,j in list(zip(a,b))]

def vvsub(a,b):
    if type(b) in (int,float):
        return [i-b for i in a]
    return [i-j for i,j in list(zip(a,b))]

def isPDBAtom(l):
    return l.startswith("ATOM") or l.startswith("HETATM")

def pdbAtom(a):
    ##01234567890123456789012345678901234567890123456789012345678901234567890123456789
    ##ATOM   2155 HH11 ARG C 203     116.140  48.800   6.280  1.00  0.00
    ## ===>   atom name,   res name,     res id, chain,       x,            y,             z       
    return (S(a[12:16]),S(a[17:21]),I(a[22:26]),a[21],F(a[30:38])/10,F(a[38:46])/10,F(a[46:54])/10)

d2r = 3.14159265358979323846264338327950288/180
def pdbBoxRead(a):
    # Convert a PDB CRYST1 entry to a lattice definition.
    # Convert from Angstrom to nanometer
    fa, fb, fc, aa, ab, ac = [float(i) for i in a.split()[1:7]]
    ca, cb, cg, sg         = math.cos(d2r*aa), math.cos(d2r*ab), math.cos(d2r*ac) , math.sin(d2r*ac)
    wx, wy                 = 0.1*fc*cb, 0.1*fc*(ca-cb*cg)/sg
    wz                     = math.sqrt(0.01*fc*fc - wx*wx - wy*wy)
    return [0.1*fa, 0, 0, 0.1*fb*cg, 0.1*fb*sg, 0, wx, wy, wz]

def groAtom(a):
    #012345678901234567890123456789012345678901234567890
    #    1PRN      N    1   4.168  11.132   5.291
    ## ===>   atom name,   res name,     res id, chain,       x,          y,          z       
    return (S(a[10:15]), S(a[5:10]),   I(a[:5]), " ", F(a[20:28]),F(a[28:36]),F(a[36:44]))

def groBoxRead(a):    
    b = [F(i) for i in a.split()] + 6*[0] # Padding for rectangular boxes
    return b[0],b[3],b[4],b[5],b[1],b[6],b[7],b[8],b[2]

def readBox(a):
    x = [ float(i) for i in a.split(",") ] + 6*[0]
    if len(x) == 12: # PDB format
        return pdbBoxRead("CRYST1 "+" ".join([str(i) for i in x]))
    else:            # GRO format
        return x[0],x[3],x[4],x[5],x[1],x[6],x[7],x[8],x[2]

class Structure:
    def __init__(self,filename=None):
        self.title   = ""
        self.atoms   = []
        self.coord   = []
        self.rest    = []
        self.box     = []        
        self._center = None

        if filename:
            lines = open(filename).readlines()
            # Try extracting PDB atom/hetatm definitions
            self.rest   = []
            self.atoms  = [pdbAtom(i) for i in lines if isPDBAtom(i) or self.rest.append(i)]
            if self.atoms:             
                # This must be a PDB file
                self.title = "THIS IS INSANE!\n"
                for i in self.rest:
                    if i.startswith("TITLE"):
                        self.title = i
                self.box   = [0,0,0,0,0,0,0,0,0]
                for i in self.rest:
                    if i.startswith("CRYST1"):
                        self.box = pdbBoxRead(i)                
            else:
                # This should be a GRO file
                self.atoms = [groAtom(i) for i in lines[2:-1]]
                self.rest  = [lines[0],lines[1],lines[-1]]
                self.box   = groBoxRead(lines[-1])
                self.title = lines[0]
            self.coord = [i[4:7] for i in self.atoms]
            self.center()

    def __nonzero__(self):
        return bool(self.atoms)

    def __len__(self):
        return len(self.atoms)

    def __iadd__(self,s):
        for i in range(len(self)):
            self.coord[i] = vvadd(self.coord[i],s)
        return self

    def center(self,other=None):
        if not self._center:
            self._center = [ sum(i)/len(i) for i in list(zip(*self.coord))]
        if other:
            s = vvsub(other,self._center)
            for i in range(len(self)):
                self.coord[i] = vvadd(self.coord[i],s)
            self._center = other
            return s # return the shift
        return self._center

    def diam(self):
        if self._center != (0,0,0):
            self.center((0,0,0))
        return 2*math.sqrt(max([i*i+j*j+k*k for i,j,k in self.coord]))

    def diamxy(self):
        if self._center != (0,0,0):
            self.center((0,0,0))
        return 2*math.sqrt(max([i*i+j*j for i,j,k in self.coord]))

    def fun(self,fn):
        return [fn(i) for i in list(zip(*self.coord))]

# Mean of deviations from initial value
def meand(v):
    return sum([i-v[0] for i in v])/len(v)

# Sum of squares/crossproducts of deviations
def ssd(u,v):
    return sum([(i-u[0])*(j-v[0]) for i,j in list(zip(u,v))])/(len(u)-1)

# Parse a string for a lipid as given on the command line (LIPID[:NUMBER]) 
def parse_mol(x):
    l = x.split(":")
    return l[0], len(l) == 1 and 1 or float(l[1])

## MIJN EIGEN ROUTINE ##

# Quite short piece of code for diagonalizing symmetric 3x3 matrices :)

# Analytic solution for third order polynomial
def solve_p3( a, b, c ):
    Q,R,a3 = (3*b-a**2)/9.0, (-27*c+a*(9*b-2*a**2))/54.0, a/3.0
    if Q**3 + R**2:
        t,R13 = math.acos(R/math.sqrt(-Q**3))/3, 2*math.sqrt(-Q)
        u,v,w = math.cos(t), math.sin(t+math.pi/6), math.cos(t+math.pi/3)
        return R13*u-a3, -R13*v-a3, -R13*w-a3
    else:
        R13   = math.sqrt3(R)
        return 2*R13-a3, -R13-a3, -R13-a3

# Normalization of 3-vector
def normalize(a):
    f = 1.0/math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
    return f*a[0],f*a[1],f*a[2]

# Eigenvectors for a symmetric 3x3 matrix:
# For symmetric matrix A the eigenvector v with root r satisfies
#   v.Aw = Av.w = rv.w = v.rw
#   v.(A-rI)w = v.Aw - v.rw = 0 for all w
# This means that for any two vectors p,q the eigenvector v follows from:
#   (A-rI)p x (A-rI)q
# The input is var(x),var(y),var(z),cov(x,y),cov(x,z),cov(y,z)
# The routine has been checked and yields proper eigenvalues/-vectors
def mijn_eigen_sym_3x3(a,d,f,b,c,e):
    a,d,f,b,c,e=1,d/a,f/a,b/a,c/a,e/a
    b2, c2, e2, df = b*b, c*c, e*e, d*f
    roots = list(solve_p3(-a-d-f, df-b2-c2-e2+a*(f+d), a*e2+d*c2+f*b2-a*df-2*b*c*e))
    roots.sort(reverse=True)
    ux, uy, uz = b*e-c*d, b*c-a*e, a*d-b*b
    u = (ux+roots[0]*c,uy+roots[0]*e,uz+roots[0]*(roots[0]-a-d))
    v = (ux+roots[1]*c,uy+roots[1]*e,uz+roots[1]*(roots[1]-a-d))
    w = u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0] # Cross product
    return normalize(u),normalize(v),normalize(w),roots

# Very simple option class
class Option:
    def __init__(self,func=str,num=1,default=None,description=""):
        self.func        = func
        self.num         = num
        self.value       = default
        self.description = description
    def __nonzero__(self): 
        return self.value != None
    def __str__(self):
        return self.value and str(self.value) or ""
    def setvalue(self,v):
        if len(v) == 1:
            self.value = self.func(v[0])
        else:
            self.value = [ self.func(i) for i in v ]


def SphereGrid(n):
    grid_points = []
    for k in range(n):
        y = (2.*k+1)/n-1
        phi = k*2.3999632297286531
        r = math.sqrt(1-y*y)
        grid_points.append([math.cos(phi)*r, math.sin(phi)*r,y])
    return grid_points

#All functions that follow return a set of 3d coordinates and normals. This is all that is needs for later in the code
#where the actaul lipids are built

#This function creates a set of points with inverted curvature. The points are placed such that they are evenly distributed.    
def InvSphereGrid(den,inner_rad,start_ang,orad):
    grid_points = []
    direcs = []
    frden = int(-orad*den*2*(start_ang-np.pi/2)/np.pi)+1
    rings = np.linspace(start_ang,np.pi/2,frden)
    for xr,r in enumerate(rings[1:]):
        rad = inner_rad+(1-np.cos(r))*orad
        rden = int(rad*den)+1
        new_ring = np.linspace(0,np.pi*2,4*rden)[:-1]
        rrand = random.random()
        for nr in new_ring:
            ang1 = xr*rrand+nr
            grid_points.append([np.cos(ang1)*rad,np.sin(ang1)*rad,orad*np.sin(r)])
            direcs.append([np.cos(ang1)*np.cos(r),np.sin(ang1)*np.cos(r),np.sin(r)])
    return grid_points,direcs
 
#Same as above but for the usual curvature   
def SphereGridN(den,end_ang,orad):
    grid_points = []
    direcs = []
    frden = int(orad*den*2*(end_ang)/np.pi)+1
    rings = np.linspace(0,end_ang,frden)
    for xr,r in enumerate(rings):
        rad = np.sin(r)*orad
        rden = int(rad*den)
        new_ring = np.linspace(0,np.pi*2,4*rden+2)[:-1]
        rrand = random.random()
        for nr in new_ring:
            ang1 = xr*rrand+nr
            grid_points.append([np.cos(ang1)*rad,np.sin(ang1)*rad,orad-orad*np.cos(r)])
            direcs.append([np.cos(ang1)*np.sin(r),np.sin(ang1)*np.sin(r),np.cos(r)])

    return grid_points,direcs
    

#Same as above two but for a disk
def DiskGridFB(den,outer_rad):
    gr = (1+np.sqrt(5))/2.0
    grid_points = []
    direcs = []
    N = int((outer_rad*outer_rad*np.pi)*den)
    for i in range(N):
        ang1 = 2*np.pi*np.fmod((i/gr),1)
        #print(ang1)
        rad = np.sqrt(i/(N-1))*outer_rad
        grid_points.append([np.cos(ang1)*rad,np.sin(ang1)*rad,0])
        direcs.append([0,0,1])
    #print(len(grid_points))
    return grid_points,direcs

#Same as above two but for a disk
def DiskGrid(den,inner_rad,xbox,ybox,outer_rad):
    grid_points = []
    direcs = []
    if outer_rad<0:
        outer_rad = np.sqrt(xbox*xbox+ybox*ybox)
    else:
        new_den =2*den/np.pi
        new_den = new_den*new_den
        return DiskGridFB(new_den,outer_rad)
    frden = int(den*2*(outer_rad-inner_rad)/np.pi)
    rings = np.linspace(inner_rad,outer_rad,frden)
    for xr,r in enumerate(rings[1:]):
        rad = r
        rden = int(rad*den)+1
        new_ring = np.linspace(0,np.pi*2,4*rden)[:-1]
        rrand = random.random()
        for nr in new_ring:
            ang1 = xr*rrand+nr
            if(-xbox <np.cos(ang1)*rad < xbox-0.2 and -ybox < np.sin(ang1)*rad < ybox-0.2):
                grid_points.append([np.cos(ang1)*rad,np.sin(ang1)*rad,0])
                direcs.append([0,0,1])
    #print(len(grid_points))
    return grid_points,direcs


#Same as above but for a tube
def TubeGrid(den,rad,leng,keep_end):
    grid_points = []
    direcs = []
    frden = int(den*2*(leng)/np.pi)+1
    rings = np.linspace(0,leng,frden)
    if(not keep_end):
        rings = rings[:-1]
    for xr,r in enumerate(rings):
        lenny = r
        rden = int(rad*den)+1
        new_ring = np.linspace(0,np.pi*2,4*rden)[:-1]
        rrand = random.random()
        for nr in new_ring:
            ang1 = xr*rrand+nr
            grid_points.append([np.cos(ang1)*rad,np.sin(ang1)*rad,-lenny])
            direcs.append([np.cos(ang1),np.sin(ang1),0])
    return grid_points,direcs

#standard deviation for gaussians
stand = 0.5

#function that evalualtes a gaussian on a grid (points)
def eval_gaussian(points,cen,std,height):
    return height*np.exp(-np.linalg.norm((points-cen[None,None,:]),axis=2)/(2*std*std))


#function that evaluates a sum of gaussians at a single point
@jax.jit
def eval_gaussianj(point,all_cens,std,height):
    ret_val = 0
    def egj_loop(ret_val,ind):
        ret_val += height*jnp.exp(-jnp.linalg.norm(point-all_cens[ind])/(2*std*std))
        return ret_val,ind
    ret_val,_ = jax.lax.scan(egj_loop,ret_val,jnp.arange(all_cens.shape[0]))
    return ret_val
    


#gradient of above
eval_g_grad = jax.grad(eval_gaussianj,argnums=0)

#this function evalualtes a sum of gaussians on a grid
def gaussian_grid(points,xstart,ystart,xend,yend,xnum,ynum):
    grid_vals = np.zeros((xnum,ynum,2))
    grid_vals[:,:,0] += np.linspace(xstart,xend,xnum)[None,:]
    grid_vals[:,:,1] += np.linspace(ystart,yend,ynum)[:,None]

    ret_vals = np.zeros((xnum,ynum))
    count =0
    for i in points:
        count += 1
        ret_vals += eval_gaussian(grid_vals,i,stand,1)
    return ret_vals,grid_vals


#This takes a grid of values and draws along a contour at value coff, points are places along this with a given spacing
#This contour can be drawn further out by a constant width from coff. This is the main part of building a micelle
#This works by taking a step defined by (df/dy,-df/dx )*ds which by definition is along the contour
def draw_along(spacing,all_cens,grid,grid_vals,coff,rad_plus,z):
    max_iter = 100000
    tol = 0.1
    ds = 0.01
    tot_ds = 0
    start = jnp.array(grid_vals[jnp.logical_and(grid  > coff-tol,grid < coff+tol)][0])
    grad = eval_g_grad(start,all_cens,stand,1)
    grad_dir = -grad/jnp.linalg.norm(grad)
    start_p = start+rad_plus*grad_dir
    prev_start = start_p.copy()
    all_cens = jnp.array(all_cens)
    path=jnp.zeros((int(max_iter*ds/spacing)+1,2))
    normals=jnp.zeros((int(max_iter*ds/spacing)+1,2))
    def da_loop(path_data,ind):
        path = path_data[0]
        tot_ds = path_data[1]
        count = path_data[2]
        normals = path_data[3]
        start = path_data[4]
        prev_start = path_data[5]
        def early_term(path,tot_ds,count,normals,start,prev_start):
            return path,tot_ds,count,normals,start,prev_start
        def nearly_term(path,tot_ds,count,normals,start,prev_start):
            grad = eval_g_grad(start,all_cens,stand,1)
            grad_dir = -grad/jnp.linalg.norm(grad)
            norm = jnp.array([grad[1],-grad[0]])
            norm /= jnp.linalg.norm(norm)
            start += norm*ds
            tot_ds += jnp.linalg.norm(prev_start-(start+rad_plus*grad_dir))
            prev_start = start+rad_plus*grad_dir
            def adder(path,tot_ds,count,normals):
                path = path.at[count].set(start+rad_plus*grad_dir)
                normals = normals.at[count].set(grad_dir)
                count +=1
                tot_ds = 0.0
                return path,tot_ds,count,normals
            def nadder(path,tot_ds,count,normals):
                return path,tot_ds,count,normals
            
            path,tot_ds,count,normals = jax.lax.cond(tot_ds>spacing,adder,nadder,path,tot_ds,count,normals)
            return path,tot_ds,count,normals,start,prev_start
        path,tot_ds,count,normals,start,prev_start = jax.lax.cond(jnp.logical_and(jnp.linalg.norm(path[count-1]-start_p) < spacing/2,count > 5),early_term,nearly_term,path,tot_ds,count,normals,start,prev_start)
        return (path,tot_ds,count,normals,start,prev_start),ind
    path_data,_=jax.lax.scan(da_loop,(path,0.0,0,normals,start,prev_start),jnp.arange(max_iter))
    path = np.array(path_data[0])[:path_data[2]]
    normals = np.array(path_data[3])[:path_data[2]]
    path = np.pad(path,((0,0),(0,1)),"constant",constant_values=(z,z))
    return path,normals


#gets the average length of lipids for building an accurate micelle.
def get_avlip_len(lipids):
    liplens = 0
    for lipid in lipids:
        atoms    = list(zip(lipidsa[lipid][1].split(),lipidsx[lipidsa[lipid][0]],lipidsy[lipidsa[lipid][0]],lipidsz[lipidsa[lipid][0]]))
        at,ax,ay,az = list(zip(*[i for i in atoms if i[0] != "-"]))
        az       = [(0.5+(i-min(az)))*options["-bd"].value for i in az ]
        
        lipLen = max(az)-min(az)
        liplens += lipLen
    return liplens/len(lipids)

#This uses all the above to build the micelle. Using contours with radius and height varing as a semi circle
#Additionally the center is filled with a disc of lipids
def build_micelle(area_l,all_cens,grid,grid_vals,coff,pbcx,pbcy,avlip_len):
    dims = np.array([pbcx,pbcy,0])
    orad = 2.2    
    end_ang = np.pi
    den = np.pi/2*np.sqrt(1/area_l)
    frden = int(orad*den*2*(end_ang)/np.pi)+1
    rings = np.linspace(0,end_ang,frden)
    gpoints = np.zeros((0,3))
    normals = np.zeros((0,3))
    #print(avlip_len)
    sp = (0.5*(1.15-avlip_len)/(-0.65))+(0.15*(1.8-avlip_len)/0.65)
    if sp < 0.15:
        sp = 0.15
    zpush = 2.0-avlip_len
    for xr,r in enumerate(rings):
        rad = np.sin(r)*orad
        rad_a = sp*(frden/2)-np.abs(sp*(frden/2)-sp*xr)
        rad_z = zpush*np.cos(r)
        znorm = np.array([0,0,1])
        z = np.cos(r)*orad
        r1,nms = draw_along(np.sqrt(area_l),all_cens,grid,grid_vals,coff,rad,z)
        nms_a = np.pad(nms,((0,0),(0,1)),"constant",constant_values=(0,0))
        gpoints = np.concatenate((gpoints,np.array(r1+nms_a*rad_a+znorm*rad_z)))
        norms = nms*np.sin(r)
        norms = np.pad(norms,((0,0),(0,1)),"constant",constant_values=(np.cos(r),np.cos(r)))
        normals = np.concatenate((normals,norms))

    disk_lips = []
    disk_dirs = []
    disk_grid,disc_direcs = DiskGrid(den,0,pbcx,pbcy,-1)
    disc_direcs = np.array(disc_direcs)

    for i,d in enumerate(disk_grid):
        val = eval_gaussianj(jnp.array(d[:2]),jnp.array(all_cens),stand,1)
        if(val > coff+5):
            disk_lips.append(d-np.array([0,0,2+zpush]))
            disk_dirs.append(-disc_direcs[i])
            disk_lips.append(d+np.array([0,0,2+zpush]))
            disk_dirs.append(disc_direcs[i])
    disk_lips = np.array(disk_lips)
    disk_dirs = np.array(disk_dirs)
    gpoints = np.concatenate((gpoints,disk_lips))
    normals = np.concatenate((normals,disk_dirs))
    
    gpoints_max_rad = np.max(np.linalg.norm(disk_lips[:,:2]-dims[:2]/2,axis=1))
    #print(disk_lips[:,:2]-dims[:2]/2)
    #print(np.linalg.norm(disk_lips[:,:2]-dims[:2]/2,axis=1))


    return gpoints,normals,gpoints_max_rad+1.5

#This is an implementation of the KL divergence
def kl_d(a,b):
    kl = np.where(a != 0, a * np.log(a / b), 0)
    kl  = kl[np.isfinite(kl)]
    return np.sum(kl)
#another gaussian
def gaussian(x,sig,mu):
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp(-0.5*np.power(((x-mu)/sig),2))
#The distribution of glycan strands in ecoli
def ecoli_gdist(x):
    return 0.75*gaussian(x,4,8.9)+0.25*gaussian(x,10,45)


#enforces pbc   
def enforce_pbc(pbcv,points):
    points = np.where(points >= pbcv, points-pbcv, points)
    points = np.where(points <0, points+pbcv, points)
    return points

#loads an itp file
def load_itp(fn):
    tmp_f = open(fn,"r")
    tmp_lines = tmp_f.readlines()
    tmp_f.close()
    
    data = [[],[],[],[],[]]           
    
    data_ind = -1
    for lines in tmp_lines:
        if lines == "[ atoms ]\n":
            data_ind = 0
        elif lines == "[ bonds ]\n": 
            data_ind = 1
        elif lines == "[ constraints ]\n":
            data_ind = 2
        elif lines == "[ angles ]\n":
            data_ind = 3
        elif lines == "[ dihedrals ]\n":
            data_ind = 4
        elif len(lines) < 5:
            data_ind = -1
        elif lines[0] != ";":
            if(data_ind > -1):
                data[data_ind].append(lines.split())
    return data

          
#Function that creates a leaflet using all of the above
def create_leaflet(den,xbox,ybox,curv_A,curv_B,ch_ang,pore,leng,keep_end,mem_outer_red): 
    if(curv_A < 1e-5 or curv_B < 1e-5):
        gpoints3,direcs3 = np.array(DiskGrid(den,0,xbox,ybox,mem_outer_red))
        return gpoints3,direcs3
    rad_A = 1.0/curv_A
    rad_B = 1.0/curv_B 

    if(pore):  
        ch_ang=np.pi/2 
                
    c1 = np.cos(ch_ang)
    s1 = np.sin(ch_ang)
    
    c2 = np.cos(np.pi/2-ch_ang)
    s2 = np.sin(np.pi/2-ch_ang)
          
    in_rad = rad_A*s1+rad_B*(c2-1)
    
    
    if(not pore):
        gpoints,direcs = np.array(SphereGridN(den,ch_ang,rad_A))      
    else:
        gpoints,direcs = np.array(TubeGrid(den,in_rad,leng,keep_end))
       
        
        
    gpoints2,direcs2 = np.array(InvSphereGrid(den,in_rad,(np.pi/2-ch_ang),rad_B))
    gpoints3,direcs3 = np.array(DiskGrid(den,rad_B+in_rad,xbox,ybox,mem_outer_red))
    shift = -gpoints[-1,2]+s2*rad_B-rad_B
    if(pore):
         gpoints[:,2] = gpoints[:,2]-rad_B
         gpoints2[:,2] = gpoints2[:,2]-rad_B
         gpoints3[:,2] = gpoints3[:,2]
    else:
        gpoints2[:,2] = gpoints2[:,2]-rad_B -shift
        gpoints3[:,2] = gpoints3[:,2] -shift
    gpoints = np.concatenate([gpoints,gpoints2,gpoints3])
    direcs = np.concatenate([direcs,direcs2,direcs3])
    return gpoints,direcs


#A fully analytic function of the leaflet. This is very useful for accurate exclusion of solute 
def leaflet_function(x,y,curv_A,curv_B,ch_ang,pore,leng):
    if(curv_A < 1e-5 or curv_B < 1e-5):
        return 0,np.array([0,0,1])
    rad_p = np.sqrt(x*x+y*y)
    if(pore):  
        ch_ang=np.pi/2    
    
    c1 = np.cos(ch_ang)
    s1 = np.sin(ch_ang)
    
    c2 = np.cos(np.pi/2-ch_ang)
    s2 = np.sin(np.pi/2-ch_ang)
    rad_A = 1.0/curv_A
    rad_B = 1.0/curv_B   
    sphere_rad = s1*rad_A
    in_rad = rad_A*s1+rad_B*(c2-1)
    shift = (1-c1)*rad_B-(s2*rad_A-rad_A)
    if(pore):
        shift = 0
    if(rad_p< sphere_rad):
        if(pore):
            z = leng
            direc = np.array([-x,-y,0])
            return z,direc/np.linalg.norm(direc)
        else:
            z = np.sqrt(rad_A*rad_A-rad_p*rad_p)-rad_A
            direc = np.array([-x,-y,z+rad_A])
            return z,direc/np.linalg.norm(direc)
    elif(rad_p < in_rad+rad_B):
        z = -np.sqrt(rad_B*rad_B-(rad_p-rad_B-in_rad)*(rad_p-rad_B-in_rad))+rad_B-shift
        xydirec = np.array([x,y])
        xydirec /= np.linalg.norm(xydirec)
        direc = np.array([(x-xydirec[0]*(rad_B+in_rad)),(y-xydirec[1]*(rad_B+in_rad)),-(z+shift-rad_B)])
        return z,direc/np.linalg.norm(direc)
    else:
        return -shift,np.array([0,0,1])

#Get the minimum dimensions required to fit all the curvature   
def get_box_size(curv_A,curv_B,ch_ang,pore):
    if(curv_A < 1e-5 or curv_B < 1e-5):
        return 0,0
    if(pore):  
        ch_ang=np.pi/2    
    
    c1 = np.cos(ch_ang)
    s1 = np.sin(ch_ang)
    
    c2 = np.cos(np.pi/2-ch_ang)
    s2 = np.sin(np.pi/2-ch_ang)
    rad_A = 1.0/curv_A
    rad_B = 1.0/curv_B 
    
    in_rad = rad_A*s1+rad_B*(c2-1)
    height = rad_A*(1-c1)+rad_B*(1-c1)
    return in_rad+rad_B,height
   
#A function for binning points into a fixed grid
def put_into_grid(x,r,n,s):
    cx = x[0]-s[0]
    cy = x[1]-s[1]
    norm_x = cx/r[0]
    norm_y = cy/r[1]
    grid_x = n[0]*norm_x
    grid_y = n[1]*norm_y
    return int(np.floor(grid_x)),int(np.floor(grid_y))
    
#Used with above   
def bin_grid(grid_in,r,n,s):
    grid_final = np.zeros((n[0],n[1]))
    grid = np.zeros((n[0],n[1],grid_in.shape[0]+1))
    for k in range(grid_in.shape[0]):
        gx,gy = put_into_grid(grid_in[k,:2],r,n,s)
        grid[gx,gy,int(grid[gx,gy,-1])] = grid_in[k,2]
        grid[gx,gy,-1] += 1
    for i in range(n[0]):
        for j in range(n[1]):
            if(grid[i,j,-1] == 0):
                grid_final[i,j] = np.nan
            else:
                grid_final[i,j] = np.sum(grid[i,j,:-1])/grid[i,j,-1]
    return grid_final

#Another method of binning points.
def get_box_slice(points,p,r):
    pslice = points[points[:,0]>p[0]-r[0]]
    pslice = pslice[pslice[:,0]<p[0]+r[0]]
    pslice = pslice[pslice[:,1]>p[1]-r[1]]
    pslice = pslice[pslice[:,1]<p[1]+r[1]]
    pslice = pslice[pslice[:,2]>p[2]-r[2]]
    pslice = pslice[pslice[:,2]<p[2]+r[2]]
    return pslice
    
    
#This is a function for sorting atoms for a neater topology file
def reorder_atoms(atoms,coords):
    lip_names = []
    lip_split = []
    lip_split_coords = []
    for i,a in enumerate(atoms):
        if(a[1] in lip_names):
            lip_split[lip_names.index(a[1])].append(list(a))
            lip_split_coords[lip_names.index(a[1])].append(coords[i])
        else:
            lip_names.append(a[1])
            lip_split.append([])
            lip_split_coords.append([])
            lip_split[-1].append(list(a))
            lip_split_coords[-1].append(coords[i])
    count = 1
    prev = 1    
    for i in lip_split:
        for k in i:
            if(k[2] == prev):
                k[2] = count
            else:
                prev = k[2]
                count += 1
                k[2] = count
    new_atoms = []
    new_coords = []
    for i in range(len(lip_split)):
        for k in range(len(lip_split[i])):
            new_atoms.append(lip_split[i][k])
            new_coords.append(lip_split_coords[i][k])
    return new_atoms,new_coords
    
#Ignore this  
def LocBias(coords,lnums,lipids,steps,loc_lipids,bias_grid):
    lipids=list(lipids)
    lip_ranges = []
    total = 0
    for l in range(len(lipids)):
        lip_ranges.append([total,total+lnums[l]])
        total += lnums[l]
    swap = []
    nswap = []
    for l in range(len(lipids)):
        if lipids[l] in loc_lipids:
            swap.append(l)
        else:
            nswap.append(l) 
    for i in range(steps):
        bulk = random.choice(nswap)
        swaper = random.choice(swap)
        bstart,bend = lip_ranges[bulk]
        sstart,send = lip_ranges[swaper]
        blip = random.randint(bstart,bend-1)
        slip = random.randint(sstart,send-1)
        bcoord = coords[blip]
        scoord = coords[slip]
        binda,bindb = put_into_grid(bcoord[:2],[pbcx, pbcy],[bias_grid.shape[0],bias_grid.shape[1]],[0,0])
        sinda,sindb = put_into_grid(scoord[:2],[pbcx, pbcy],[bias_grid.shape[0],bias_grid.shape[1]],[0,0])
        
        swap_val = -bias_grid[sinda,sindb]+bias_grid[binda,bindb]
        swap_prob = 1/(1+np.exp(-0.3*swap_val))
        
        if random.random() < swap_prob:
            coords[slip]=bcoord
            coords[blip]=scoord
    return coords
            
#gets charge between a lower and upper bound, for neutralising compartments
def get_charge(lz,uz):
    last = None
    mcharge = 0
    for ind,j in enumerate(membrane.atoms):
        pos = membrane.coord[ind]
        if not j[0].strip().startswith('v') and j[1:3] != last:
            if(pos[2] > lz and pos[2] < uz):
                mcharge += charges.get(j[1].strip(),0)
        last = j[1:3]
        

    last = None
    pcharge = 0
    for ind,j in enumerate(protein.atoms):
        pos = protein.coord[ind]
        if not j[0].strip().startswith('v') and j[1:3] != last:
            if(pos[2] > lz and pos[2] < uz):
                pcharge += charges.get(j[1].strip(),0)  
        last = j[1:3]
    total_charge = mcharge+pcharge
    return total_charge,pcharge,mcharge,0
            
tm   = []
lipL = []
lipU = []
lipL_loc = []
lipU_loc = []
lipLO = []
lipUO = []
solv = []
pos_ions = [[],[],[]]
neg_ions = [[],[],[]]

# HII edit - lipid definition, for extra lipid definitaions
usrmols  = []
usrheads = []
usrlinks = []
usrtails = []
usrLipHeadMapp = { # Define supported lipid head beads. One letter name mapped to atom name
    "C":  ('NC3'), # NC3 = Choline
    "E":  ('NH3'), # NH3 = Ethanolamine 
    "G":  ('GL0'), # GL0 = Glycerol
    "S":  ('CNO'), # CNO = Serine
    "P":  ('PO4'), # PO4 = Phosphate
    "O":  ('PO4')  # PO4 = Phosphate acid
    }
usrIndexToLetter = "A B C D E F G H I J K L M N".split() # For naming lipid tail beads 

# Description
desc = ""

# Option list
options = [
#   option           type number default description
# HII edit - lipid definition (last options are for additional lipid specification)
    """
Input/output related options
""",
    ("-f",      Option(tm.append,   1,        None, "Input GRO or PDB file 1: Protein")),
    ("-o",      Option(str,         1,        None, "Output GRO file: Membrane with Protein")),
    ("-p",      Option(str,         1,        None, "Optional rudimentary topology file")),
    ("-ct",      Option(str,         1,        None, "This will create a template of the membrane only.")),
    ("-in_t",      Option(str,         1,        None, "Input template for placement of multiple proteins.")),
    ("-fs",      Option(str,         1,        None, "Input text file with multiple proteins")),
    """
Options related to system size. It is reccomended to use -x, -y, -z only. 
Many options that were available here have been removed and will be readded 
at a later date.
""",
    ("-d",      Option(float,       1,           0, "Distance between periodic images (nm)")),
    ("-dz",     Option(float,       1,           0, "Z distance between periodic images (nm)")),
    ("-x",      Option(vector,      1,           0, "X dimension or first lattice vector of system (nm)")),
    ("-y",      Option(vector,      1,           0, "Y dimension or first lattice vector of system (nm)")),
    ("-z",      Option(vector,      1,           0, "Z dimension or first lattice vector of system (nm)")),
    ("-box",    Option(readBox,     1,        None, "Box in GRO (3 or 9 floats) or PDB (6 floats) format, comma separated")),
    ("-n",      Option(str,         1,        None, "Index file --- TO BE IMPLEMENTED")),
    """
Membrane/lipid related options.  
The options -l and -u can be given multiple times. Option -u can be
used to set the lipid type and abundance for the upper leaflet. Option
-l sets the type and abundance for the lower leaflet if option -u is
also given, or for both leaflets if option -u is not given. -lo and -uo
behave as -l and -u except -l can set abundance for -lo and -uo is these
are not specified. Curvature can be specified using -curv and pores can be
added using -pore.
""",
    ("-l",      Option(lipL.append, 1,   None, "Lipid type and relative abundance (NAME[:#])")),
    ("-u",      Option(lipU.append, 1,   None, "Lipid type and relative abundance (NAME[:#])")),
    ("-lo",      Option(lipLO.append, 1,   None, "Lipid type and relative abundance (For double membrane definitions) (NAME[:#])")),
    ("-uo",      Option(lipUO.append, 1,   None, "Lipid type and relative abundance (For double membrane definitions) (NAME[:#])")),
    ("-a",      Option(float,       1,        0.60, "Area per lipid (nm*nm)")),
    ("-au",     Option(float,       1,        None, "Area per lipid (nm*nm) for upper layer")),
    ("-ao",      Option(float,       1,        None, "Area per lipid (nm*nm) for outer membrane")),
    ("-auo",     Option(float,       1,        None, "Area per lipid (nm*nm) for outer membrane upper layer")),
    ("-asym",   Option(int,         1,        None, "Membrane asymmetry (number of lipids)")),
    ("-rand",   Option(float,       1,         0.1, "Random kick size (maximum atom displacement)")),
    ("-bd",     Option(float,       1,         0.3, "Bead distance unit for scaling z-coordinates (nm)")),
    ("-ps",        Option(float,        1,          0, "Specifies the distance (nm) between inner and outer membrane, when set to 0 only a single membrane is built")),
    ("-curv",        Option(str,        1,          "0,0,1", "Curvature of the membrane, consists of 3 comma separated values. The curvature at the middle the curvature as it relaxes back to planar and the direction of the curvature.")),
    ("-curv_o",        Option(str,        1,          "0,0,1", "Curvature of the outer membrane, see -curv")),
    ("-curv_ext",        Option(float,        1,          3, "Extent of curved region in the absence of a protein, this also controls the size of the pore if -pore is used")),
    ("-pore",        Option(bool,        0,          None, "Create a pore, with inner radius equal to -curv_ext and length equal to -ps")),
    ("-loc",      Option(str,         1,        None, "Currently WIP do not use. Input .npz for localisation data to bias lipid arangment.")),
    ("-l_loc",      Option(lipL_loc.append, 1,   None, "Currently WIP do not use. Lipid type for localisation")),
    ("-u_loc",      Option(lipU_loc.append, 1,   None, "Currently WIP do not use. Lipid type for localisation")),
    ("-micelle",        Option(bool,        0,          None, "Builds a micelle around a protein instead of a bilayer")),
    ("-radius",        Option(float,        1,          -1, "Radius of membrane outer disk. This is by default the whole cell")),
    """
Protein related options. -fudge gives the exclusion radius around the protein.
""",
    ("-center", Option(bool,        0,        None, "Center the protein on z")),
    ("-rotate", Option(str,         1,        None, "Rotate protein (random|princ|angle(float))")),
    ("-fudge",  Option(float,       1,         0.3, "Fudge factor for allowing lipid-protein overlap")),
    ("-ring",   Option(bool,        0,        None, "Put lipids inside the protein")),
    ("-dm",     Option(float,       1,        None, "Shift protein with respect to membrane")),
    """
Solvent related options.
""",
    ("-sol",    Option(solv.append, 1,        None, "Solvent type and relative abundance (NAME[:#])")),
    ("-sold",   Option(float,       1,         0.5, "Solvent diameter")),
    ("-solr",   Option(float,       1,         0.1, "Solvent random kick")),
    ("-excl",   Option(float,       1,         1.5, "Exclusion range (nm) for solvent addition relative to membrane center")),
    """
Charge related options.
""",
    ("-posi_c0",   Option(pos_ions[0].append,         1,        None, "Positive ion type and relative abundance (NAME[:#]) in compartment 0")),
    ("-negi_c0",   Option(neg_ions[0].append,         1,        None, "Negative ion type and relative abundance (NAME[:#]) in compartment 0")),
    ("-posi_c1",   Option(pos_ions[1].append,         1,        None, "Positive ion type and relative abundance (NAME[:#]) in compartment 1")),
    ("-negi_c1",   Option(neg_ions[1].append,         1,        None, "Negative ion type and relative abundance (NAME[:#]) in compartment 1")),
    ("-posi_c2",   Option(pos_ions[2].append,         1,        None, "Positive ion type and relative abundance (NAME[:#]) in compartment 2")),
    ("-negi_c2",   Option(neg_ions[2].append,         1,        None, "Negative ion type and relative abundance (NAME[:#]) in compartment 2")),
    ("-ion_conc",   Option(str,       1,         "0.15,0.15,0.15", "Concentration of ions in each compartment")),
    ("-charge", Option(str,         1,      "auto", "Charge of system. Set to auto to infer from residue names")),
    ("-charge_ratio", Option(str,         1,      None, "Ratios of charge in each compartment")),
    ("-zpbc", Option(bool,         0,      None, "Toggles if water compartments are determined with pbc in the Z axis.")),
    """
Define additional lipid types (same format as in lipid-martini-itp-v01.py)
""",
    ("-alname",  Option(usrmols.append,         1,        None, "Additional lipid name, x4 letter")),
    ("-alhead",  Option(usrheads.append,        1,        None, "Additional lipid head specification string")),
    ("-allink",  Option(usrlinks.append,        1,        None, "Additional lipid linker specification string")),
    ("-altail",  Option(usrtails.append,        1,        None, "Additional lipid tail specification string")),
    ]
    
args = sys.argv[1:]

if '-h' in args or '--help' in args:
    print("\n",__file__)
    print(desc or "\nSomeone ought to write a description for this script...\n")
    for thing in options:
        print(type(thing) != str and "%10s  %s"%(thing[0],thing[1].description) or thing)
    print("")
    sys.exit()


# Convert the option list to a dictionary, discarding all comments
options = dict([i for i in options if not type(i) == str])


# Process the command line
while args:
    ar = args.pop(0)
    options[ar].setvalue([args.pop(0) for i in range(options[ar].num)])



absoluteNumbers = not options["-d"]


# HII edit - lipid definition
# Add specified lipid definition to insane lipid library
for name, head, link, tail in list(zip(usrmols,usrheads,usrlinks,usrtails)):
    moltype = "usr_"+name
    lipidsx[moltype] = []
    lipidsy[moltype] = []
    lipidsz[moltype] = []
    headArray = (head).split()
    linkArray = (link).split()
    tailsArray = (tail).split()
    lipidDefString = ""  

    if len(tailsArray) != len(linkArray):
        print("Error, Number of tails has to equal number of linkers")
        sys.exit()

    # Find longest tail 
    maxTail = 0
    for cTail in tailsArray:
       if len(cTail) > maxTail:
           maxTail = len(cTail)
    cBeadZ = maxTail + len(headArray) # longest tail + linker (always x1) + lengths of all heads - 1 (as it starts on 0)

    # Add head beads
    for cHead in headArray:
        lipidsx[moltype].append(0)
        lipidsy[moltype].append(0)
        lipidsz[moltype].append(cBeadZ)
        cBeadZ -= 1
        lipidDefString += usrLipHeadMapp[cHead] + " "

    # Add linkers
    for i,cLinker in enumerate(linkArray):
        lipidsx[moltype].append(max(i-0.5,0))
        lipidsy[moltype].append(0)
        lipidsz[moltype].append(cBeadZ)
        if cLinker == 'G': 
            lipidDefString += "GL" + str(i+1) + " "
        elif cLinker == 'A':
            lipidDefString += "AM" + str(i+1) + " "
        else:
            print("Error, linker type not supported")
            sys.exit()

    # Add tails 
    for i,cTail in enumerate(tailsArray):
        cBeadZ = maxTail - 1
        
        for j,cTailBead in enumerate(cTail):
            lipidsx[moltype].append(i)
            lipidsy[moltype].append(0)
            lipidsz[moltype].append(cBeadZ)
            cBeadZ -= 1
            lipidDefString += cTailBead + str(j+1) + usrIndexToLetter[i] + " "
   
    lipidsa[name] = (moltype,lipidDefString)
# End user lipid definition


# HII edit - lipid definition, had to move this one below the user lipid definitions to scale them to.
# First all X/Y coordinates of templates are centered and scaled (magic numbers!)
for i in lipidsx.keys():
    cx = (min(lipidsx[i])+max(lipidsx[i]))/2
    lipidsx[i] = [0.25*(j-cx) for j in lipidsx[i]]
    cy = (min(lipidsy[i])+max(lipidsy[i]))/2
    lipidsy[i] = [0.25*(j-cy) for j in lipidsy[i]]


# Periodic boundary conditions

# option -box overrides everything
if options["-box"].value:
    options["-x"].value = options["-box"].value[:3]
    options["-y"].value = options["-box"].value[3:6]
    options["-z"].value = options["-box"].value[6:]
    
    
mem_outer_red = options["-radius"].value




is_micelle = options["-micelle"].value
#print(is_micelle)

if(is_micelle is not None):
    is_micelle = True
else:
    is_micelle = False

#print(is_micelle)

# options -x, -y, -z take precedence over automatic determination
pbcSetX = 0
if type(options["-x"].value) in (list,tuple):
    pbcSetX = options["-x"].value
elif options["-x"].value:
    pbcSetX = [options["-x"].value,0,0]

pbcSetY = 0
if type(options["-y"].value) in (list,tuple):
    pbcSetY = options["-y"].value
elif options["-y"].value:
    pbcSetY = [0,options["-y"].value,0]

pbcSetZ = 0
if type(options["-z"].value) in (list,tuple):
    pbcSetZ = options["-z"].value
elif options["-z"].value:
    pbcSetZ = [0,0,options["-z"].value]




################
## I. PROTEIN ##
################


protein  = Structure()
protein_lip = Structure()
prot     = []
xshifts  = [0] # Shift in x direction per protein

#Reading and processing inputs

double_mem = False
if(options["-ps"].value > 1e-5):
    double_mem = True
using_temp = False
if(options["-fs"].value is not None):
    using_temp = True
    
zpbc = False
if(options["-zpbc"].value is not None):
    zpbc = True


add_pore = options["-pore"].value
if(add_pore is not None):
    add_pore = True
else:
    add_pore = False


zdist = float(options["-ps"].value)

curv_vals = options["-curv"].value.split(",")
curvo_vals = options["-curv_o"].value.split(",")


curv_vals_np = np.zeros(3)
curv_vals_np[2] += 1

if(len(curv_vals) == 1):
    curv_vals_np[0] = float(curv_vals[0])
    curv_vals_np[1] = float(curv_vals[0])
elif(len(curv_vals) == 2):
    curv_vals_np[0] = float(curv_vals[0])
    curv_vals_np[1] = float(curv_vals[1])
elif(len(curv_vals) == 3):
    curv_vals_np[0] = float(curv_vals[0])
    curv_vals_np[1] = float(curv_vals[1])
    curv_vals_np[2] = float(curv_vals[2])
    
curvo_vals_np = np.zeros(3)
curvo_vals_np[2] += 1

if(len(curvo_vals) == 1):
    curvo_vals_np[0] = float(curvo_vals[0])
    curvo_vals_np[1] = float(curvo_vals[0])
elif(len(curvo_vals) == 2):
    curvo_vals_np[0] = float(curvo_vals[0])
    curvo_vals_np[1] = float(curvo_vals[1])
elif(len(curvo_vals) == 3):
    curvo_vals_np[0] = float(curvo_vals[0])
    curvo_vals_np[1] = float(curvo_vals[1])
    curvo_vals_np[2] = float(curvo_vals[2])
    



rcurvs_mid = np.array([curv_vals_np[1],curvo_vals_np[1]])
curvs_mid = np.array([curv_vals_np[0],curvo_vals_np[0]])

if(add_pore):
    if(rcurvs_mid[0] < 1e-5):
        print("WARNING: Return curvature (inner) cannot be 0 setting to 0.1")
        rcurvs_mid[0] = 0.1
    if(rcurvs_mid[1] < 1e-5):
        print("WARNING: Return curvature (outer) cannot be 0 setting to 0.1")
        rcurvs_mid[1] = 0.1
 
if(add_pore):
    cdirs = [1,-1]
else:
    cdirs = [float(curv_vals[2]),float(curvo_vals[2])]
    
if(curvs_mid[0] < 1e-5):
    ncurves = np.zeros(2)
else:
    ncurves = np.array([1/((1/curvs_mid[0])+2),1/((1/curvs_mid[0])-2)])
    
if(curvs_mid[1] < 1e-5):
    ncurves_o = np.zeros(2)
else:
    ncurves_o = np.array([1/((1/curvs_mid[1])+2),1/((1/curvs_mid[1])-2)]) 

curv_up = np.array([ncurves[0],ncurves_o[0]])
curv_lo = np.array([ncurves[1],ncurves_o[1]])
 
if(rcurvs_mid[0] < 1e-5):
    nrcurves = np.zeros(2)
else:
    nrcurves = np.array([1/((1/rcurvs_mid[0])+2),1/((1/rcurvs_mid[0])-2)])
    
if(rcurvs_mid[1] < 1e-5):
    nrcurves_o = np.zeros(2)
else:
    nrcurves_o = np.array([1/((1/rcurvs_mid[1])+2),1/((1/rcurvs_mid[1])-2)]) 
    
rcurv_up = np.array([nrcurves[1],nrcurves_o[1]])
rcurv_lo = np.array([nrcurves[0],nrcurves_o[0]])


extent= float(options["-curv_ext"].value)
if(add_pore):
    for i in range(2):
        curvs_mid[i] = 1/extent

prot_dirs = []
prot_rings = []


if not tm or options["-ct"].value != None or using_temp:
 
    ang_exts = np.array([np.arcsin(curvs_mid[0]*extent),np.arcsin(curvs_mid[1]*extent)]) 
    resi = 0
    bsize,bheight = get_box_size(curvs_mid[0],rcurvs_mid[0],ang_exts[0],add_pore)
    
   
    if(pbcSetX[0] < bsize*2+2):
        pbcSetX[0] = bsize*2+2
    if(pbcSetY[1] < bsize*2+2):
        pbcSetY[1] = bsize*2+2  
        
    pbcx = pbcSetX and pbcSetX[0]
    pbcy = pbcSetY and pbcSetY[1]
    pbcz = pbcSetZ and pbcSetZ[2]

## B. PROTEIN ---
    #for writing a template for muliple protein placments
    if(options["-ct"].value != None):
        new_file = open(options["-ct"].value,"w")
        count = 0
        trufal = [True,False]
        ang_exts = np.array([np.arcsin(curvs_mid[0]*extent),np.arcsin(curvs_mid[1]*extent)])
        for i in range(2):
            if(i == 0 or double_mem):
                if(add_pore):
                    inner_leng = zdist-(1/rcurvs_mid[i])
                else:
                    inner_leng = 0         
                temp_points,direcs = create_leaflet(4,pbcx/2,pbcy/2,curvs_mid[i],rcurvs_mid[i],ang_exts[i],add_pore,inner_leng,trufal[i],mem_outer_red)
                if(cdirs[i] > 0):
                    temp_points[:,2] = -temp_points[:,2]
                    direcs[:,2] = -direcs[:,2]
                   
                temp_points =(temp_points+np.array([pbcx/2,pbcy/2,zdist*(2*i-1)]))*10
                for gp in temp_points:
                    count += 1
                    count_str = (6-len(str(count)))*" "+str(count)
                    c = "ATOM "+count_str+" BB   DUM     1       0.000   0.000  15.000  1.00  0.00" 
                    xp = np.format_float_positional(gp[0],precision=3)
                    yp = np.format_float_positional(gp[1],precision=3)
                    zp = np.format_float_positional(gp[2],precision=3)
                    xp += "0"*(3-len((xp.split(".")[1])))
                    yp += "0"*(3-len((yp.split(".")[1])))
                    zp += "0"*(3-len((zp.split(".")[1])))
                    new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:]+"\n"    
                    new_file.write(new_c)
            
        new_file.close()
        info_fn = options["-ct"].value.split(".")[0]+".txt"
        info_file = open(info_fn,"w")
        info_file.write("Curvature values:"+options["-curv"].value+"\n")
        info_file.write("Outer curvature values:"+options["-curv_o"].value+"\n")
        info_file.write("Pore:"+str(add_pore)+"\n")
        info_file.write("X Y Z:"+str(options["-x"].value)+","+str(options["-y"].value)+","+str(options["-z"].value)+"\n")
        info_file.write("Box:"+str(options["-box"].value))
        info_file.close()
        
        exit()



    if(using_temp):
        #WHen a template is present it is used to build membrane
        in_prots = options["-fs"].value
        prot_file = open(in_prots,"r")
        lines = prot_file.read().split("\n")
        prot_file.close()
        
  
        
        temp_file = open(options["-in_t"].value,"r")
        temp_lines = temp_file.read().split("\n")
        temp_file.close()
        poses = np.zeros((len(lines),3))
        for tl in temp_lines:
            stl = tl.split()
            if(len(stl)>2):
                prot_no = int(float(stl[-2]))-1
                if(prot_no> -1):
                    zpos = float(tl[46:54])
                    ypos = float(tl[38:46])
                    xpos = float(tl[30:38])
                    poses[prot_no] = np.array([xpos,ypos,zpos])/10
        for pr in lines:
            pr_list = pr.split()
            prot = pr_list[0]
            direction = float(pr_list[1])
            rings = int(pr_list[2])
            if(rings == 0):
                rings = False
            else:
                rings = True
            tm.append(prot)
            prot_dirs.append(direction)
            prot_rings.append(rings)
        tm_lip    = [ Structure(i) for i in tm ]
        tm    = [ Structure(i) for i in tm ]

if(using_temp or tm):
    if(not using_temp):
        #when not using a template
        tm_lip    = [ Structure(i) for i in tm ]
        tm    = [ Structure(i) for i in tm ]
              
        
        
        xmax = np.max(np.array(tm[0].coord)[:,0])
        ymax = np.max(np.array(tm[0].coord)[:,1])
        xmin = np.min(np.array(tm[0].coord)[:,0])
        ymin = np.min(np.array(tm[0].coord)[:,1])
        xrang = xmax-xmin
        yrang = ymax-ymin
        
        rang = np.max([xrang,yrang])/2
        extent = rang
        ang_exts = np.array([np.arcsin(curvs_mid[0]*extent),np.arcsin(curvs_mid[1]*extent)])
        
        bsize,bheight = get_box_size(curvs_mid[0],rcurvs_mid[0],ang_exts[0],add_pore)
        if(pbcSetX[0] < bsize*2+2):
            pbcSetX[0] = bsize*2+2
        if(pbcSetY[1] < bsize*2+2):
            pbcSetY[1] = bsize*2+2
        if(pbcSetZ[2] < bheight*2+10):
            pbcSetZ[2] = bheight*2+10
        
        #print(bheight*2+2)
            
        pbcx = pbcSetX and pbcSetX[0]
        pbcy = pbcSetY and pbcSetY[1]
        pbcz = pbcSetZ and pbcSetZ[2]
        
        
        poses = np.zeros((1,3))
        prot_dirs.append(1)
        if(options["-ring"].value != None):
            rings = True
        else:
            rings = False
        prot_rings.append(rings)  
        
        poses[0][0] = pbcx/2
        poses[0][1] = pbcy/2
            
            
    for pind,prot in enumerate(tm):

        prot_lip = tm_lip[pind]
        ## a. NO MEMBRANE --
        if not lipL:
            pass
        ## b. PROTEIN AND MEMBRANE --
        else:
           
            # Have to build a membrane around the protein. 
            # So first put the protein in properly.


            # Center the protein and store the shift
            shift = prot.center((0,0,0))
            shift_lip = prot_lip.center((0,0,0))

            ## 5. Determine the minimum and maximum x and y of the protein 
            pmin, pmax = prot.fun(min), prot.fun(max)
            prng       = (pmax[0]-pmin[0],pmax[1]-pmin[1],pmax[2]-pmin[2])
            center     = (0.5*(pmin[0]+pmax[0]),0.5*(pmin[1]+pmax[1]))


            pbcz += options["-dz"].value or options["-d"].value or 0


            ## 2. Shift of protein relative to the membrane center
            zshift = 0
            if not options["-center"].value:
                zshift = -shift[2]
            if options["-dm"].value:
                if options["-dm"].value < 0:
                    zshift += options["-dm"].value # - max(list(zip(*prot.coord))[2])
                else:                        
                    zshift += options["-dm"].value # - min(list(zip(*prot.coord))[2])


            if(poses[pind][2] < 0):
                mind = 0
            else:
                mind = 1
            if(add_pore):
                inner_leng = zdist-(1/rcurvs_mid[mind])
            else:
                inner_leng = 0 
            fun_pos,normal = leaflet_function(poses[pind][0]-pbcx/2,poses[pind][1]-pbcy/2,curvs_mid[mind],rcurvs_mid[mind],ang_exts[mind],add_pore,inner_leng)                              
            if(cdirs[mind] > 0):
                fun_pos = -fun_pos
                normal[2] = -normal[2]

            downn = np.array([0,0,-1*prot_dirs[pind]])                  
            angl = np.dot(normal,downn)
            if(angl < -1+1e-5):
                downn = np.array([0,1e-4,-1*prot_dirs[pind]]) 
                downn = downn/np.linalg.norm(downn)                 
                angl = np.dot(normal,downn)

            v = np.cross(downn,normal)
            vmat = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            rot_mat = np.eye(3)+vmat+np.dot(vmat,vmat)*1/(1+angl)
            
            prot += [0,0,zshift]
            prot_lip.coord = list(np.array(prot_lip.coord)+[0,0,zshift])                
  
            if(not prot_rings[pind]):
                for dmi in range(2):
                    if(dmi == 0 or (not using_temp and zdist > 1e-5)):
                        if(not using_temp and zdist > 1e-5):
                            ztestpos = zdist*(2*dmi-1)
                        else:
                            ztestpos = 0
                       
                        in_mem = get_box_slice(np.array(prot_lip.coord),[0,0,ztestpos],[10,10,2])
                        for im in in_mem:
                            for dim in range((int(np.linalg.norm(im))+1)):
                                imdir = im/np.linalg.norm(im)                        
                                new_coord = imdir*dim
                                new_coord[2] = ztestpos+1
                                prot_lip.coord.extend([new_coord])
                                new_coord[2] = ztestpos-1
                                prot_lip.coord.extend([new_coord])
                
            
            prot.coord = np.dot(rot_mat,np.array(prot.coord).T).T    
            prot_lip.coord = np.dot(rot_mat,np.array(prot_lip.coord).T).T     
            prot += poses[pind]
            prot_lip.coord = list(np.array(prot_lip.coord)+poses[pind])


        # And we collect the atoms
        protein.atoms.extend(prot.atoms)
        protein.coord.extend(prot.coord)
        
        
        protein_lip.atoms.extend(prot_lip.atoms)
        protein_lip.coord.extend(prot_lip.coord)
        

    prot_up,prot_lo,prot_up_a,prot_up_b,prot_lo_a,prot_lo_b = [],[],[],[],[],[]
        

    # Current residue ID is set to that of the last atom
    resi = protein.atoms[-1][2]

atid      = len(protein)+1
molecules = []

# The box dimensions are now (likely) set.
# If a protein was given, it is positioned in the center of the
# rectangular brick.


# Override lattice vectors if they were set explicitly
box = [[0,0,0],[0,0,0],[0,0,0]]
box[0] = pbcSetX
box[1] = pbcSetY
box[2] = pbcSetZ

grobox = (box[0][0],box[1][1],box[2][2],
          box[0][1],box[0][2],box[1][0],
          box[1][2],box[2][0],box[2][1])

pbcx, pbcy, pbcz = box[0][0], box[1][1], box[2][2]



rx, ry, rz = pbcx+1e-8, pbcy+1e-8, pbcz+1e-8


#################
## 2. MEMBRANE ##
#################

membrane = Structure()
mi_rad = 0
if lipL:
    #reading inputs
    area_l = float(options["-a"].value)
    if(not options["-au"].value is None):
        area_u = float(options["-au"].value)
    else:
        area_u = float(options["-a"].value)

    if(not options["-ao"].value is None):
        area_lo = float(options["-ao"].value)
    else:
        area_lo = float(options["-a"].value)
    
    
    if(not options["-auo"].value is None):
        area_uo = float(options["-auo"].value)
    else:
        if(not options["-ao"].value is None):
            area_uo = float(options["-ao"].value)
        else:
            area_uo = float(options["-a"].value)
    
    density_l = [np.pi/2*np.sqrt(1/area_l),np.pi/2*np.sqrt(1/area_lo)]
    density_u = [np.pi/2*np.sqrt(1/area_u),np.pi/2*np.sqrt(1/area_uo)]
    area_ls = [area_l,area_lo]


   
    
    up_grids = []
    lo_grids = []
    direcss = []
    direcs2s = []
    
    lipU = lipU or lipL
    trufal = [True,False]
    #building membrae grid points and normals based on inputs
    for i in range(2):
        if(options["-ps"].value > 1e-5 or i == 0):
            if(add_pore):
                inner_leng = zdist-(1/rcurvs_mid[i])
                curv_up[i] = 1/(extent+2)
                curv_lo[i] = 1/(extent-2)
            else:
                inner_leng = 0
            if(is_micelle):
                prot_points = np.array(protein.coord)
                prot_points = prot_points[np.logical_and(prot_points[:,2]+zdist*(i*2-1) < 2,prot_points[:,2]+zdist*(i*2-1) > -2)]
                prot_points = prot_points[:,:2]
                prot_points = prot_points[::2]
                prot_points = jnp.array(prot_points)


                ret_val,grid_vals = gaussian_grid(prot_points,-pbcx/2,-pbcx/2,pbcx/2,pbcx/2,300,300)
                av_liplen = get_avlip_len(lipU)
                up_grid,direcs,mi_rad = build_micelle(area_ls[i],prot_points,ret_val,grid_vals,8,pbcx,pbcy,av_liplen)

                lo_grid = np.zeros((0,3))
                direcs2 = np.zeros((0,3))
            else:
                up_grid,direcs = create_leaflet(density_u[i],pbcx/2,pbcy/2,curv_up[i],rcurv_up[i],ang_exts[i],add_pore,inner_leng,trufal[i],mem_outer_red)
                lo_grid,direcs2 = create_leaflet(density_l[i],pbcx/2,pbcy/2,curv_lo[i],rcurv_lo[i],ang_exts[i],add_pore,inner_leng,trufal[i],mem_outer_red)
                up_grid[:,2] = -up_grid[:,2]+2
                lo_grid[:,2] = -lo_grid[:,2]-2


            if(cdirs[i] < 0):
                temp = up_grid
                up_grid = lo_grid
                lo_grid= temp
                
                dtemp = -direcs
                direcs = -direcs2
                direcs2= dtemp        
                
                up_grid[:,2] = -up_grid[:,2]
                lo_grid[:,2] = -lo_grid[:,2]
                direcs[:,2] = -direcs[:,2]
                direcs2[:,2] = -direcs2[:,2]
            if(not is_micelle):
                up_grid = up_grid+np.array([pbcx/2,pbcy/2,0])
                lo_grid = lo_grid+np.array([pbcx/2,pbcy/2,0])
        else:
            up_grid = np.zeros((0,3))
            direcs = np.zeros((0,3))
            lo_grid = np.zeros((0,3))
            direcs2 = np.zeros((0,3))
            
        up_grids.append(up_grid)
        lo_grids.append(lo_grid)
        direcss.append(direcs)
        direcs2s.append(direcs2)   

    upper, lower = [], []
    random.seed()
    
    
    #lipU = lipU or lipL
    lipU_loc = lipU_loc or lipL_loc
    lipLO = lipLO or lipL
    lipUO = lipUO or lipLO
    lipUs = [lipU,lipUO]
    lipLs = [lipL,lipLO]
    
    av_liplenU = get_avlip_len(lipU)
    av_liplenUO = get_avlip_len(lipUO)
    av_liplenL = get_avlip_len(lipL)
    av_liplenLO = get_avlip_len(lipLO)
    
    #removing lipids within -fudge of protein
    prot_coords = np.array(protein_lip.coord)
    tailsU = np.linspace(0,av_liplenU,10)
    tailsUO = np.linspace(0,av_liplenUO,10)
    tailsL = np.linspace(0,av_liplenL,10)
    tailsLO = np.linspace(0,av_liplenLO,10)
    tailUs = [tailsU,tailsUO]
    tailsLs = [tailsL,tailsLO]
    exprot = float(options["-fudge"].value)
    zdist = float(options["-ps"].value)
    for gi in range(2):
        total_up_lips = up_grids[gi].shape[0]
        total_lo_lips = lo_grids[gi].shape[0]
        if(options["-ps"].value > 1e-5 or gi == 0):
            upper_tmp, lower_tmp = [], []
            for i in range(total_up_lips):
                poser = np.array([up_grids[gi][i][0],up_grids[gi][i][1],up_grids[gi][i][2]+zdist*(gi*2-1)])
                direc = np.array([direcss[gi][i][0],direcss[gi][i][1],direcss[gi][i][2]])
                coll = False
                if tm:
                    for dr in tailUs[gi]:
                        prot_slice = get_box_slice(prot_coords,poser-dr*direc,[exprot,exprot,exprot])
                        for pp in prot_slice:
                            if(np.linalg.norm(poser-dr*direc-pp)<exprot):
                                coll = True
                                break
                        if(coll):
                            break
                if(not coll):
                     upper_tmp.append((random.random(),up_grids[gi][i][0],up_grids[gi][i][1],up_grids[gi][i][2],direcss[gi][i][0],direcss[gi][i][1],direcss[gi][i][2]))
                     upper.append((random.random(),up_grids[gi][i][0],up_grids[gi][i][1],up_grids[gi][i][2],direcss[gi][i][0],direcss[gi][i][1],direcss[gi][i][2]))
            for i in range(total_lo_lips):
                poser = np.array([lo_grids[gi][i][0],lo_grids[gi][i][1],lo_grids[gi][i][2]+zdist*(gi*2-1)])
                direc = np.array([direcs2s[gi][i][0],direcs2s[gi][i][1],direcs2s[gi][i][2]])
                coll = False
                if tm:
                    for dr in tailsLs[gi]:
                        prot_slice = get_box_slice(prot_coords,poser+dr*direc,[exprot,exprot,exprot])
                        for pp in prot_slice:
                            if(np.linalg.norm(poser+dr*direc-pp)<exprot):
                                coll = True
                                break
                        if(coll):
                            break
                if(not coll):
                     lower_tmp.append((random.random(),lo_grids[gi][i][0],lo_grids[gi][i][1],lo_grids[gi][i][2],direcs2s[gi][i][0],direcs2s[gi][i][1],direcs2s[gi][i][2]))
                     lower.append((random.random(),lo_grids[gi][i][0],lo_grids[gi][i][1],lo_grids[gi][i][2],direcs2s[gi][i][0],direcs2s[gi][i][1],direcs2s[gi][i][2]))

            
            # Sort on the random number
            upper_tmp.sort()
            lower_tmp.sort()
            
            
    
    
            # Extract coordinates, taking asymmetry in account
            asym  = options["-asym"].value or 0
            upper_tmp = [i[1:] for i in upper_tmp[max(0, asym):]]
            lower_tmp = [i[1:] for i in lower_tmp[max(0,-asym):]]
            print("; %d lipids in upper leaflet, %d lipids in lower leaflet"%(len(upper_tmp),len(lower_tmp)),file=sys.stderr )
    
            # Types of lipids, relative numbers, fractions and numbers
            
            # Upper leaflet (+1)
            lipU_new, numU = list(zip(*[ parse_mol(i) for i in lipUs[gi] ]))
            totU       = float(sum(numU))
            num_up     = [int(len(upper_tmp)*i/totU) for i in numU]
            
            lip_up     = [l for i,l in list(zip(num_up,lipU_new)) for j in range(i)]

            leaf_up    = ( 1,list(zip(lip_up,upper_tmp)))
            
            # Lower leaflet (-1)
            lipL_new, numL = list(zip(*[ parse_mol(i) for i in lipLs[gi] ]))
            totL       = float(sum(numL))
            num_lo     = [int(len(lower_tmp)*i/totL) for i in numL]
            lip_lo     = [l for i,l in list(zip(num_lo,lipL_new)) for j in range(i)]
            
            #lower_tmp = LocBias(lower_tmp,num_lo,lipL_new,10000,lipL_loc,loc_grid[0])
            leaf_lo    = (-1,list(zip(lip_lo,lower_tmp)))
            if(gi == 0):
                molecules  = list(zip(lipU_new,num_up)) + list(zip(lipL_new,num_lo))
            else:
                molecules  += list(zip(lipU_new,num_up)) + list(zip(lipL_new,num_lo))
    
            kick       = options["-rand"].value
            
            
            
            # Build the membrane
            for leaflet,leaf_lip in [leaf_up,leaf_lo]:
                for lipid, pos_dir in leaf_lip:
                    pos = pos_dir[:3]
                    dirs = pos_dir[3:]
                    # Increase the residue number by one
                    resi += 1
                    # Set the random rotation for this lipid
                    rangle   = 2*random.random()*math.pi
                    rcos     = math.cos(rangle)
                    rsin     = math.sin(rangle)
                    # Fetch the atom list with x,y,z coordinates
                    atoms    = list(zip(lipidsa[lipid][1].split(),lipidsx[lipidsa[lipid][0]],lipidsy[lipidsa[lipid][0]],lipidsz[lipidsa[lipid][0]]))
                    # Only keep atoms appropriate for the lipid
                    at,ax,ay,az = list(zip(*[i for i in atoms if i[0] != "-"]))
                    # The z-coordinates are spaced at 0.3 nm,
                    # starting with the first bead at 0.15 nm
                    #+leaflet*(0.5+(i-min(az)))*options["-bd"].value
                    az       = [ leaflet*-2+pos[2]+leaflet*(0.5+(i-min(az)))*options["-bd"].value for i in az ]
                    xx       = list(zip( ax,ay ))
                    nx       = [rcos*i-rsin*j+pos[0]+random.random()*kick for i,j in xx]
                    ny       = [rsin*i+rcos*j+pos[1]+random.random()*kick for i,j in xx]
                    # Add the atoms to the list
                    nppos = np.array(dirs)
                    pos2 = np.array([pos[0],pos[1],pos[2]])
                    downn = np.array([0,0,1])                  
                    angl = np.dot(nppos,downn)
                    if(angl < -1+1e-5):
                        downn = np.array([0,1e-4,1]) 
                        downn = downn/np.linalg.norm(downn)                 
                        angl = np.dot(nppos,downn)

                    v = np.cross(downn,nppos)
                    vmat = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
                    rot_mat = np.eye(3)+vmat+np.dot(vmat,vmat)*1/(1+angl)
                    new_poses = [np.dot(rot_mat,np.array([nx[i],ny[i],az[i]])-pos2)+pos2 for i in range(len(az))]               
                    for i in range(len(at)):
                        atom  = "%5d%-5s%5s%5d"%(resi,lipid,at[i],atid)
                        membrane.coord.append((new_poses[i][0],new_poses[i][1],new_poses[i][2]+zdist*(gi*2-1)))               
                        
                        membrane.atoms.append((at[i],lipid,resi,0,0,0))
                        atid += 1
                     


    # Now move everything to the center of the box before adding solvent
    mz  = pbcz/2
    z   = [ i[2] for i in protein.coord+membrane.coord ]
    mz -= (max(z)+min(z))/2
    protein += (0,0,mz)
    membrane += (0,0,mz)
else:
    mz=0




#PG layer


################
## 3. SOLVENT ##
################

# Charge of the system so far


charge,pcharge,mcharge,_ = get_charge(0,pbcz)

plen, mlen, slen = 0, 0, 0
plen = protein and len(protein) or 0

print("; NDX Protein %d %d" % (1, protein and plen or 0),file=sys.stderr)
print("; Charge of protein: %f" % pcharge,file=sys.stderr)

mlen = membrane and len(membrane) or 0

pglen = 0

print("; NDX Membrane %d %d" % (1+plen, membrane and plen+mlen or 0),file=sys.stderr)
print("; Charge of membrane: %f" % mcharge,file=sys.stderr)


print("; Total charge: %f" % charge,file=sys.stderr)


def _point(y,phi):
    r = math.sqrt(1-y*y)
    return math.cos(phi)*r, y, math.sin(phi)*r


def pointsOnSphere(n):
    return [_point((2.*k+1)/n-1,k*2.3999632297286531) for k in range(n)]
    
if solv:

    # Set up a grid
    d        = 1/(options["-sold"].value*np.sqrt(3)/2)
    da =     1/options["-sold"].value

    nx,ny,nz = int(1+da*pbcx),int(1+d*pbcy),int(1+d*pbcz)
      
    dx,dy,dz = pbcx/nx,pbcy/ny,pbcz/nz
    zdist = options["-ps"].value
    inner_lengs = [zdist,zdist]
   
      
    dzdist = int(zdist/dz)+1
    
    excl,hz  = int(nz*options["-excl"].value/pbcz), int(0.5*nz)

    zshift   = 0
    if membrane:
        if(options["-ps"].value > 1e-5):
            memz   = [i[2] for i in membrane.coord]
            midz   = (max(memz)+min(memz))/2
            memz_upp   = [i[2] for i in membrane.coord if i[2]>midz]
            midz_upp   = (max(memz_upp)+min(memz_upp))/2
            memz_low   = [i[2] for i in membrane.coord if i[2]<=midz]
            midz_low   = (max(memz_low)+min(memz_low))/2
            midz = (midz_upp+midz_low)/2
            excl -= 1
        else:
            memz   = [i[2] for i in membrane.coord]
            midz   = (max(memz)+min(memz))/2
        hz     = int(nz*midz/pbcz)  # Grid layer in which the membrane is located
        zshift = (hz+0.5)*nz - midz # Shift of membrane middle to center of grid layer
    # Initialize a grid of solvent, spanning the whole cell
    # Exclude all cells within specified distance from membrane center
    if(lipL):
        if(options["-ps"].value > 1e-5):
            grids = []      
            for gk in range(2):
            
                if(cdirs[gk] < 0):
                    grid   = [[[(pbcz*i)/nz < -leaflet_function((pbcx*k)/nx-pbcx/2+((i+j)%2)*0.5*dx,(pbcy*j)/ny-pbcy/2+(i%2)*0.5*dy,curv_up[gk],rcurv_up[gk],ang_exts[gk],add_pore,inner_lengs[gk]-2)[0]-2+mz+zdist*(2*gk-1) or (pbcz*i)/nz > -leaflet_function((pbcx*k)/nx-pbcx/2+((i+j)%2)*0.5*dx,(pbcy*j)/ny-pbcy/2+(i%2)*0.5*dy,curv_lo[gk],rcurv_lo[gk],ang_exts[gk],add_pore,inner_lengs[gk]+2)[0]+2+mz+zdist*(2*gk-1) for i in range(nz)] for j in range(ny)] for k in range(nx)]
                else:  
                    grid   = [[[(pbcz*i)/nz > leaflet_function((pbcx*k)/nx-pbcx/2+((i+j)%2)*0.5*dx,(pbcy*j)/ny-pbcy/2+(i%2)*0.5*dy,curv_up[gk],rcurv_up[gk],ang_exts[gk],add_pore,inner_lengs[gk]-2)[0]+2+mz+zdist*(2*gk-1) or (pbcz*i)/nz < leaflet_function((pbcx*k)/nx-pbcx/2+((i+j)%2)*0.5*dx,(pbcy*j)/ny-pbcy/2+(i%2)*0.5*dy,curv_lo[gk],rcurv_lo[gk],ang_exts[gk],add_pore,inner_lengs[gk]+2)[0]-2+mz+zdist*(2*gk-1) for i in range(nz)] for j in range(ny)] for k in range(nx)]
                grids.append(grid)
            grids = np.array(grids)
            grid = grids[0]*grids[1]
        
        else:            
            if is_micelle:
                rad = mi_rad
            else:
                if mem_outer_red>0:
                    rad = mem_outer_red    
                else:
                    rad = 1000
            if(cdirs[0] < 0):
                grid   = [[[((pbcx*k)/nx-pbcx/2)*((pbcx*k)/nx-pbcx/2)+((pbcy*j)/ny-pbcy/2)*((pbcy*j)/ny-pbcy/2) > rad*rad or (pbcz*i)/nz < -leaflet_function((pbcx*k)/nx-pbcx/2+((i+j)%2)*0.5*dx,(pbcy*j)/ny-pbcy/2+(i%2)*0.5*dy,curv_up[0],rcurv_up[0],ang_exts[0],add_pore,inner_lengs[0])[0]-2+mz or (pbcz*i)/nz > -leaflet_function((pbcx*k)/nx-pbcx/2+((i+j)%2)*0.5*dx,(pbcy*j)/ny-pbcy/2+(i%2)*0.5*dy,curv_lo[0],rcurv_lo[0],ang_exts[0],add_pore,inner_lengs[0])[0]+2+mz for i in range(nz)] for j in range(ny)] for k in range(nx)]

            else:  
                grid   = [[[((pbcx*k)/nx-pbcx/2)*((pbcx*k)/nx-pbcx/2)+((pbcy*j)/ny-pbcy/2)*((pbcy*j)/ny-pbcy/2) > rad*rad or (pbcz*i)/nz > leaflet_function((pbcx*k)/nx-pbcx/2+((i+j)%2)*0.5*dx,(pbcy*j)/ny-pbcy/2+(i%2)*0.5*dy,curv_up[0],rcurv_up[0],ang_exts[0],add_pore,inner_lengs[0])[0]+2+mz or (pbcz*i)/nz < leaflet_function((pbcx*k)/nx-pbcx/2+((i+j)%2)*0.5*dx,(pbcy*j)/ny-pbcy/2+(i%2)*0.5*dy,curv_lo[0],rcurv_lo[0],ang_exts[0],add_pore,inner_lengs[0])[0]-2+mz for i in range(nz)] for j in range(ny)] for k in range(nx)]
    else:
        grid   =   [[[True for i in range(nz)] for j in range(ny)] for k in range(nx)]
    # Flag all cells occupied by protein or membrane
    for p,q,r in protein.coord+membrane.coord:
        for s,t,u in pointsOnSphere(20):
            x,y,z = p+0.33*s,q+0.33*t,r+0.33*u
            
            if z >= pbcz:
                x -= box[2][0]
                y -= box[2][1]
                z -= box[2][2]
            if z < 0:
                x += box[2][0]
                y += box[2][1]
                z += box[2][2]
            if y >= pbcy: 
                x -= box[1][0]
                y -= box[1][1]
            if y < 0: 
                x += box[1][0]
                y += box[1][1]
            if x >= pbcx: 
                x -= box[0][0]
            if x < 0: 
                x += box[0][0]


            zgrid = int(nz*z/rz)
            y = y-(0.5*dy)*(zgrid%2)
            ygrid = int(ny*y/ry)
            x = x - (0.5*dx)*((zgrid+ygrid)%2)


            grid[int(nx*x/rx)][ygrid][zgrid] = False
    anti_freeze = 0.2
    if(is_micelle):
        anti_freeze = 0.2
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if(random.random() < anti_freeze):
                    grid[i][j][k] = False


    # Set the center for each solvent molecule
    kick = options["-solr"].value
    grid = [ [R(),(i+0.5+R()*kick)*dx + (0.5*dx)*((j+k)%2),(j+0.5+R()*kick)*dy+(0.5*dy)*(k%2),(k+0.5+R()*kick)*dz] 
             for i in range(nx) for j in range(ny) for k in range(nz) if grid[i][j][k] ]
    
    comp_nos = np.zeros(3)    
    if(zdist > 1e-5):
        for gr in grid:
            lz = leaflet_function(gr[1]-pbcx/2,gr[2]-pbcy/2,curv_up[0],rcurv_up[0],ang_exts[0],add_pore,inner_lengs[0])[0]+mz-zdist
            uz = -leaflet_function(gr[1]-pbcx/2,gr[2]-pbcy/2,curv_up[1],rcurv_up[1],ang_exts[1],add_pore,inner_lengs[1])[0]+mz+zdist
            
            if(gr[3] > lz and gr[3] < uz):
                gr[0] += 1
                comp_nos[1] += 1
            elif gr[3] < lz:
                comp_nos[0] += 1
            elif gr[3] > uz:
                if(not (zpbc or add_pore)):
                    gr[0] += 2
                comp_nos[2] += 1
    
    else:
        for gr in grid:
            lz = cdirs[0]*leaflet_function(gr[1]-pbcx/2,gr[2]-pbcy/2,curv_up[0],rcurv_up[0],ang_exts[0],add_pore,inner_lengs[0])[0]+mz
            if(gr[3] > lz):
                if(not (zpbc or add_pore)):
                    gr[0] += 1
                comp_nos[2] += 1
            elif gr[3] < lz:
                comp_nos[0] += 1
    if(zpbc or add_pore):
        comp_nos[0] = comp_nos[0]+comp_nos[2]
        comp_nos[2] = 0
        

    # Sort on the random number
    grid.sort()

    # 'grid' contains all positions on which a solvent molecule can be placed.
    # The number of positions is taken as the basis for determining the salt concentration.
    # This is fine for simple salt solutions, but may not be optimal for complex mixtures
    # (like when mixing a 1M solution of this with a 1M solution of that

    # First get names and relative numbers for each solvent
    solnames, solnums = list(zip(*[ parse_mol(i) for i in solv ]))
    solnames, solnums = list(solnames), list(solnums)
    totS       = float(sum(solnums))

    # Set the number of ions to add
    nna, ncl = 0, 0
  
    charge_rats = np.zeros(3)
    if options["-charge_ratio"].value:
        concs = options["-charge_ratio"].value.split(",")
        # If the concentration is set negative, set the charge to zero  
        countc = 0    
        for c in concs:
            charge_rats[countc] = float(c)
            countc += 1
        

        
    
    non_zero = 0
    for cn in comp_nos:
        if(cn > 1e-5):
            non_zero += 1
    

    concentrations = np.zeros(3)
    if options["-ion_conc"].value:
        concs = options["-ion_conc"].value.split(",")
        # If the concentration is set negative, set the charge to zero  
        countc = 0    
        for c in concs:
            concentrations[countc] = float(c)
            countc += 1

        nsol = ("SPC" in solnames and 1 or 4)*len(grid)
                       
    # Correct number of grid cells for placement of solvent

           
    if options["-charge"].value != "0":
        charge = (options["-charge"].value != "auto") and int(options["-charge"].value) or charge
    else:
        charge = 0
 
    ncharges = np.zeros(3,dtype=int)
    if options["-charge_ratio"].value:
        charge_rats /= np.sum(charge_rats[:non_zero])
        ncharges = np.array(charge*charge_rats,dtype=int)
    else:
        if(zdist > 1e-5):
            ncharges[1],_,_,_ = get_charge(mz-zdist,mz+zdist)
            ncharges[2],_,_,_ = get_charge(mz+zdist,pbcz)
            ncharges[0],_,_,_ = get_charge(0,mz-zdist)
            if(zpbc):
                ncharges[0] = ncharges[0]+ncharges[2]
        else:
            if(zpbc):
                ncharges[0] = charge
            else:
                ncharges[2],_,_,_ = get_charge(mz,pbcz)
                ncharges[0],_,_,_ = get_charge(0,mz)


    #dealing with charge of each compartment
    diff_c = charge-np.sum(ncharges)
    ncharges[0] += diff_c
    new_solnames = []
    new_num_sol = []
    rcomp_nos = comp_nos/np.sum(comp_nos)

    if(len(pos_ions[1]) == 0):
        pos_ions[1] = pos_ions[0]
    if(len(pos_ions[2]) == 0):
        pos_ions[2] = pos_ions[0]

    if(len(neg_ions[1]) == 0):
        neg_ions[1] = neg_ions[0]
    if(len(neg_ions[2]) == 0):
        neg_ions[2] = neg_ions[0]
    ion_names = []
    for pi in pos_ions:
        for pii in pi:
            mole = parse_mol(pii)[0]
            if(mole not in solnames):
                solnames.append(mole)
                ion_names.append(mole)
    for pi in neg_ions:
        for pii in pi:
            mole = parse_mol(pii)[0]
            if(mole not in solnames):
                solnames.append(mole)
                ion_names.append(mole)
    
    for ci,cn in enumerate(comp_nos):
        added = 0
        new_nsol = nsol*rcomp_nos[ci]
        total_ions  = int(1+(concentrations[ci]*new_nsol/(27.7+concentrations[ci])))
        
        cperi = -ncharges[ci]/total_ions

        negi = neg_ions[ci]
        posi = pos_ions[ci]


        total_negc = 0
        total_posc = 0
        total_negi = 0
        total_posi = 0
        for ni_c in negi:
            iname, icount = parse_mol(ni_c)
            total_negc += ion_charges[iname]*icount
            total_negi += icount

        for pi_c in posi:
            iname, icount = parse_mol(pi_c)
            total_posc += ion_charges[iname]*icount
            total_posi += icount


        if(total_negc-(total_negi/total_posi) == 0):
            A_pos = (cperi-total_negc/total_negi)/(total_posc-total_negc*(total_posi/total_negi))
            A_neg = (cperi-total_posc*A_pos)/total_negc
        elif(total_posc-(total_posi/total_negi) == 0):
            A_neg = (cperi-total_posc/total_posi)/(total_negc-total_posc*(total_negi/total_posi))
            A_pos = (cperi-total_negc*A_neg)/total_posc
        else:
            A_neg = (cperi-total_posc/total_posi)/(total_negc-total_posc*(total_negi/total_posi))
            A_pos = (cperi-total_negc/total_negi)/(total_posc-total_negc*(total_posi/total_negi))

        if(A_pos < 0):
            A_pos = 0
            A_neg = cperi/total_negc
        if(A_neg < 0):
            A_neg = 0
            A_pos = cperi/total_posc

        A_pos = max(0,A_pos)*total_ions
        A_neg = max(0,A_neg)*total_ions

        negi_dict = {}
        posi_dict = {}

        for pi_c in ion_names:
            negi_dict[pi_c]=0
            posi_dict[pi_c]=0
            
        for ni_c in negi:
            iname, icount = parse_mol(ni_c)
            negi_dict[iname]=icount*A_neg

        total_pic = 0
        for pi_c in posi:
            iname, icount = parse_mol(pi_c)
            posi_dict[iname]=icount*A_pos

        ion_nums = []
        for inn in ion_names:
            ion_nums.append(int(posi_dict[inn]+negi_dict[inn]))

        

        total_ion_charge = 0
        for ind,inn in enumerate(ion_names):
            total_ion_charge += ion_charges[inn]*ion_nums[ind]


        cdiff = ncharges[ci]+total_ion_charge


        if(cdiff > 0):
            for ind,inn in enumerate(ion_names):
                if(ion_charges[inn] == -1):
                    ion_nums[ind] += cdiff
                    break

        elif(cdiff < 0):
            for ind,inn in enumerate(ion_names):
                if(ion_charges[inn] == 1):
                    ion_nums[ind] -= cdiff
                    break
  

        ngrid   = cn - np.sum(ion_nums)
        num_sol = [int(ngrid*i/totS) for i in solnums]
        for inn in ion_nums:
            num_sol.append(inn)

        

        for n,sn in enumerate(solnames):
            adj_cn = num_sol          
            if(cn > 1e-5):
                new_solnames.append(sn)
                new_num_sol.append(num_sol[n])
                added += num_sol[n]
        differ = int(cn-added)
        if(len(new_num_sol)>0):
            new_num_sol[-len(solnames)]+=differ

    num_sol = []    
    for i in range(len(solnames)):
        num_sol.append(np.sum(new_num_sol[i::len(solnames)]))
    print("THIS SHOULD BE ZERO",len(grid)-sum(num_sol))
            
     

    # Names and grid positions for solvent molecules
    solvent    = list(zip([s for i,s in list(zip(new_num_sol,new_solnames)) for j in range(i)],grid))

    solvent_split = []
    for sni in range(len(solnames)):
        solvent_split.append([])
    for si in solvent:
        if(si[0] in solnames):
            solvent_split[solnames.index(si[0])].append(si)
    solvent = []
    for si in solvent_split:
        for ssi in si:
            solvent.append(ssi)

    # Extend the list of molecules (for the topology)
    molecules.extend(list(zip(solnames,num_sol)))
    
    cc = 0
    # Build the solvent
    sol = []
    for resn,(rndm,x,y,z) in solvent:
        
        cc += 1
        resi += 1
        solmol = solventParticles.get(resn)
        if solmol and len(solmol) > 1:       
            # Random rotation (quaternion)
            u,  v,  w       = random.random(), 2*math.pi*random.random(), 2*math.pi*random.random()
            s,  t           = math.sqrt(1-u), math.sqrt(u)
            qw, qx, qy, qz  = s*math.sin(v), s*math.cos(v), t*math.sin(w), t*math.cos(w)
            qq              = qw*qw-qx*qx-qy*qy-qz*qz         
            for atnm,(px,py,pz) in solmol:                
                qp = 2*(qx*px + qy*py + qz*pz)
                rx = x + qp*qx + qq*px + qw*(qy*pz-qz*py)
                ry = y + qp*qy + qq*py + qw*(qz*px-qx*pz)
                rz = z + qp*qz + qq*pz + qw*(qx*py-qy*px)
                sol.append(("%5d%-5s%5s%5d"%(resi%1e5,resn,atnm,atid%1e5),(rx,ry,rz)))
                atid += 1
        else:          
            sol.append(("%5d%-5s%5s%5d"%(resi%1e5,resn,solmol and solmol[0][0] or resn,atid%1e5),(x,y,z)))
            atid += 1
else:
    solvent, sol = None, []


## Write the output ##
slen = solvent and len(sol) or 0

print("; NDX Solvent %d %d" % (1+plen+mlen+pglen, solvent and plen+mlen+slen+pglen or 0),file=sys.stderr)
print("; NDX System %d %d" % (1, plen+mlen+slen+pglen),file=sys.stderr)
print("; \"I mean, the good stuff is just INSANE\" --Julia Ormond",file=sys.stderr)

# Open the output stream
oStream = options["-o"] and open(options["-o"].value,"w") or sys.stdout

# print(the title)
if membrane.atoms:
    title  = "INSANE! Membrane UpperLeaflet>"+":".join(lipU)+"="+":".join([str(i) for i in numU])
    title += " LowerLeaflet>"+":".join(lipL)+"="+":".join([str(i) for i in numL])

    if protein:
        title = "Protein in " + title
else:
    title = "Insanely solvated protein."

print(title,file=oStream)

 #print(the number of atoms)
print("%5d"%(len(protein)+len(membrane)+len(sol)),file=oStream)


membrane.atoms,membrane.coord = reorder_atoms(membrane.atoms,membrane.coord)

new_molecules = []
mol_types = []
for m in molecules:
    if(m[0] in mol_types):
        new_molecules[mol_types.index(m[0])][1] += m[1]
    else:
        mol_types.append(m[0])
        new_molecules.append(list(m))


# print(the atoms)
id = 1
if protein:
    for i in range(len(protein)):
        at,rn,ri = protein.atoms[i][:3]
        x,y,z    = protein.coord[i]
        oStream.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"%(ri%1e5,rn,at,id%1e5,x,y,z))
        id += 1
if membrane:
    for i in range(len(membrane)):
        at,rn,ri = membrane.atoms[i][:3]
        x,y,z    = membrane.coord[i]
        oStream.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"%(ri%1e5,rn,at,id%1e5,x,y,z))
        id += 1
if sol:
    # print(the solvent)
    print("\n".join([i[0]+"%8.3f%8.3f%8.3f"%i[1] for i in sol]),file=oStream)
    

# print(the box)
print("%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f\n"%grobox,file=oStream)
oStream.close()
if options["-p"].value:
    # Write a rudimentary topology file
    with open(options["-p"].value,"w") as top:
        #print('#include "martini_v3.itp"\n',file=top)
        print('#include "martini_v3.0.0.itp"\n',file=top)
        print('#include "martini_v3.0.0_ions_v1.itp"\n',file=top)
        print('#include "martini_v3.0.0_solvents_v1.itp"\n',file=top)
        print('#include "martini_v3.0.0_phospholipids_v1.itp"\n',file=top)
        if protein:
            print('#include "protein-cg.itp"\n',file=top)
        print('[ system ]\n; name\n%s\n\n[ molecules ]\n; name  number'%title,file=top)
        if protein:
            print("%-10s %5d"%("Protein",1),file=top)
        print("\n".join("%-10s %7d"%tuple(i) for i in new_molecules),file=top)

else:
    print("\n".join("%-10s %7d"%tuple(i) for i in new_molecules),file=sys.stderr)
