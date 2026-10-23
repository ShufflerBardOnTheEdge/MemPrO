"""
Regression tests for Insane4MemPrO membrane building and mempro-leaflets.

Run from the repository root with:  python -m unittest discover tests
Each test builds a small system in a temporary directory (a few seconds each).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSANE = os.path.join(ROOT, "mempro", "Insane4MemPrO.py")
LEAFLETS = os.path.join(ROOT, "mempro", "leaflet_check.py")
sys.path.insert(0, os.path.join(ROOT, "mempro"))
import leaflet_check  # noqa: E402

IONS = ["-sol", "W", "-negi_c0", "CL", "-posi_c0", "NA"]
TMP_DIRS = []


def tearDownModule():
    for d in TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


def build(args, ions=True):
    """Builds a system and returns (log, molecule counts, path of the .gro)."""
    tmp = tempfile.mkdtemp(prefix="insane_test_")
    TMP_DIRS.append(tmp)
    cmd = [sys.executable, INSANE] + args + (IONS if ions else ["-sol", "W"]) + ["-o", "sys.gro", "-p", "topol.top"]
    out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    log = out.stdout + out.stderr
    if out.returncode != 0:
        raise AssertionError("Insane4MemPrO failed:\n" + log[-3000:])
    counts = {}
    with open(os.path.join(tmp, "topol.top")) as f:
        mols = f.read().split("[ molecules ]")[1]
    for line in mols.splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith(";"):
            counts[parts[0]] = counts.get(parts[0], 0) + int(parts[1])
    return log, counts, os.path.join(tmp, "sys.gro")


def leaflet_counts(log):
    """(upper, lower) lipid counts for each membrane, from the build log."""
    return [tuple(map(int, m)) for m in re.findall(r"; (\d+) lipids in upper leaflet, (\d+) lipids in lower leaflet", log)]


def leaflet_areas(log):
    return [tuple(map(float, m)) for m in re.findall(r"Lipid area \(nm\^2\): upper leaflet (\S+), lower leaflet (\S+)", log)]


class TestFlatFilling(unittest.TestCase):
    def test_flat_leaflet_gets_box_area_over_apl(self):
        log, counts, _ = build(["-x", "12", "-y", "10", "-z", "10", "-l", "POPE"])
        self.assertEqual(leaflet_counts(log), [(207, 207)])  # 120 nm^2 / 0.58
        self.assertEqual(counts["POPE"], 414)

    def test_square_grid_same_count(self):
        log, counts, _ = build(["-x", "12", "-y", "10", "-z", "10", "-l", "POPE", "-grid", "square"])
        self.assertEqual(leaflet_counts(log), [(207, 207)])

    def test_explicit_area(self):
        log, counts, _ = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPE", "-a", "0.7"])
        self.assertEqual(leaflet_counts(log), [(143, 143)])  # 100 / 0.7

    def test_mixed_leaflet_fills_box_area(self):
        log, counts, _ = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE:7", "-l", "POPG:2", "-l", "CARD:1"])
        for up, lo in leaflet_areas(log):
            self.assertAlmostEqual(up, 225, delta=2)
            self.assertAlmostEqual(lo, 225, delta=2)
        # composition ratio kept, cardiolipin counted per lipid
        self.assertAlmostEqual(counts["POPE"] / counts["CARD"], 7, delta=0.3)
        area = (counts["POPE"] * 0.58 + counts["POPG"] * 0.64 + counts["CARD"] * 1.30) / 2
        self.assertAlmostEqual(area, 225, delta=3)

    def test_apl_override(self):
        log, counts, _ = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPE", "-apl", "POPE:0.5"])
        self.assertEqual(leaflet_counts(log), [(200, 200)])

    def test_temperature_and_m3(self):
        log, _, _ = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPC", "-temp", "310", "-apl_ff", "m3"])
        self.assertIn("Area per lipid (nm^2): upper 0.612, lower 0.612", log)


class TestDoubleMembraneAndCurvature(unittest.TestCase):
    def test_outer_membrane_lipid_a(self):
        log, counts, _ = build(["-x", "20", "-y", "20", "-z", "40", "-ps", "10", "-l", "POPE:8", "-l", "POPG:1", "-l", "CARD:1",
                                "-uo", "LIPA", "-negi_c1", "CL", "-posi_c1", "NA", "-negi_c2", "CL", "-posi_c2", "NA:1", "-posi_c2", "CA:4"])
        self.assertEqual(counts["LIPA"], 256)  # 400 / 1.56
        for up, lo in leaflet_areas(log):
            self.assertAlmostEqual(up / lo, 1, delta=0.01)

    def test_reversed_curvature_keeps_leaflet_areas(self):
        # Previously each leaflet was built with the other's density when the curvature was reversed
        log, _, _ = build(["-x", "30", "-y", "30", "-z", "25", "-l", "POPE", "-u", "POPG", "-au", "0.9", "-curv", "0.13,0.1,-1"])
        (up, lo), = leaflet_areas(log)
        self.assertAlmostEqual(up / lo, 1, delta=0.05)

    def test_short_return_arc_does_not_crash(self):
        build(["-x", "25", "-y", "25", "-z", "20", "-l", "POPE", "-curv", "0.1,0.3,-1"])


class TestIonsAndBalancing(unittest.TestCase):
    def test_no_ions_warns_instead_of_crashing(self):
        log, counts, _ = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPE:8", "-l", "POPG:2"], ions=False)
        self.assertIn("WARNING: Compartment 0 is left with a net charge", log)

    def test_asym_counts_lipids(self):
        log, _, _ = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE:7", "-l", "POPG:2", "-l", "CARD:1", "-asym", "20"])
        (up, lo), = leaflet_counts(log)
        self.assertEqual(lo - up, 20)

    def test_balance_number(self):
        log, _, _ = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE", "-u", "LIPA", "-balance", "number",
                           "-posi_c0", "CA"])
        (up, lo), = leaflet_counts(log)
        self.assertEqual(up, lo)

    def test_mismatch_warning(self):
        log, _, _ = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE", "-asym", "30"])
        self.assertIn("WARNING: The leaflet lipid areas differ by more than 5%", log)


class TestLipidAreas(unittest.TestCase):
    def setUp(self):
        self.areas = leaflet_check.load_lipid_areas(INSANE)

    def test_table_and_fallbacks(self):
        get = self.areas["get_apl"]
        self.assertEqual(get("POPE"), 0.58)
        self.assertEqual(get("POPC.o"), 0.64)  # variant takes parent value
        self.assertEqual(get("XXPC"), 0.64)  # PC class
        self.assertEqual(get("DMPC"), 0.60)  # lipidome benchmark (experiment)
        self.assertEqual(get("DLPC"), 0.74)  # lipidome 18:2 PC, from the Martini 3 bilayers
        self.assertEqual(get("DLPC.o"), 0.61)  # Martini 3.0 12:0 PC
        self.assertAlmostEqual(get("DOEC"), 0.67 * 1.015)  # ether, from DOPC
        self.assertAlmostEqual(get("POLE"), 0.58 * 1.05)  # plasmalogen, from POPE
        self.assertEqual(get("NCER"), 0.40)  # ceramide class
        self.assertEqual(get("PSM"), 0.60)
        self.assertEqual(get("POP6"), 0.66)  # PI(4,5)P2
        self.assertEqual(get("XXXX"), 0.60)  # default

    def test_adjustments_skip_lps_and_sterols(self):
        self.areas["adjust_apls"](323.0, "m3")
        get = self.areas["get_apl"]
        self.assertAlmostEqual(get("POPC"), 0.64 * 1.05 * 0.94, places=6)
        self.assertEqual(get("LIPA"), 1.56)
        self.assertEqual(get("CHL1"), 0.40)


def membrane_charge(log):
    return float(re.search(r"; Charge of membrane: (\S+)", log).group(1))


def residue_beads(gro, resname):
    """Bead names of the first residue with the given name in a .gro file."""
    with open(gro) as f:
        lines = f.read().splitlines()[2:-1]
    beads, first = [], None
    for line in lines:
        if line[5:10].strip() == resname:
            if first is None:
                first = line[0:5]
            if line[0:5] != first:
                break
            beads.append(line[10:15].strip())
    return beads


class TestLipidome(unittest.TestCase):
    def test_lipidome_classes_build_with_itp_bead_names(self):
        log, counts, gro = build(["-x", "12", "-y", "12", "-z", "12", "-l", "POPC:4", "-l", "PSM:2", "-l", "CHOL:2", "-l", "DOEC:1",
                                  "-l", "POLE:1", "-u", "NCER:1", "-u", "TOCL:1", "-u", "DO2B:1", "-u", "DOTAP:1", "-u", "POPC:6"])
        for name in ("PSM", "CHOL", "DOEC", "POLE", "NCER", "TOCL", "DO2B", "DOTAP"):
            self.assertGreater(counts.get(name, 0), 0, name)
        # Bead names as in the Martini 3 lipidome itp files
        self.assertEqual(residue_beads(gro, "DOEC"), "NC3 PO4 ET1 ET2 C1A D2A C3A C4A C1B D2B C3B C4B".split())
        self.assertEqual(residue_beads(gro, "POLE"), "NH3 PO4 GL1 PL2 C1A D2A C3A C4A C1B C2B C3B C4B".split())
        self.assertEqual(residue_beads(gro, "PSM"), "NC3 PO4 OH1 AM2 T1A C2A C3A C4A C1B C2B C3B C4B".split())
        self.assertEqual(residue_beads(gro, "NCER")[:3], ["COH", "OH1", "AM2"])
        self.assertEqual(residue_beads(gro, "TOCL")[:4], ["GLC", "PO41", "GL11", "GL21"])
        expected = -2 * counts["TOCL"] - counts["DO2B"] + counts["DOTAP"]
        self.assertEqual(membrane_charge(log), expected)

    def test_phosphoinositide_charges(self):
        # PIPs were previously missing from the charge table, so they were not neutralised
        log, counts, gro = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPC:8", "-l", "POP6:1", "-l", "SAPI:1"])
        self.assertEqual(membrane_charge(log), -4 * counts["POP6"] - counts["SAPI"])
        self.assertEqual(residue_beads(gro, "POP6"), "C1 C2 C3 C4 PO4 P4 P5 GL1 GL2 C1A D2A C3A C4A C1B C2B C3B C4B".split())

    def test_lipid_a_and_cardiolipin_charges(self):
        log, counts, _ = build(["-x", "12", "-y", "12", "-z", "12", "-l", "POPE:6", "-l", "CDL0:1", "-l", "CDL1:1", "-l", "CDL2:1",
                                "-u", "LIPA", "-posi_c0", "CA"])
        self.assertEqual(membrane_charge(log), -4 * counts["LIPA"] - counts["CDL1"] - 2 * counts["CDL2"])

    def test_lipid_itp_definitions_and_warning(self):
        tmp = tempfile.mkdtemp(prefix="insane_itp_")
        TMP_DIRS.append(tmp)
        itp = os.path.join(tmp, "lipids.itp")
        with open(itp, "w") as f:
            # A new lipid defined by its ;@INSANE line, and a POPC whose bead names differ from the template
            f.write(";@INSANE alhead=S P, allink=E E, altail=CCCC CCCC, alname=XXES, charge=-1.0\n"
                    "[ moleculetype ]\nXXES 1\n[ atoms ]\n")
            for i, b in enumerate("CNO PO4 ET1 ET2 C1A C2A C3A C4A C1B C2B C3B C4B".split()):
                f.write("%d C1 1 XXES %s %d %s\n" % (i + 1, b, i + 1, "-1.0" if b == "PO4" else "0"))
            f.write("[ moleculetype ]\nPOPC 1\n[ atoms ]\n1 Q1 1 POPC NC3 1 1.0\n2 Q5 1 POPC PHO 2 -1.0\n")
        log, counts, gro = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPC:1", "-l", "XXES:1", "-lipid_itp", itp])
        self.assertGreater(counts["XXES"], 0)
        self.assertEqual(residue_beads(gro, "XXES"), "CNO PO4 ET1 ET2 C1A C2A C3A C4A C1B C2B C3B C4B".split())
        self.assertEqual(membrane_charge(log), -counts["XXES"])
        self.assertIn("WARNING: The bead names of POPC", log)


class TestLeafletCheck(unittest.TestCase):
    def run_check(self, gro):
        out = subprocess.run([sys.executable, LEAFLETS, gro, "-insane", INSANE], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_symmetric_and_asymmetric(self):
        _, _, gro = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE:7", "-l", "POPG:2", "-l", "CARD:1"])
        self.assertIn("no change suggested", self.run_check(gro))
        _, _, gro = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE:7", "-l", "POPG:2", "-l", "CARD:1", "-asym", "30"])
        self.assertIn("-asym -30", self.run_check(gro))


if __name__ == "__main__":
    unittest.main()
