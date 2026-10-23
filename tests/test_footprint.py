"""
Tests for the per-leaflet protein footprint (-footprint map, docs/footprint-design.md).

Run from the repository root with:  python -m unittest discover tests
The synthetic-shape tests call the footprint block directly (well under a second each); the build tests run
Insane4MemPrO on small synthetic proteins (a few seconds each). The QmcA test builds a 234k-bead system and only runs
with MEMPRO_SLOW_TESTS=1.
"""

import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSANE = os.path.join(ROOT, "mempro", "Insane4MemPrO.py")
LEAFLETS = os.path.join(ROOT, "mempro", "leaflet_check.py")
QMCA = os.path.join(ROOT, "tests", "data", "QmcA")
sys.path.insert(0, os.path.join(ROOT, "mempro"))
import leaflet_check  # noqa: E402

FP = leaflet_check.load_footprint(INSANE)
R = FP["FP_RADII"]["R"]
HALF = 1.9  # leaflet half-thickness (midplane to phosphate plane) used for the shapes
BOX = (10.0, 10.0)
TMP_DIRS = []


def tearDownModule():
    for d in TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


def solid(radius_at, z0=-3.0, z1=3.0, dz=0.3, dr=0.4, centre=(5.0, 5.0)):
    """Beads filling a solid of revolution about a vertical axis, with radius radius_at(z), in layers dz apart."""
    pts = []
    for z in np.arange(z0, z1 + 1e-9, dz):
        rmax = radius_at(z)
        for r in np.append(np.arange(0, rmax, dr), rmax):
            n = max(1, int(round(2 * np.pi * r / dr)))
            for a in np.arange(n) * 2 * np.pi / n:
                pts.append([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), z])
    return np.array(pts)


def ring(radius, spacing, centre=(5.0, 5.0), z0=-3.0, z1=3.0):
    """A hollow cylinder of beads spacing nm apart."""
    angles = np.arange(0, 2 * np.pi, spacing / radius)
    return np.array([[centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), z]
                     for z in np.arange(z0, z1 + 1e-9, 0.3) for a in angles])


def disc(radius, z, centre=(5.0, 5.0)):
    return solid(lambda _: radius, z, z, centre=centre)


def core_z():
    """Heights of the core slices of an upper leaflet (as in footprint_maps)."""
    n = int(np.ceil(HALF / FP["FP_SLICE"]))
    return (np.arange(n) + 0.5) / n * HALF


def swept(r, slope):
    """Cross-section radius of a bead-swept surface: the bead radius is horizontal only on a vertical wall."""
    return r + R * np.sqrt(1 + slope ** 2)


def maps(pts, **kw):
    return FP["footprint_maps"](pts, np.full(len(pts), R), BOX, 0.0, HALF, HALF, **kw)


class TestSyntheticShapes(unittest.TestCase):
    def assertArea(self, value, expected, rel=0.05):
        self.assertAlmostEqual(value, expected, delta=rel * expected, msg="%.2f vs %.2f" % (value, expected))

    def test_cylinder_is_pi_r_squared(self):
        for r0 in (1.0, 2.0):
            m = maps(solid(lambda z: r0))
            for leaf in ("upper", "lower"):
                self.assertArea(m[leaf]["footprint"], np.pi * (r0 + R) ** 2)
                self.assertAlmostEqual(m[leaf]["free"], 100 - m[leaf]["footprint"], places=6)

    def test_no_protein(self):
        m = maps(np.zeros((0, 3)))
        self.assertAlmostEqual(m["upper"]["footprint"], 0.0)
        self.assertAlmostEqual(m["lower"]["free"], 100.0)

    def test_cone_leaflets_differ_as_predicted(self):
        # Radius grows upwards. The upper leaflet is set by the widest headgroup slice (no lipid head fits under the
        # protein there), the lower one by the mean core cross-section (its head layer is narrower than its core)
        cone = lambda z: 1.5 + 0.3 * z
        m = maps(solid(cone))
        head_top = HALF + FP["FP_HEAD"] * (1 - 0.5 / round(FP["FP_HEAD"] / FP["FP_SLICE"]))
        upper = np.pi * swept(cone(head_top), 0.3) ** 2
        lower = np.mean(np.pi * swept(cone(-core_z()), 0.3) ** 2)
        self.assertArea(m["upper"]["footprint"], upper)
        self.assertArea(m["lower"]["footprint"], lower)
        self.assertArea(m["upper"]["footprint"] - m["lower"]["footprint"], upper - lower, rel=0.1)

    def test_hourglass_is_symmetric(self):
        m = maps(solid(lambda z: 1.0 + 0.4 * abs(z)))
        self.assertArea(m["upper"]["footprint"], m["lower"]["footprint"], rel=0.01)

    def test_bulge_counts_core_volume_not_largest_section(self):
        # Widest at the midplane: the footprint is the mean core cross-section, well below the largest one
        bulge = lambda z: 2.0 - 0.4 * abs(z)
        m = maps(solid(bulge))
        mean_core = np.mean(np.pi * swept(bulge(core_z()), 0.4) ** 2)
        for leaf in ("upper", "lower"):
            self.assertArea(m[leaf]["footprint"], mean_core)
            self.assertLess(m[leaf]["footprint"], 0.8 * np.pi * (2.0 + R) ** 2)

    def test_domain_on_headgroups_and_above(self):
        # A soluble domain resting in the upper headgroup layer occupies that leaflet only
        m = maps(disc(2.8, HALF + 0.3))
        self.assertArea(m["upper"]["footprint"], np.pi * (2.8 + R) ** 2)
        self.assertAlmostEqual(m["lower"]["footprint"], 0.0)
        # One 1.5 nm above the phosphate plane leaves room for the lipids beneath it
        m = maps(disc(2.8, HALF + 1.5))
        self.assertAlmostEqual(m["upper"]["footprint"], 0.0)

    def test_closed_barrel_interior_is_excluded_unless_ring(self):
        barrel = ring(1.5, 0.35)
        closed = maps(barrel)["upper"]["footprint"]
        self.assertArea(closed, np.pi * (1.5 + R) ** 2)
        opened = maps(barrel, enclosed_free=True)["upper"]["footprint"]
        self.assertLess(opened, 0.65 * closed)
        self.assertGreater(opened, np.pi * ((1.5 + R) ** 2 - (1.5 - R) ** 2))

    def test_open_ring_interior_is_free(self):
        # Gaps between the beads let lipids in (as for the QmcA ring)
        m = maps(ring(4.0, 1.2))
        self.assertLess(m["upper"]["footprint"], 0.2 * np.pi * (4.0 + R) ** 2)

    def test_periodic_boundaries(self):
        centred = maps(solid(lambda z: 1.5))["upper"]["footprint"]
        corner = maps(solid(lambda z: 1.5, centre=(0.0, 0.0)))["upper"]["footprint"]
        self.assertAlmostEqual(centred, corner, delta=0.01 * centred)

    def test_bead_radii(self):
        radii = FP["fp_bead_radii"](["BB", "SC1", "CA", "BB", "SC2", "BB"], ["LEU", "ALA", "LEU", "GLY", "TRP", "DUM"])
        self.assertEqual(list(radii), [FP["FP_RADII"]["R"], FP["FP_RADII"]["T"], 0.0, FP["FP_RADII"]["S"], FP["FP_RADII"]["T"], 0.0])


class TestSiteSelection(unittest.TestCase):
    def test_exact_count_and_no_sites_on_protein(self):
        import random
        m = maps(solid(lambda z: 1.5))
        place = m["upper"]["place"]
        xy = np.random.RandomState(0).uniform(0, 10, (150, 2))
        for n_target in (100, 160):
            keep, added, spare = FP["fp_select_sites"](xy, place, BOX, n_target, 10, random.Random(1))
            self.assertEqual(len(keep) + len(added), n_target)
            self.assertEqual(len(spare), 10)
            sites = np.concatenate([xy[keep], added, spare])
            self.assertTrue(np.all(FP["fp_lookup"](place, sites, BOX) >= 0.5))


def write_pdb(path, pts):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(pts * 10):
            f.write("ATOM  %5d  BB  LEU A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n" % (i + 1, i + 1, x, y, z))
        f.write("END\n")


def build(pts, args):
    tmp = tempfile.mkdtemp(prefix="footprint_test_")
    TMP_DIRS.append(tmp)
    write_pdb(os.path.join(tmp, "prot.pdb"), pts)
    cmd = [sys.executable, INSANE, "-f", "prot.pdb", "-x", "10", "-y", "10", "-z", "10", "-l", "POPE", "-sol", "W",
           "-o", "sys.gro", "-p", "topol.top"] + args
    out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    log = out.stdout + out.stderr
    if out.returncode != 0:
        raise AssertionError("Insane4MemPrO failed:\n" + log[-3000:])
    return log, os.path.join(tmp, "sys.gro")


def map_leaflets(log):
    """(footprint, free area, sites) of the upper and lower leaflets from the -footprint map log."""
    return [tuple(map(float, m)) for m in
            re.findall(r"leaflet: protein footprint (\S+) nm\^2, free area (\S+) nm\^2, (\d+) sites", log)]


def leaflet_counts(log):
    return [tuple(map(int, m)) for m in re.findall(r"; (\d+) lipids in upper leaflet, (\d+) lipids in lower leaflet", log)]


class TestBuildWithMap(unittest.TestCase):
    def test_cylinder_build_fills_free_area(self):
        log, gro = build(solid(lambda z: 1.5), ["-footprint", "map"])
        (fp_up, free_up, n_up), (fp_lo, free_lo, n_lo) = map_leaflets(log)
        self.assertAlmostEqual(fp_up, np.pi * (1.5 + R) ** 2, delta=0.05 * np.pi * (1.5 + R) ** 2)
        self.assertEqual(n_up, round(free_up / 0.58))
        self.assertEqual(leaflet_counts(log), [(n_up, n_lo)])
        self.assertNotIn("WARNING: Only", log)
        # No lipid head inside the protein
        with open(gro) as f:
            lines = f.read().splitlines()[2:-1]
        heads = np.array([[float(l[20:28]), float(l[28:36])] for l in lines if l[5:10].strip() == "POPE" and l[10:15].strip() == "NH3"])
        prot = np.array([[float(l[20:28]), float(l[28:36])] for l in lines if l[5:10].strip() == "LEU"])
        r = np.linalg.norm(heads - prot.mean(0), axis=1)
        self.assertGreater(r.min(), 1.5)

    def test_cone_build_leaflet_counts(self):
        log, _ = build(solid(lambda z: 1.5 + 0.3 * z), ["-footprint", "map", "-mt", "3.8"])
        self.assertIn("phosphate planes +1.90/-1.90 nm (from -mt)", log)
        (fp_up, _, n_up), (fp_lo, _, n_lo) = map_leaflets(log)
        self.assertEqual(leaflet_counts(log), [(n_up, n_lo)])
        self.assertAlmostEqual(n_lo - n_up, (fp_up - fp_lo) / 0.58, delta=1)
        self.assertGreater(n_lo - n_up, 15)

    def test_checker_agrees_with_builder(self):
        # Without -mt both take the leaflet planes from the lipids as built, so the leaflets are matched
        _, gro = build(solid(lambda z: 1.5 + 0.3 * z), ["-footprint", "map"])
        out = subprocess.run([sys.executable, LEAFLETS, gro, "-insane", INSANE], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("no change suggested", out.stdout)

    def test_fp_map_file(self):
        tmp = tempfile.mkdtemp(prefix="footprint_npz_")
        TMP_DIRS.append(tmp)
        npz = os.path.join(tmp, "fp.npz")
        build(solid(lambda z: 1.5), ["-footprint", "map", "-fp_map", npz])
        data = np.load(npz)
        self.assertEqual(data["upper0_avail"].shape, (100, 100))
        self.assertAlmostEqual(data["lower0_place"].min(), 0.0)


@unittest.skipUnless(os.environ.get("MEMPRO_SLOW_TESTS"), "set MEMPRO_SLOW_TESTS=1 to build QmcA (234k beads)")
class TestQmcA(unittest.TestCase):
    def test_leaflets_matched(self):
        tmp = tempfile.mkdtemp(prefix="qmca_test_")
        TMP_DIRS.append(tmp)
        with gzip.open(os.path.join(QMCA, "protein-cg_rank_1.pdb.gz"), "rt") as f, open(os.path.join(tmp, "prot.pdb"), "w") as g:
            g.write(f.read())
        cmd = [sys.executable, INSANE, "-f", "prot.pdb", "-x", "32.11502", "-y", "32.262005", "-z", "24.711554",
               "-l", "POPE:8", "-l", "POPG:2", "-sol", "W", "-negi_c0", "CL", "-posi_c0", "NA", "-footprint", "map",
               "-o", "sys.gro", "-p", "topol.top"]
        out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        (fp_up, _, _), (fp_lo, _, _) = map_leaflets(out.stdout + out.stderr)
        self.assertGreater(fp_lo, 3 * fp_up)  # the soluble cap rests on the lower headgroups
        chk = subprocess.run([sys.executable, LEAFLETS, os.path.join(tmp, "sys.gro"), "-insane", INSANE], capture_output=True, text=True)
        diff = float(re.search(r"Differential strain \(upper - lower\): (\S+)%", chk.stdout).group(1))
        self.assertLess(abs(diff), 1.0)


if __name__ == "__main__":
    unittest.main()
