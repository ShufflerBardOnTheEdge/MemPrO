"""
Leaflet area matching for membranes built with Insane4MemPrO.

Reads one or more frames (.gro) from a short NPT equilibration and, for each
membrane, compares the area available to each leaflet (box area minus the
protein's footprint in that leaflet) with the area its lipids should occupy
(from the Insane4MemPrO area-per-lipid table). Lipids cannot flip between
leaflets in Martini (except sterols), so a difference in how stretched the two
leaflets are persists, and can induce curvature. The suggested -asym moves
lipids so both leaflets are equally strained.

This is area matching, as used for asymmetric LPS outer membranes. It is an
approximation to setting zero differential stress between the leaflets
(Hossein & Deserno 2020), which would require lateral pressure profiles.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

WATER_IONS = {"W", "SOL", "PW", "TW", "SW", "NA", "CL", "CA", "K", "MG", "NA+", "CL-", "ION"}
AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
               "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HSD", "HSE", "HSP", "HIE", "HID", "HIP"}


def load_lipid_areas(insane_path=None):
    """Loads the area-per-lipid table block from Insane4MemPrO.py so both use the same values."""
    if insane_path is None:
        insane_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Insane4MemPrO.py")
    src = open(insane_path).read()
    start = src.index("# BEGIN LIPID AREAS")
    end = src.index("# END LIPID AREAS")
    ns = {}
    exec(src[start:end], ns)
    return ns


def read_gro_frames(paths):
    """Yields (resid, resname, atomname, xyz, box) for every frame in the given .gro files."""
    for path in paths:
        with open(path) as f:
            lines = f.read().splitlines()
        i = 0
        while i + 1 < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            n = int(lines[i + 1])
            atoms = lines[i + 2:i + 2 + n]
            box = np.array(lines[i + 2 + n].split()[:3], dtype=float)
            resid = np.array([int(a[0:5]) for a in atoms])
            resname = np.array([a[5:10].strip() for a in atoms])
            atomname = np.array([a[10:15].strip() for a in atoms])
            xyz = np.array([[float(a[20:28]), float(a[28:36]), float(a[36:44])] for a in atoms])
            yield resid, resname, atomname, xyz, box
            i += n + 3


def split_molecules(resid, resname, mask):
    """Groups atom indices of the selected atoms into residues (consecutive atoms with the same resid and name)."""
    idx = np.where(mask)[0]
    mols = []
    for k, i in enumerate(idx):
        if k == 0 or resid[i] != resid[idx[k - 1]] or resname[i] != resname[idx[k - 1]] or i != idx[k - 1] + 1:
            mols.append([i])
        else:
            mols[-1].append(i)
    return mols


def footprint(points_xy, box_xy, radius, spacing=0.1):
    """Area (nm^2) of the box within radius of any of the points, periodic in x and y."""
    if len(points_xy) == 0:
        return 0.0
    gx = np.arange(spacing / 2, box_xy[0], spacing)
    gy = np.arange(spacing / 2, box_xy[1], spacing)
    grid = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    tree = cKDTree(points_xy % box_xy, boxsize=box_xy)
    d, _ = tree.query(grid)
    return (d < radius).mean() * box_xy[0] * box_xy[1]


def analyse_frame(frame, areas, radius, lipid_names=None):
    resid, resname, atomname, xyz, box = frame
    is_prot = np.isin(resname, list(AMINO_ACIDS))
    if lipid_names:
        is_lip = np.isin(resname, list(lipid_names))
    else:
        is_lip = ~is_prot & ~np.isin(resname, list(WATER_IONS))
    lipids = split_molecules(resid, resname, is_lip)
    names = np.array([resname[m[0]] for m in lipids])
    # The first bead of each lipid is in its headgroup (PO4/GL0/PO1, ROH for sterols). Head beads form one layer per
    # leaflet with the ~3-4 nm hydrophobic core between them, so membranes are split at gaps over 5.5 nm in head z
    # and leaflets at the largest gap within a membrane (this also works for LPS, whose last beads are sugars)
    heads = np.array([xyz[m[0]] for m in lipids])
    order = np.argsort(heads[:, 2])
    breaks = np.where(np.diff(heads[order, 2]) > 5.5)[0]
    groups = np.split(order, breaks + 1)
    results = []
    for g in groups:
        hz = heads[g, 2]
        split = np.argmax(np.diff(hz)) if len(g) > 1 else 0
        lower, upper = g[:split + 1], g[split + 1:]
        z_up = np.mean(heads[upper, 2]) if len(upper) else hz.max()
        z_lo = np.mean(heads[lower, 2]) if len(lower) else hz.min()
        mid = (z_up + z_lo) / 2
        prot = xyz[is_prot]
        leaf = {}
        for label, sel, z0, z1 in (("upper", upper, mid, z_up + 0.5), ("lower", lower, z_lo - 0.5, mid)):
            # Largest protein cross-section in 0.5 nm slices from the midplane to just beyond the headgroups: the
            # cross-section of transmembrane helices (projecting the whole leaflet would over-count tilted ones), or
            # the area covered by a domain resting on the headgroups, where no lipid fits underneath
            edges = np.linspace(z0, z1, max(2, int(round((z1 - z0) / 0.5)) + 1))
            fp = np.max([footprint(prot[(prot[:, 2] > a) & (prot[:, 2] < b)][:, :2], box[:2], radius)
                          for a, b in zip(edges[:-1], edges[1:])])
            counts = {}
            for n in names[sel]:
                counts[n] = counts.get(n, 0) + 1
            ref = sum(areas["get_apl"](n) * c for n, c in counts.items())
            leaf[label] = {"n": len(sel), "counts": counts, "free": box[0] * box[1] - fp, "protein": fp, "ref": ref}
        results.append({"z": mid, "box": box[0] * box[1], "leaflets": leaf})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mempro-leaflets", description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="Equilibrated frame(s) (.gro, single or multi-frame)")
    parser.add_argument("-apl", action="append", default=[], help="Area per lipid override NAME:AREA (as in Insane4MemPrO)")
    parser.add_argument("-temp", type=float, default=None, help="Temperature (K) for the built-in areas (as in Insane4MemPrO)")
    parser.add_argument("-apl_ff", choices=["exp", "m3"], default="exp", help="Built-in areas: exp or m3 (as in Insane4MemPrO)")
    parser.add_argument("-radius", type=float, default=0.26, help="Protein bead radius (nm) for its cross-section (Default: 0.26, a Martini bead)")
    parser.add_argument("-lipids", default=None, help="Comma-separated lipid residue names (default: all residues that are not protein, water or ions)")
    parser.add_argument("-insane", default=None, help="Path to Insane4MemPrO.py to take the area table from (default: the installed copy)")
    args = parser.parse_args(argv)

    areas = load_lipid_areas(args.insane)
    if args.temp is not None or args.apl_ff == "m3":
        areas["adjust_apls"](args.temp, args.apl_ff)
    for ap in args.apl:
        name, val = ap.split(":")
        areas["lipid_apl"][name] = float(val)
    lipid_names = set(args.lipids.split(",")) if args.lipids else None

    frames = [analyse_frame(f, areas, args.radius, lipid_names) for f in read_gro_frames(args.files)]
    if not frames:
        sys.exit("No frames read")
    print("Read %d frame(s)" % len(frames))
    for mi in range(len(frames[0])):
        mems = [fr[mi] for fr in frames if len(fr) > mi]
        print("\nMembrane %d (z ~ %.1f nm), mean box area %.1f nm^2" % (mi + 1, np.mean([m["z"] for m in mems]), np.mean([m["box"] for m in mems])))
        strain = {}
        for label in ("upper", "lower"):
            free = np.mean([m["leaflets"][label]["free"] for m in mems])
            prot = np.mean([m["leaflets"][label]["protein"] for m in mems])
            l0 = mems[0]["leaflets"][label]
            comp = ", ".join("%s %d" % kv for kv in sorted(l0["counts"].items()))
            strain[label] = free / l0["ref"] - 1 if l0["ref"] > 0 else 0.0
            print("  %s leaflet: %d lipids (%s)" % (label.capitalize(), l0["n"], comp))
            print("    available area %.1f nm^2 (protein %.1f), lipid reference area %.1f nm^2, strain %+.1f%%"
                  % (free, prot, l0["ref"], 100 * strain[label]))
        up, lo = mems[0]["leaflets"]["upper"], mems[0]["leaflets"]["lower"]
        diff = strain["upper"] - strain["lower"]
        print("  Differential strain (upper - lower): %+.1f%%" % (100 * diff))
        # Number of lipids to remove from the more compressed leaflet so both are equally strained
        if abs(diff) < 0.005 or up["n"] == 0 or lo["n"] == 0:
            print("  Leaflets are matched to within 0.5%; no change suggested.")
            continue
        if diff < 0:
            # upper leaflet more compressed: remove upper lipids until free_up/ref_up' = free_lo/ref_lo
            target = np.mean([m["leaflets"]["upper"]["free"] for m in mems]) / (1 + strain["lower"])
            n_move = int(round((up["ref"] - target) / (up["ref"] / up["n"])))
            print("  Upper leaflet is more compressed: remove ~%d lipids from it (Insane4MemPrO: -asym %d)" % (n_move, n_move))
        else:
            target = np.mean([m["leaflets"]["lower"]["free"] for m in mems]) / (1 + strain["upper"])
            n_move = int(round((lo["ref"] - target) / (lo["ref"] / lo["n"])))
            print("  Lower leaflet is more compressed: remove ~%d lipids from it (Insane4MemPrO: -asym -%d)" % (n_move, n_move))
        print("  (-asym removes lipids at random from the built leaflet; rebuild and re-check after equilibration)")


if __name__ == "__main__":
    main()
