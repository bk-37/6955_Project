"""
Extract total body mass from scaled OpenSim .osim model files.
Usage: python extract_osim_mass.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

def get_total_mass(osim_path: Path) -> float:
    tree = ET.parse(osim_path)
    root = tree.getroot()
    total = 0.0
    for body in root.iter("Body"):
        mass_el = body.find("mass")
        if mass_el is not None:
            total += float(mass_el.text.strip())
    return total

# Point this at wherever your scaled osim models live
OSIM_ROOT = Path(".")  # change to your Ulrich data root

osim_files = sorted(OSIM_ROOT.rglob("*.osim"))
if not osim_files:
    print("No .osim files found — update OSIM_ROOT")
else:
    print(f"{'File':<60} {'Mass (kg)':>10}")
    print("-" * 72)
    for f in osim_files:
        try:
            mass = get_total_mass(f)
            if mass > 0:
                print(f"{str(f.relative_to(OSIM_ROOT)):<60} {mass:>10.2f}")
        except Exception as e:
            print(f"{f.name:<60} ERROR: {e}")

WALKER2D_MASS = 23.68
print(f"\nWalker2d mass: {WALKER2D_MASS:.2f} kg")
print(f"For BW-normalized GRF comparison:")
print(f"  sim_GRF_BW   = sim_contact_force   / ({WALKER2D_MASS:.2f} * 9.81)")
print(f"  subj_GRF_BW  = forceplate_force    / (subject_mass * 9.81)")
print(f"Both are dimensionless — mass scaling cancels out.")
