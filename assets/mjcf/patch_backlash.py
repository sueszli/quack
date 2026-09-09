# Adds a passive backlash hinge next to each servo joint in an MJCF export, so sim has the real robot's gear play.
import argparse
import math
import re
import sys

SERVO_CLASS = "chosen_actuator"
JOINT_RE = re.compile(r"^(\s*)<joint\b[^>]*/>\s*$")
OPEN_JOINT_RE = re.compile(r"^\s*<joint\b(?![^>]*/>)")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

DEFAULTS_BLOCK = """\
  <!-- Backlash injected by patch_backlash.py: {total:g} deg total play (symmetric +/-{half_deg:g} deg) -->
  <default>
    <default class="backlash">
      <!-- stiff limit constraint: with a range this small the default
           solref (0.02,1) lets the joint overshoot its limits ~2x under
           load. 0.01 = 2*sim_dt (mjlab velocity tasks run dt=0.005),
           the stiffest stable setting; solimp raises the impedance so
           the gear-teeth contact is nearly rigid. -->
      <joint damping="0.01" frictionloss="0" armature="0.001" limited="true" \
range="{lo:.17g} {hi:.17g}" solreflimit="0.01 1" solimplimit="0.95 0.999 0.0001 0.5 2"/>
    </default>
  </default>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch an onshape-to-robot MJCF export in place, adding one passive backlash hinge per servo joint. Must run as the LAST post_import_command, after the sed passes.")
    parser.add_argument("xml", help="MJCF file to modify in place")
    parser.add_argument("--backlash-deg", type=float, default=2.0, help="TOTAL backlash play in degrees (peak-to-peak); joint range is symmetric +/-deg/2 (default: 2.0)")
    args = parser.parse_args()

    half = math.radians(args.backlash_deg) / 2.0

    with open(args.xml) as f:
        lines = f.readlines()

    if any('class="backlash"' in line for line in lines):
        print(f"[patch_backlash] {args.xml} already contains backlash joints — aborting.")
        return 1
    if not any("<worldbody>" in line for line in lines):
        print("[patch_backlash] ERROR: no <worldbody> found — is this an MJCF file?")
        return 1

    straddling = [i + 1 for i, line in enumerate(lines) if OPEN_JOINT_RE.match(line)]
    if straddling:
        print(f"[patch_backlash] ERROR: {args.xml} has <joint> elements spanning multiple lines (line(s) {', '.join(map(str, straddling))}). This line-based scanner would skip them and emit fewer backlash hinges than servos; aborting instead.")
        return 1

    out: list[str] = []
    added: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and "<worldbody>" in line:
            out.append(DEFAULTS_BLOCK.format(total=args.backlash_deg, half_deg=args.backlash_deg / 2, lo=-half, hi=half))
            inserted = True
        out.append(line)

        m = JOINT_RE.match(line)
        if m is None:
            continue
        attrs = dict(ATTR_RE.findall(line))
        name = attrs.get("name")
        if attrs.get("class") != SERVO_CLASS or not name:
            continue
        pos = f' pos="{attrs["pos"]}"' if "pos" in attrs else ""
        out.append(f'{m.group(1)}<joint axis="{attrs.get("axis", "0 0 1")}"{pos} name="passive_{name}_backlash" type="hinge" class="backlash"/>\n')
        added.append(name)

    if not added:
        print(f'[patch_backlash] ERROR: no joints with class="{SERVO_CLASS}" found.')
        return 1

    with open(args.xml, "w") as f:
        f.writelines(out)

    print(f"[patch_backlash] added {len(added)} backlash joints (+/-{args.backlash_deg / 2:g} deg = +/-{half:.5f} rad) to {args.xml}: {', '.join(added)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
