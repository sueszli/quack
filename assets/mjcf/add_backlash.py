#!/usr/bin/env python3
"""Inject gearbox-backlash joints into an onshape-to-robot MJCF export.

For every actuated servo joint (``class="chosen_actuator"``) this inserts an
unactuated hinge on the same body / same axis right after it:

    <joint axis="0 0 1" name="left_hip_yaw" ... class="chosen_actuator"/>
    <joint axis="0 0 1" name="passive_left_hip_yaw_backlash" class="backlash"/>

The composite link rotation is main + backlash: the main joint is the servo
output (BAM drives it), the backlash joint is the play between the servo and
the link, free to wander within ±(backlash/2).

Naming: the ``passive_`` prefix means the new joints are automatically excluded
by every existing regex in the task configs (actuators ``^(?!passive_).*``,
joint obs, pose reward). The encoder-through-backlash handling lives on the
mjlab side (BacklashEncoderBamActuatorCfg + joint_pos/vel_rel_backlash obs).

Meant to run as the LAST post_import_command of an onshape-to-robot config
(see config_mjcf_groundcontact_backlash.json), but works standalone on any
already-exported robot xml:

    python3 ../add_backlash.py robot_groundcontact_backlash.xml --backlash-deg 2.0   # run from the model dir

``--backlash-deg`` is the TOTAL peak-to-peak play (what you measure wiggling
the horn with the servo held); the joint range is symmetric ±deg/2.

Edits the file in place, and refuses to run twice (the pipeline re-exports from
Onshape each time, so a second pass means something is wired wrong).
"""

import argparse
import math
import re
import sys

SERVO_CLASS = "chosen_actuator"
JOINT_RE = re.compile(r'^(\s*)<joint\b[^>]*/>\s*$')
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# solreflimit: with a range this small MuJoCo's default solref (0.02,1) lets the
# joint overshoot its limits ~2x under load, i.e. double the play we asked for.
# 0.01 = 2*sim_dt (mjlab velocity tasks run dt=0.005) is the stiffest stable
# setting; solimp raises the impedance so gear-teeth contact is nearly rigid.
DEFAULTS_BLOCK = """\
  <!-- Backlash injected by add_backlash.py: {total:g} deg total play (symmetric +/-{half_deg:g} deg) -->
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="MJCF file to modify in place")
    parser.add_argument("--backlash-deg", type=float, default=2.0,
                        help="TOTAL backlash play in degrees (peak-to-peak); "
                             "joint range is symmetric +/-deg/2 (default: 2.0)")
    args = parser.parse_args()

    half = math.radians(args.backlash_deg) / 2.0

    with open(args.xml) as f:
        lines = f.readlines()

    if any('class="backlash"' in line for line in lines):
        print(f"[add_backlash] {args.xml} already contains backlash joints — aborting.")
        return 1
    if not any("<worldbody>" in line for line in lines):
        print("[add_backlash] ERROR: no <worldbody> found — is this an MJCF file?")
        return 1

    out: list[str] = []
    added: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and "<worldbody>" in line:
            out.append(DEFAULTS_BLOCK.format(
                total=args.backlash_deg, half_deg=args.backlash_deg / 2,
                lo=-half, hi=half))
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
        out.append(
            f'{m.group(1)}<joint axis="{attrs.get("axis", "0 0 1")}"{pos} '
            f'name="passive_{name}_backlash" type="hinge" class="backlash"/>\n'
        )
        added.append(name)

    if not added:
        print(f'[add_backlash] ERROR: no joints with class="{SERVO_CLASS}" found.')
        return 1

    with open(args.xml, "w") as f:
        f.writelines(out)

    print(f"[add_backlash] added {len(added)} backlash joints "
          f"(+/-{args.backlash_deg / 2:g} deg = +/-{half:.5f} rad) to {args.xml}: "
          f"{', '.join(added)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
