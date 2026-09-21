"""Duplicates /World/SimpleCar to /World/SimpleCar2 inside simple_car.usd,
remapping all internal relationship targets (joint body0/body1, material
bindings) so it is a fully independent car, then shifts it sideways on the
track so it doesn't overlap the first car.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd

CAR_USD = "C:/Users/sanja/Desktop/thesis/Simulation_Environment_Setup/simple_car.usd"
SRC = "/World/SimpleCar"
DST = "/World/SimpleCar2"
LATERAL_OFFSET = 3.0  # meters, sideways shift on the (9m-wide) track

# prims (relative to the car root) that carry their own world-space
# translate op and so need to be shifted along with the whole car
SHIFTED_PRIMS = [
    "chassis",
    "rear_left",
    "rear_right",
    "front_left_knuckle",
    "front_right_knuckle",
    "front_left",
    "front_right",
]

stage = Usd.Stage.Open(CAR_USD)

if stage.GetPrimAtPath(DST):
    stage.RemovePrim(Sdf.Path(DST))

root_layer = stage.GetRootLayer()
Sdf.CopySpec(root_layer, Sdf.Path(SRC), root_layer, Sdf.Path(DST))

# remap every relationship target that pointed into the original car's
# subtree (joint body0/body1 rels, material:binding rels, ...) to the new one
new_root = stage.GetPrimAtPath(DST)
for prim in Usd.PrimRange(new_root):
    for rel in prim.GetRelationships():
        targets = rel.GetTargets()
        if not targets:
            continue
        new_targets = []
        changed = False
        for t in targets:
            t_str = str(t)
            if t_str == SRC or t_str.startswith(SRC + "/"):
                new_targets.append(Sdf.Path(DST + t_str[len(SRC):]))
                changed = True
            else:
                new_targets.append(t)
        if changed:
            rel.SetTargets(new_targets)

# shift the duplicated car sideways so it sits in its own lane
for name in SHIFTED_PRIMS:
    prim = stage.GetPrimAtPath(f"{DST}/{name}")
    translate_attr = prim.GetAttribute("xformOp:translate")
    pos = translate_attr.Get()
    translate_attr.Set(Gf.Vec3d(pos[0], pos[1] + LATERAL_OFFSET, pos[2]))

stage.Save()

with open("C:/Users/sanja/Desktop/thesis/add_second_car_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
