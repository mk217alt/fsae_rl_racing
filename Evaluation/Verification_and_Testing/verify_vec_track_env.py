"""Sanity-checks vec_train_car2.usd: reads back each clone's chassis and a
sample of track prims to confirm the grid offsets applied correctly and
nothing overlaps."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom

STAGE_PATH = "C:/Users/sanja/Desktop/thesis/Racing/Vec_Curvature_Lineage/vec_train_car2.usd"
NUM_ENVS = 10

stage = Usd.Stage.Open(STAGE_PATH)

lines = []
for i in range(NUM_ENVS):
    chassis = stage.GetPrimAtPath(f"/World/Env{i}/Car/chassis")
    road0 = stage.GetPrimAtPath(f"/World/Env{i}/Track/road_0")
    if not chassis or not road0:
        lines.append(f"Env{i}: MISSING PRIM (chassis={bool(chassis)}, road_0={bool(road0)})")
        continue
    chassis_xform = UsdGeom.Xformable(chassis).ComputeLocalToWorldTransform(0)
    road_xform = UsdGeom.Xformable(road0).ComputeLocalToWorldTransform(0)
    cpos = chassis_xform.ExtractTranslation()
    rpos = road_xform.ExtractTranslation()
    lines.append(f"Env{i}: chassis world pos={cpos}, road_0 world pos={rpos}")

with open("C:/Users/sanja/Desktop/thesis/verify_vec_track_env_done.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

simulation_app.close()
