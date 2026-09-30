"""Quick readback check: confirm the road surface is now flush at z=0 and
the start/finish tiles exist, on both deploy stages."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom

lines = []
for label, path, road_prim, tile_prefix in [
    ("vec_deploy_car2.usd", f"{ROOT}/1_simulation_scenes/vec_deploy_car2.usd", "/World/Track/road", "/World/Track/start_finish_"),
    ("race_deploy.usd", f"{ROOT}/1_simulation_scenes/race_deploy.usd", "/World/Track/road", "/World/Track/start_finish_"),
]:
    stage = Usd.Stage.Open(path)
    road = stage.GetPrimAtPath(road_prim)
    if not road:
        lines.append(f"{label}: MISSING road prim at {road_prim}")
        continue
    xform = UsdGeom.Xformable(road)
    world_pos = xform.ComputeLocalToWorldTransform(0).ExtractTranslation()
    ops = xform.GetOrderedXformOps()
    scale = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale = op.Get()
    road_top_z = world_pos[2] + (scale[2] / 2.0 if scale else 0)
    lines.append(f"{label}: road center world z={world_pos[2]:.4f}, scale_z={scale[2] if scale else 'N/A'}, computed top z={road_top_z:.4f} (want ~0.0)")

    tile_count = 0
    for i in range(20):
        tile = stage.GetPrimAtPath(f"{tile_prefix}{i}")
        if tile:
            tile_count += 1
    lines.append(f"{label}: found {tile_count} start_finish tile prims")

with open(f"{ROOT}/run_output/verify_deploy_fixes_log.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

with open(f"{ROOT}/run_output/verify_deploy_fixes_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
