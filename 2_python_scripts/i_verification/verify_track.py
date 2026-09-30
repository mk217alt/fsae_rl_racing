import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf

CAR_USD = f"{ROOT}/1_simulation_scenes/simple_car.usd"
stage = Usd.Stage.Open(CAR_USD)

with open(f"{ROOT}/run_output/verify_track_result.txt", "w") as f:
    road_positions = []
    count = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/Track/road_"):
            count += 1
            xformable = UsdGeom.Xformable(prim)
            world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = world_xf.ExtractTranslation()
            road_positions.append((t[0], t[1]))

    f.write(f"total road segments: {count}\n")
    xs = [p[0] for p in road_positions]
    ys = [p[1] for p in road_positions]
    f.write(f"x range: {min(xs):.2f} to {max(xs):.2f} (span {max(xs)-min(xs):.2f} m)\n")
    f.write(f"y range: {min(ys):.2f} to {max(ys):.2f} (span {max(ys)-min(ys):.2f} m)\n")
    f.write("\nroad segment centers in order:\n")
    for x, y in road_positions:
        f.write(f"({x:.2f}, {y:.2f})\n")
    f.flush()

simulation_app.close()
