from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage, get_current_stage

CAR_USD = "C:/Users/sanja/Desktop/thesis/Simulation/Simulation_Environment_Setup/simple_car.usd"

with open("C:/Users/sanja/Desktop/thesis/diagnose_result.txt", "w") as f:
    f.write("--- after open_stage, before World() ---\n")
    open_stage(CAR_USD)
    stage = get_current_stage()
    for prim in stage.Traverse():
        f.write(f"{prim.GetPath()} {prim.GetTypeName()}\n")

    f.write("\n--- after World() + reset ---\n")
    world = World(stage_units_in_meters=1.0)
    world.reset()
    stage2 = get_current_stage()
    f.write(f"same stage object as before? {stage2 == stage}\n")
    f.write(f"stage2 root layer identifier: {stage2.GetRootLayer().identifier}\n")
    for prim in stage2.Traverse():
        f.write(f"{prim.GetPath()} {prim.GetTypeName()}\n")

    f.write("\n--- world positions of any Xformable under /World ---\n")
    for prim in stage2.Traverse():
        p = str(prim.GetPath())
        if p.count("/") == 2 and p.startswith("/World/"):
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                f.write(f"{p}: translation={world_xf.ExtractTranslation()}\n")
    f.flush()

simulation_app.close()
