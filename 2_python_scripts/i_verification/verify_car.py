import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd

stage = Usd.Stage.Open(f"{ROOT}/1_simulation_scenes/simple_car.usd")
with open(f"{ROOT}/run_output/verify_car_result.txt", "w") as f:
    for prim in stage.Traverse():
        f.write(f"{prim.GetPath()} {prim.GetTypeName()} {list(prim.GetAppliedSchemas())}\n")
    f.flush()

simulation_app.close()
