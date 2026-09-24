"""Removes the decorative body_visuals subtree from simple_car.usd, reverting
to the plain, purely-functional chassis + 4 wheels car."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd

CAR_USD = f"{ROOT}/1_simulation_scenes/simple_car.usd"
BODY = "/World/SimpleCar/chassis/body_visuals"

stage = Usd.Stage.Open(CAR_USD)
if stage.GetPrimAtPath(BODY):
    stage.RemovePrim(Sdf.Path(BODY))
stage.Save()

with open(f"{ROOT}/run_output/remove_car_body_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
