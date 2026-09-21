"""Removes the decorative body_visuals subtree from simple_car.usd, reverting
to the plain, purely-functional chassis + 4 wheels car."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd

CAR_USD = "C:/Users/sanja/Desktop/thesis/Simulation_Environment_Setup/simple_car.usd"
BODY = "/World/SimpleCar/chassis/body_visuals"

stage = Usd.Stage.Open(CAR_USD)
if stage.GetPrimAtPath(BODY):
    stage.RemovePrim(Sdf.Path(BODY))
stage.Save()

with open("C:/Users/sanja/Desktop/thesis/remove_car_body_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
