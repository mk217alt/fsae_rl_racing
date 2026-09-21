from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd

stage = Usd.Stage.Open("C:/Users/sanja/Desktop/thesis/Simulation/Simulation_Environment_Setup/simple_car.usd")
with open("C:/Users/sanja/Desktop/thesis/verify_car_result.txt", "w") as f:
    for prim in stage.Traverse():
        f.write(f"{prim.GetPath()} {prim.GetTypeName()} {list(prim.GetAppliedSchemas())}\n")
    f.flush()

simulation_app.close()
