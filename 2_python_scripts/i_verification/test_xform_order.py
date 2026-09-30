import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf

stage = Usd.Stage.CreateInMemory()
cube = UsdGeom.Cube.Define(stage, "/test")
cube.CreateSizeAttr(1.0)
cube.AddScaleOp().Set(Gf.Vec3f(2.0, 1.0, 0.4))
cube.AddRotateZOp().Set(45.0)
cube.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))

with open(f"{ROOT}/run_output/test_xform_order_result.txt", "w") as f:
    xformable = UsdGeom.Xformable(cube.GetPrim())
    f.write(f"xformOpOrder: {[op.GetOpName() for op in xformable.GetOrderedXformOps()]}\n")
    m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    f.write(f"full matrix:\n{m}\n")
    f.write(f"extracted translation: {m.ExtractTranslation()}\n")
    f.flush()

simulation_app.close()
