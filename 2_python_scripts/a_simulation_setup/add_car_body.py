"""Adds a purely-visual Formula-style body around the existing simple_car.usd
chassis/wheel frame (no physics changes - just cosmetic geometry parented
under the chassis so it rides along with it).

v2: body_visuals resets the inherited xform stack so it does NOT pick up the
chassis Cube's non-uniform (2.0, 1.0, 0.4) box scale - that scale was
silently stretching/mispositioning every child shape in v1, which is why
the body looked disconnected from the chassis with huge gaps.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

CAR_USD = f"{ROOT}/1_simulation_scenes/simple_car.usd"
CAR = "/World/SimpleCar"
CHASSIS = f"{CAR}/chassis"
BODY = f"{CHASSIS}/body_visuals"

WHEEL_X_OFFSET = 0.8
WHEEL_Y_OFFSET = 0.6
WHEEL_RADIUS = 0.3
CHASSIS_Z = 0.45  # matches create_simple_car.py

stage = Usd.Stage.Open(CAR_USD)

body_mat = UsdShade.Material.Get(stage, f"{CAR}/Looks/body")
wheel_mat = UsdShade.Material.Get(stage, f"{CAR}/Looks/wheel")

# wipe any previous (broken) body_visuals subtree and rebuild clean
if stage.GetPrimAtPath(BODY):
    stage.RemovePrim(Sdf.Path(BODY))

body_xform = UsdGeom.Xform.Define(stage, BODY)
# reset the inherited xform stack so the chassis Cube's box scale does not
# warp/mis-position this whole subtree, then re-place it at the chassis's
# own world height (chassis has no rotation, so this is a plain translate)
body_xform.SetResetXformStack(True)
body_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CHASSIS_Z))


def bind(path, material):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material)


def add_box(name, position, size, material):
    path = f"{BODY}/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    bind(path, material)
    return path


def add_cone(name, position, radius, height, axis, material):
    path = f"{BODY}/{name}"
    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(height)
    cone.CreateAxisAttr(axis)
    cone.AddTranslateOp().Set(Gf.Vec3d(*position))
    bind(path, material)
    return path


def add_cylinder(name, position, radius, height, axis, material):
    path = f"{BODY}/{name}"
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    cyl.AddTranslateOp().Set(Gf.Vec3d(*position))
    bind(path, material)
    return path


wheel_z = WHEEL_RADIUS - CHASSIS_Z  # wheel center height, relative to chassis center

# nose cone, flush against the chassis front face (x = +1.0)
add_cone("nose", (1.35, 0.0, -0.03), radius=0.25, height=0.7, axis="X", material=body_mat)

# cockpit hump, flush on top of the chassis (chassis top is at local z = +0.2)
add_box("cockpit", (0.25, 0.0, 0.35), (0.5, 0.45, 0.3), body_mat)

# wide sidepods running the length between the wheels, reaching out to meet
# them (chassis side is at y = +-0.5; wheels sit at y = +-0.6)
add_box("side_pod_left", (-0.15, 0.68, wheel_z + 0.05), (1.5, 0.38, 0.42), body_mat)
add_box("side_pod_right", (-0.15, -0.68, wheel_z + 0.05), (1.5, 0.38, 0.42), body_mat)

# wheel arches: a body-colored ring concentric with each wheel (same axis
# and center as the wheel itself, slightly larger radius and a bit narrower
# than the wheel's width) so it is guaranteed to touch/overlap the wheel -
# the tire pokes out a little on each side, the rest reads as one piece
WHEEL_HALF_HEIGHT = 0.1
ARCH_RADIUS = WHEEL_RADIUS + 0.05
ARCH_WIDTH = WHEEL_HALF_HEIGHT * 2.0 * 0.7
for name, x in (("front", WHEEL_X_OFFSET), ("rear", -WHEEL_X_OFFSET)):
    for side, y in (("left", WHEEL_Y_OFFSET), ("right", -WHEEL_Y_OFFSET)):
        add_cylinder(
            f"{name}_{side}_wheel_arch",
            (x, y, wheel_z),
            radius=ARCH_RADIUS,
            height=ARCH_WIDTH,
            axis="Y",
            material=body_mat,
        )

# front wing + endplates
add_box("front_wing", (1.72, 0.0, -0.15), (0.12, 1.3, 0.06), wheel_mat)
add_box("front_wing_endplate_left", (1.72, 0.65, -0.15), (0.15, 0.03, 0.18), wheel_mat)
add_box("front_wing_endplate_right", (1.72, -0.65, -0.15), (0.15, 0.03, 0.18), wheel_mat)

# rear wing struts + wing + endplates
add_box("rear_wing_strut_left", (-1.15, 0.3, 0.375), (0.06, 0.06, 0.35), wheel_mat)
add_box("rear_wing_strut_right", (-1.15, -0.3, 0.375), (0.06, 0.06, 0.35), wheel_mat)
add_box("rear_wing", (-1.15, 0.0, 0.59), (0.15, 1.2, 0.08), wheel_mat)
add_box("rear_wing_endplate_left", (-1.15, 0.6, 0.59), (0.22, 0.03, 0.22), wheel_mat)
add_box("rear_wing_endplate_right", (-1.15, -0.6, 0.59), (0.22, 0.03, 0.22), wheel_mat)

stage.Save()

with open(f"{ROOT}/run_output/add_car_body_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
