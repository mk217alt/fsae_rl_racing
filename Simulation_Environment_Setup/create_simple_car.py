"""Builds a simple 4-wheeled car with front-wheel steering as a USD stage in Isaac Sim.

Layout (car forward = +X, up = +Z):
    - Chassis: a box rigid body
    - Rear wheels (RL, RR): attached directly to the chassis via a revolute
      joint about the wheel axis (Y) with a velocity drive -> these are the
      drive wheels.
    - Front wheels (FL, FR): each attached to a small steering "knuckle" via
      a revolute joint about the wheel axis (Y, velocity drive), and the
      knuckle is attached to the chassis via a revolute joint about the
      vertical axis (Z, position drive) -> this is the steering joint.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage

# ---------------------------------------------------------------------------
# scene setup
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = get_current_stage()

CAR = "/World/SimpleCar"
CHASSIS = f"{CAR}/chassis"

CHASSIS_SIZE = Gf.Vec3f(2.0, 1.0, 0.4)   # length (x), width (y), height (z)
WHEEL_RADIUS = 0.3
WHEEL_HALF_HEIGHT = 0.1
WHEEL_MASS = 5.0
CHASSIS_MASS = 80.0
CHASSIS_Z = WHEEL_RADIUS + CHASSIS_SIZE[2] / 2.0 - 0.05

WHEEL_X_OFFSET = 0.8
WHEEL_Y_OFFSET = 0.6


def make_material(path, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind_material(prim_path, material):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(material)


def add_rigid_box(path, position, size, mass):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(size[0] / 1.0, size[1] / 1.0, size[2] / 1.0))
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    return prim


def add_link_with_box(path, position, size, mass):
    """A rigid body 'link' as a plain (unscaled) Xform, with the actual box
    shape as a child prim carrying only the collision/visual geometry.

    Keeping the rigid-body frame unscaled (rather than putting RigidBodyAPI
    directly on a scaled Cube) is the standard PhysX/Isaac Sim link pattern,
    and lets extra cosmetic geometry be parented under this same link without
    inheriting the box's own non-uniform scale.
    """
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)

    box_path = f"{path}/collision"
    cube = UsdGeom.Cube.Define(stage, box_path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return prim, box_path


def add_rigid_wheel(path, position, mass):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(WHEEL_RADIUS)
    cyl.CreateHeightAttr(WHEEL_HALF_HEIGHT * 2.0)
    cyl.CreateAxisAttr("Y")
    cyl.AddTranslateOp().Set(Gf.Vec3d(*position))
    prim = cyl.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateAngularDampingAttr(0.02)
    return prim


def add_knuckle(path, position, mass=0.5):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    return prim


def add_drive_joint(path, body0, body1, axis, local_pos0, local_pos1, drive_type,
                     target=0.0, stiffness=0.0, damping=1e4, max_force=1e5,
                     limits=None):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    if limits is not None:
        joint.CreateLowerLimitAttr(limits[0])
        joint.CreateUpperLimitAttr(limits[1])

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateMaxForceAttr(max_force)
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    if drive_type == "velocity":
        drive.CreateTargetVelocityAttr(target)
    else:
        drive.CreateTargetPositionAttr(target)
    return joint


# ---------------------------------------------------------------------------
# build the car
# ---------------------------------------------------------------------------
chassis_prim, chassis_box_path = add_link_with_box(CHASSIS, (0, 0, CHASSIS_Z), CHASSIS_SIZE, CHASSIS_MASS)
UsdPhysics.ArticulationRootAPI.Apply(chassis_prim)

body_mat = make_material(f"{CAR}/Looks/body", (0.1, 0.35, 0.85))
wheel_mat = make_material(f"{CAR}/Looks/wheel", (0.05, 0.05, 0.05))
bind_material(chassis_box_path, body_mat)

wheel_z = WHEEL_RADIUS

# --- rear wheels: driven, fixed steering (attached straight to chassis) ---
for name, y in (("rear_left", WHEEL_Y_OFFSET), ("rear_right", -WHEEL_Y_OFFSET)):
    wheel_path = f"{CAR}/{name}"
    pos = (-WHEEL_X_OFFSET, y, wheel_z)
    add_rigid_wheel(wheel_path, pos, WHEEL_MASS)
    bind_material(wheel_path, wheel_mat)
    add_drive_joint(
        f"{CAR}/{name}_drive_joint",
        CHASSIS,
        wheel_path,
        axis="Y",
        local_pos0=(pos[0], pos[1], pos[2] - CHASSIS_Z),
        local_pos1=(0, 0, 0),
        drive_type="velocity",
        target=0.0,
        damping=1e4,
    )

# --- front wheels: steerable via a knuckle, also driven ---
for name, y in (("front_left", WHEEL_Y_OFFSET), ("front_right", -WHEEL_Y_OFFSET)):
    knuckle_path = f"{CAR}/{name}_knuckle"
    wheel_path = f"{CAR}/{name}"
    pos = (WHEEL_X_OFFSET, y, wheel_z)

    add_knuckle(knuckle_path, pos)
    add_drive_joint(
        f"{CAR}/{name}_steer_joint",
        CHASSIS,
        knuckle_path,
        axis="Z",
        local_pos0=(pos[0], pos[1], pos[2] - CHASSIS_Z),
        local_pos1=(0, 0, 0),
        drive_type="position",
        target=0.0,
        stiffness=1e5,
        damping=1e3,
        limits=(-30.0, 30.0),
    )

    add_rigid_wheel(wheel_path, pos, WHEEL_MASS)
    bind_material(wheel_path, wheel_mat)
    add_drive_joint(
        f"{CAR}/{name}_drive_joint",
        knuckle_path,
        wheel_path,
        axis="Y",
        local_pos0=(0, 0, 0),
        local_pos1=(0, 0, 0),
        drive_type="velocity",
        target=0.0,
        damping=1e4,
    )

# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
out_path = "C:/Users/sanja/Desktop/thesis/Simulation_Environment_Setup/simple_car.usd"
stage.Export(out_path)
print(f"[create_simple_car] Saved car to {out_path}")

world.reset()
for _ in range(200):
    world.step(render=True)

simulation_app.close()
