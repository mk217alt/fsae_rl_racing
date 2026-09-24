"""Builds a MULTI-PAIR training stage on the unknown-track geometry (same
layout as build_unknown_track_deploy.py: 40m long straight, 8m short
straight, 6m corner radius - vs the original lineage's trained-on 16m/16m/
10m symmetric square), for continued training rather than just evaluation.

Same "N racing-pair clones" pattern as build_race_track_env.py, so training
throughput matches the established lineage (8 pairs = 16 parallel car
slots). Grid spacing widened to 70m (from the original's 60m) since this
track's footprint is bigger and non-square (52m x 20m vs the original's
36m x 36m) - 60m would leave the long straights of adjacent columns too
close together.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import math

from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

OUT_PATH = f"{ROOT}/1_simulation_scenes/unknown_track_train.usd"

NUM_PAIRS = 8
GRID_COLS = 3
GRID_SPACING = 70.0

CHASSIS_SIZE = Gf.Vec3f(2.0, 1.0, 0.4)
WHEEL_RADIUS = 0.3
WHEEL_HALF_HEIGHT = 0.1
WHEEL_MASS = 5.0
CHASSIS_MASS = 80.0
CHASSIS_Z = WHEEL_RADIUS + CHASSIS_SIZE[2] / 2.0 - 0.05
WHEEL_X_OFFSET = 0.8
WHEEL_Y_OFFSET = 0.6

HALF_LENGTH_X = 20.0
HALF_LENGTH_Y = 4.0
CORNER_RADIUS = 6.0
TRACK_WIDTH = 9.0
CURB_WIDTH = 0.15
CURB_HEIGHT = 0.15
ROAD_HEIGHT = 0.05
ARC_SEGMENTS = 10

stage = Usd.Stage.CreateNew(OUT_PATH)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world_prim = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world_prim.GetPrim())

UsdPhysics.Scene.Define(stage, "/physicsScene")

PhysicsSchemaTools.addGroundPlane(
    stage, "/World/GroundPlane", "Z", 2000.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5)
)
ground_geom_prim = stage.GetPrimAtPath("/World/GroundPlane/geom")
UsdGeom.Imageable(ground_geom_prim).MakeInvisible()

ground_material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/ground_material")
ground_material_api = UsdPhysics.MaterialAPI.Apply(ground_material.GetPrim())
ground_material_api.CreateStaticFrictionAttr(0.5)
ground_material_api.CreateDynamicFrictionAttr(0.5)
ground_material_api.CreateRestitutionAttr(0.8)
UsdShade.MaterialBindingAPI.Apply(ground_geom_prim).Bind(ground_material, materialPurpose="physics")


def make_material(path, color, roughness):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind(path, material):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material)


def add_link_with_box(path, position, size, mass):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
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
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateAngularDampingAttr(0.02)
    return prim


def add_knuckle(path, position, mass=0.5):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    return prim


def add_joint(path, body0, body1, axis, local_pos0, local_pos1, drive=None, limits=None):
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
    if drive is not None:
        drive_type, target, stiffness, damping, max_force = drive
        d = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        d.CreateTypeAttr("force")
        d.CreateMaxForceAttr(max_force)
        d.CreateStiffnessAttr(stiffness)
        d.CreateDampingAttr(damping)
        if drive_type == "velocity":
            d.CreateTargetVelocityAttr(target)
        else:
            d.CreateTargetPositionAttr(target)
    return joint


def build_car(car_path, offset, color):
    ox, oy = offset
    chassis_path = f"{car_path}/chassis"
    chassis_prim, chassis_box_path = add_link_with_box(
        chassis_path, (ox, oy, CHASSIS_Z), CHASSIS_SIZE, CHASSIS_MASS
    )
    UsdPhysics.ArticulationRootAPI.Apply(chassis_prim)
    articulation_physx = PhysxSchema.PhysxArticulationAPI.Apply(chassis_prim)
    articulation_physx.CreateSolverPositionIterationCountAttr(32)
    articulation_physx.CreateSolverVelocityIterationCountAttr(8)

    body_mat = make_material(f"{car_path}/Looks/body", color, 0.6)
    wheel_mat = make_material(f"{car_path}/Looks/wheel", (0.05, 0.05, 0.05), 0.6)
    bind(chassis_box_path, body_mat)

    wheel_z = WHEEL_RADIUS

    for name, y in (("rear_left", WHEEL_Y_OFFSET), ("rear_right", -WHEEL_Y_OFFSET)):
        wheel_path = f"{car_path}/{name}"
        pos = (ox - WHEEL_X_OFFSET, oy + y, wheel_z)
        add_rigid_wheel(wheel_path, pos, WHEEL_MASS)
        bind(wheel_path, wheel_mat)
        add_joint(
            f"{car_path}/{name}_drive_joint", chassis_path, wheel_path, axis="Y",
            local_pos0=(pos[0] - ox, pos[1] - oy, pos[2] - CHASSIS_Z), local_pos1=(0, 0, 0),
        )

    for name, y in (("front_left", WHEEL_Y_OFFSET), ("front_right", -WHEEL_Y_OFFSET)):
        knuckle_path = f"{car_path}/{name}_knuckle"
        wheel_path = f"{car_path}/{name}"
        pos = (ox + WHEEL_X_OFFSET, oy + y, wheel_z)

        add_knuckle(knuckle_path, pos)
        add_joint(
            f"{car_path}/{name}_steer_joint", chassis_path, knuckle_path, axis="Z",
            local_pos0=(pos[0] - ox, pos[1] - oy, pos[2] - CHASSIS_Z), local_pos1=(0, 0, 0),
            drive=("position", 0.0, 1e5, 1e3, 1e5), limits=(-30.0, 30.0),
        )
        add_rigid_wheel(wheel_path, pos, WHEEL_MASS)
        bind(wheel_path, wheel_mat)
        add_joint(
            f"{car_path}/{name}_drive_joint", knuckle_path, wheel_path, axis="Y",
            local_pos0=(0, 0, 0), local_pos1=(0, 0, 0),
            drive=("velocity", 0.0, 0.0, 1e4, 1e5),
        )

    filter_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(chassis_box_path))
    wheel_collision_paths = [
        f"{car_path}/rear_left", f"{car_path}/rear_right",
        f"{car_path}/front_left", f"{car_path}/front_right",
        f"{car_path}/front_left_knuckle", f"{car_path}/front_right_knuckle",
    ]
    filter_api.CreateFilteredPairsRel().SetTargets([Sdf.Path(p) for p in wheel_collision_paths])


def arc_points(center, radius, start_deg, end_deg, n):
    pts = []
    for k in range(n + 1):
        t = math.radians(start_deg + (end_deg - start_deg) * k / n)
        pts.append((center[0] + radius * math.cos(t), center[1] + radius * math.sin(t)))
    return pts


def build_track(track_path, offset):
    ox, oy = offset
    UsdGeom.Xform.Define(stage, track_path)
    road_mat = make_material(f"{track_path}/Looks/road", (0.08, 0.08, 0.09), 0.85)
    curb_mat = make_material(f"{track_path}/Looks/curb", (0.95, 0.95, 0.9), 0.85)

    ax, ay, r = HALF_LENGTH_X, HALF_LENGTH_Y, CORNER_RADIUS
    corners = [((ax, ay), 90, 0), ((ax, -ay), 0, -90), ((-ax, -ay), -90, -180), ((-ax, ay), -180, -270)]
    points = []
    for center, start_deg, end_deg in corners:
        points.extend(arc_points(center, r, start_deg, end_deg, ARC_SEGMENTS))
    shift = (ox, oy + ay + r)
    points = [(x + shift[0], y + shift[1]) for x, y in points]
    points = list(reversed(points))  # clockwise, matching car_unknown_track_env's convention

    n_points = len(points)
    half_track = TRACK_WIDTH / 2.0
    curb_offset = half_track + CURB_WIDTH / 2.0

    def add_static_box(path, position, size, angle, material, collision):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddRotateZOp().Set(angle)
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        bind(path, material)

    margin = half_track + CURB_WIDTH
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span_x = (max(xs) - min(xs)) + 2 * margin
    span_y = (max(ys) - min(ys)) + 2 * margin
    center_x = (max(xs) + min(xs)) / 2.0
    center_y = (max(ys) + min(ys)) / 2.0
    add_static_box(f"{track_path}/road", (center_x, center_y, ROAD_HEIGHT / 2.0),
                    (span_x, span_y, ROAD_HEIGHT), 0.0, road_mat, collision=False)

    for i in range(n_points):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n_points]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        nx, ny = -dy / length, dx / length

        add_static_box(f"{track_path}/curb_outer_{i}",
                        (mx + nx * curb_offset, my + ny * curb_offset, CURB_HEIGHT / 2.0),
                        (length + CURB_WIDTH, CURB_WIDTH, CURB_HEIGHT), angle, curb_mat, collision=True)
        add_static_box(f"{track_path}/curb_inner_{i}",
                        (mx - nx * curb_offset, my - ny * curb_offset, CURB_HEIGHT / 2.0),
                        (length + CURB_WIDTH, CURB_WIDTH, CURB_HEIGHT), angle, curb_mat, collision=True)


clone_offsets = []
for i in range(NUM_PAIRS):
    col, row = i % GRID_COLS, i // GRID_COLS
    clone_offsets.append((col * GRID_SPACING, row * GRID_SPACING))

for i, offset in enumerate(clone_offsets):
    env_root = f"/World/Pair{i}"
    build_track(f"{env_root}/Track", offset)
    build_car(f"{env_root}/CarA", (offset[0] - 1.5, offset[1]), (0.85, 0.2, 0.15))  # red
    build_car(f"{env_root}/CarB", (offset[0] + 1.5, offset[1]), (0.1, 0.35, 0.85))  # blue

stage.Save()

with open(f"{ROOT}/run_output/build_unknown_track_train_done.txt", "w") as f:
    f.write(f"built {NUM_PAIRS} racing pairs at offsets {clone_offsets}, saved to {OUT_PATH}\n")

simulation_app.close()
