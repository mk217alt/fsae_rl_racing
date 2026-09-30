"""Builds a small deployment stage for the vec-trained curvature-aware
car2 policy: ONE track, centered with no lane offset (matching exactly
what car_track_vec_env.py trained against), with car2 (RL) spawned on
that true centerline and car1 (human, keyboard) placed a few meters to
the side just so they don't start stacked on top of each other.

This replaces the earlier (buggy) approach of reusing simple_car.usd's
shared-lane track, which has a +3m lane-offset baked into its centerline
specifically for the OLD single-env policy - a convention the new
curvature-aware policy was never trained against.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import math

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

OUT_PATH = f"{ROOT}/1_simulation_scenes/vec_deploy_car2.usd"

CHASSIS_SIZE = Gf.Vec3f(2.0, 1.0, 0.4)
WHEEL_RADIUS = 0.3
WHEEL_HALF_HEIGHT = 0.1
WHEEL_MASS = 5.0
CHASSIS_MASS = 80.0
CHASSIS_Z = WHEEL_RADIUS + CHASSIS_SIZE[2] / 2.0 - 0.05
WHEEL_X_OFFSET = 0.8
WHEEL_Y_OFFSET = 0.6

HALF_STRAIGHT = 8.0
CORNER_RADIUS = 10.0
TRACK_WIDTH = 9.0
CURB_WIDTH = 0.15
CURB_HEIGHT = 0.15
ROAD_HEIGHT = 0.05
ARC_SEGMENTS = 10

CAR1_OFFSET = (-6.0, 0.0)  # generous clearance from car2 to reduce collision risk
CAR2_OFFSET = (0.0, 0.0)   # true center, matching training exactly
TRACK_OFFSET = (0.0, 0.0)

stage = Usd.Stage.CreateNew(OUT_PATH)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world_prim = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world_prim.GetPrim())

UsdPhysics.Scene.Define(stage, "/physicsScene")

ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
ground.CreateSizeAttr(1.0)
ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
ground.AddScaleOp().Set(Gf.Vec3f(500.0, 500.0, 1.0))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())


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

    a, r = HALF_STRAIGHT, CORNER_RADIUS
    corners = [((a, a), 90, 0), ((a, -a), 0, -90), ((-a, -a), -90, -180), ((-a, a), -180, -270)]
    points = []
    for center, start_deg, end_deg in corners:
        points.extend(arc_points(center, r, start_deg, end_deg, ARC_SEGMENTS))
    shift = (ox, oy + a + r)
    points = [(x + shift[0], y + shift[1]) for x, y in points]
    points = list(reversed(points))  # clockwise, matching car_track_vec_env exactly

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

    # One single flat, seamless road surface spanning the whole track's
    # bounding box, instead of many small per-segment boxes - avoids any
    # visible seam/step at segment joints. No collision either way (cars
    # physically drive on the flat ground plane beneath), but a single
    # piece also removes any z-fighting between adjacent segment edges.
    # Positioned so its TOP sits flush at z=0 (matching the ground plane
    # the wheels actually rest on) instead of floating above it - the
    # earlier version had the road's top at z=ROAD_HEIGHT, which visually
    # buried the bottom of every wheel by that amount.
    margin = half_track + CURB_WIDTH
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span_x = (max(xs) - min(xs)) + 2 * margin
    span_y = (max(ys) - min(ys)) + 2 * margin
    center_x = (max(xs) + min(xs)) / 2.0
    center_y = (max(ys) + min(ys)) / 2.0
    add_static_box(f"{track_path}/road", (center_x, center_y, -ROAD_HEIGHT / 2.0),
                    (span_x, span_y, ROAD_HEIGHT), 0.0, road_mat, collision=False)

    # Start/finish line: a checkered stripe painted across the road at the
    # track's own start point (points[0], the same point car2 gets reset to
    # when progress wraps around) - purely a visual lap marker, raised
    # slightly above the road surface so it doesn't z-fight with it.
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx, dy = x1 - x0, y1 - y0
    seg_len = math.hypot(dx, dy)
    fx, fy = dx / seg_len, dy / seg_len  # forward (along track) unit vector
    nx, ny = -fy, fx  # perpendicular (across track) unit vector
    angle = math.degrees(math.atan2(dy, dx))
    n_tiles = 8
    tile_width = TRACK_WIDTH / n_tiles
    tile_length = 0.5
    white_mat = make_material(f"{track_path}/Looks/start_white", (0.95, 0.95, 0.95), 0.5)
    black_mat = make_material(f"{track_path}/Looks/start_black", (0.05, 0.05, 0.05), 0.5)
    for tile_i in range(n_tiles):
        offset_across = -half_track + tile_width * (tile_i + 0.5)
        tile_x = x0 + nx * offset_across
        tile_y = y0 + ny * offset_across
        mat = white_mat if tile_i % 2 == 0 else black_mat
        add_static_box(f"{track_path}/start_finish_{tile_i}", (tile_x, tile_y, 0.005),
                        (tile_length, tile_width, 0.01), angle, mat, collision=False)

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


build_track("/World/Track", TRACK_OFFSET)
build_car("/World/Car1", CAR1_OFFSET, (0.1, 0.35, 0.85))   # blue, human
build_car("/World/Car2", CAR2_OFFSET, (0.85, 0.2, 0.15))   # red, RL policy

stage.Save()

with open(f"{ROOT}/run_output/build_vec_deploy_stage_done.txt", "w") as f:
    f.write(f"built deploy stage at {OUT_PATH}\n")

simulation_app.close()
