"""Builds a single-pair deployment stage for watching the self-play racing
policy: exactly one track, two cars (CarA red, CarB blue), matching
car_race_vec_env.py's construction and centerline convention exactly (same
clockwise direction) so the trained policy sees what it trained on.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import math

from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

OUT_PATH = f"{ROOT}/1_simulation_scenes/race_deploy.usd"

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

START_LATERAL_SPACING = 1.5

stage = Usd.Stage.CreateNew(OUT_PATH)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world_prim = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world_prim.GetPrim())

UsdPhysics.Scene.Define(stage, "/physicsScene")

# The original working single-car stage builds its ground via Isaac Sim's
# own World.scene.add_default_ground_plane() helper. Two things it does
# differently from this stage's original scaled-Cube ground, both now
# matched: (1) an explicit PhysicsMaterial (static_friction=0.5,
# dynamic_friction=0.5, restitution=0.8 - isaacsim/core/api/scenes/scene.py)
# - tested alone first, did NOT fix the chassis-collapse bug (diag_race_
# solver_fix.py stress test, identical collapse at z=0.200); (2) an
# ANALYTIC collision plane (PhysicsSchemaTools.addGroundPlane, what
# isaacsim/core/api/objects/ground_plane.py actually calls) rather than a
# finite box - untested until now. Box-vs-cylinder (wheel) contact
# generation can behave differently than plane-vs-cylinder in PhysX.
PhysicsSchemaTools.addGroundPlane(
    stage, "/World/GroundPlane", "Z", 500.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5)
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
    # Default PhysX solver iteration counts (4 position / 1 velocity) are
    # too few to keep the wheel joints stable against the drive joints'
    # high max_force (1e5) under sustained/aggressive throttle - under-
    # resolved constraints let the wheel-chassis joint numerically blow up,
    # which is what was collapsing the chassis to rest on its belly.
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

    # The front wheels only connect to the chassis TRANSITIVELY through the
    # knuckle (chassis -> knuckle -> wheel), not by a direct joint, so
    # PhysX's automatic same-articulation collision filtering (which
    # exempts directly-jointed body pairs) may not cover chassis-vs-front-
    # wheel contact. Geometrically the wheel's inner face sits right at the
    # chassis edge (y=0.5 to 0.7 vs chassis edge at y=0.5) - close enough
    # that any wobble under load could put them into real contact, and an
    # unwanted self-collision impulse there is a plausible explanation for
    # the chassis repeatably collapsing under sustained throttle. Filter
    # chassis-vs-every-wheel explicitly to rule this out.
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

    a, r = HALF_STRAIGHT, CORNER_RADIUS
    corners = [((a, a), 90, 0), ((a, -a), 0, -90), ((-a, -a), -90, -180), ((-a, a), -180, -270)]
    points = []
    for center, start_deg, end_deg in corners:
        points.extend(arc_points(center, r, start_deg, end_deg, ARC_SEGMENTS))
    shift = (ox, oy + a + r)
    points = [(x + shift[0], y + shift[1]) for x, y in points]
    points = list(reversed(points))  # clockwise, matching car_race_vec_env

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
    # Top flush at z=0 (matching the ground plane the wheels rest on),
    # instead of floating ROAD_HEIGHT above it and visually burying the
    # bottom of every wheel.
    add_static_box(f"{track_path}/road", (center_x, center_y, -ROAD_HEIGHT / 2.0),
                    (span_x, span_y, ROAD_HEIGHT), 0.0, road_mat, collision=False)

    # Start/finish line: a checkered stripe at the track's own start point
    # (points[0]) - the same point cars reset to when progress wraps - a
    # visual lap marker, raised slightly above the road to avoid z-fighting.
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx0, dy0 = x1 - x0, y1 - y0
    seg_len0 = math.hypot(dx0, dy0)
    fx0, fy0 = dx0 / seg_len0, dy0 / seg_len0
    nx0, ny0 = -fy0, fx0
    start_angle = math.degrees(math.atan2(dy0, dx0))
    n_tiles = 8
    tile_width = TRACK_WIDTH / n_tiles
    tile_length = 0.5
    white_mat = make_material(f"{track_path}/Looks/start_white", (0.95, 0.95, 0.95), 0.5)
    black_mat = make_material(f"{track_path}/Looks/start_black", (0.05, 0.05, 0.05), 0.5)
    for tile_i in range(n_tiles):
        offset_across = -half_track + tile_width * (tile_i + 0.5)
        tile_x = x0 + nx0 * offset_across
        tile_y = y0 + ny0 * offset_across
        mat = white_mat if tile_i % 2 == 0 else black_mat
        add_static_box(f"{track_path}/start_finish_{tile_i}", (tile_x, tile_y, 0.005),
                        (tile_length, tile_width, 0.01), start_angle, mat, collision=False)

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

    # Joint fillers: on a polyline offset outward (like these curbs, offset
    # by curb_offset=4.5m), adjacent segment boxes pivot at each vertex and
    # pull apart by 2*curb_offset*tan(turn_angle/2) - up to ~0.71m here at
    # this track's ~9 deg/segment corners. First attempt used fixed 1.2m
    # un-rotated cubes at every vertex - closed the physical gap but looked
    # like a chain of fat pillow blocks (much wider than the curb itself).
    # Fixed: a thin box (same CURB_WIDTH as the curbs) placed and rotated to
    # span exactly between the two real gap endpoints, only where a gap
    # actually exists - reads as a continuous line again.
    for i in range(n_points):
        xm1, ym1 = points[(i - 1) % n_points]
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n_points]
        dx0, dy0 = x0 - xm1, y0 - ym1
        len0 = math.hypot(dx0, dy0)
        dx1, dy1 = x1 - x0, y1 - y0
        len1 = math.hypot(dx1, dy1)
        if len0 < 1e-6 or len1 < 1e-6:
            continue
        n0x, n0y = -dy0 / len0, dx0 / len0
        n1x, n1y = -dy1 / len1, dx1 / len1

        for side_sign, tag in ((1.0, "outer"), (-1.0, "inner")):
            gx0, gy0 = x0 + side_sign * n0x * curb_offset, y0 + side_sign * n0y * curb_offset
            gx1, gy1 = x0 + side_sign * n1x * curb_offset, y0 + side_sign * n1y * curb_offset
            gap_len = math.hypot(gx1 - gx0, gy1 - gy0)
            if gap_len < 0.05:
                continue  # essentially collinear, the two curb boxes already meet here
            gap_angle = math.degrees(math.atan2(gy1 - gy0, gx1 - gx0))
            gmx, gmy = (gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0
            add_static_box(f"{track_path}/curb_fill_{tag}_{i}",
                            (gmx, gmy, CURB_HEIGHT / 2.0),
                            (gap_len + CURB_WIDTH, CURB_WIDTH, CURB_HEIGHT), gap_angle, curb_mat, collision=True)


build_track("/World/Track", (0.0, 0.0))
# Offsets are raw world-frame (x, y), not track-relative - at this spawn
# point the track's tangent runs along +X (confirmed empirically), so the
# START_LATERAL_SPACING has to go on the Y axis to actually separate the
# cars sideways. The original (-1.5, 0.0)/(1.5, 0.0) put that spacing on
# the along-track axis instead, spawning CarB ~3m further down the track
# than CarA rather than beside it - looked like a staggered line-up, not
# a side-by-side start.
build_car("/World/CarA", (0.0, -1.5), (0.85, 0.2, 0.15))  # red
build_car("/World/CarB", (0.0, 1.5), (0.1, 0.35, 0.85))   # blue

stage.Save()

with open(f"{ROOT}/run_output/build_race_deploy_stage_done.txt", "w") as f:
    f.write(f"built race deploy stage at {OUT_PATH}\n")

simulation_app.close()
