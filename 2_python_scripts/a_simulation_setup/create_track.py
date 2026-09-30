"""Adds a square track with rounded (curved) corners into simple_car.usd,
as static (non-simulated) geometry: a dark road surface plus collidable curb
walls along both edges, so the car can be driven around it.

The track is built by approximating a rounded-square centerline (4 straight
sides + 4 quarter-circle corners) as a closed polyline, then extruding each
polyline segment into a thin road box plus two curb boxes offset to either
side. It is added under /World/Track in the same stage as the car, so
drive_car.py picks it up automatically without any changes.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import math

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

CAR_USD = f"{ROOT}/1_simulation_scenes/simple_car.usd"
TRACK = "/World/Track"

HALF_STRAIGHT = 8.0   # half-length of each straight side, in meters
CORNER_RADIUS = 10.0  # centerline radius of each rounded corner, in meters
TRACK_WIDTH = 9.0     # drivable road width, in meters (fits ~3 cars side by side)
CURB_WIDTH = 0.15
CURB_HEIGHT = 0.15
ROAD_HEIGHT = 0.05
ARC_SEGMENTS = 10     # straight-line segments used to approximate each 90-degree corner

stage = Usd.Stage.Open(CAR_USD)

if stage.GetPrimAtPath(TRACK):
    stage.RemovePrim(Sdf.Path(TRACK))
UsdGeom.Xform.Define(stage, TRACK)


def make_material(path, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind(path, material):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material)


road_mat = make_material(f"{TRACK}/Looks/road", (0.08, 0.08, 0.09))
curb_mat = make_material(f"{TRACK}/Looks/curb", (0.95, 0.95, 0.9))


def add_static_box(path, position, size, angle_deg, material, collision):
    # USD composes xformOps in reverse of the order they're added (the
    # last-added op acts on the geometry first) - so to get the standard
    # "scale, then rotate, then move to position" behavior, translate must
    # be added first and scale last.
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.AddRotateZOp().Set(angle_deg)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    bind(path, material)


def arc_points(center, radius, start_deg, end_deg, n):
    pts = []
    for k in range(n + 1):
        t = math.radians(start_deg + (end_deg - start_deg) * k / n)
        pts.append((center[0] + radius * math.cos(t), center[1] + radius * math.sin(t)))
    return pts


# ---------------------------------------------------------------------------
# build the rounded-square centerline as a closed loop of points
# ---------------------------------------------------------------------------
a, r = HALF_STRAIGHT, CORNER_RADIUS
corners = [
    ((a, a), 90, 0),
    ((a, -a), 0, -90),
    ((-a, -a), -90, -180),
    ((-a, a), -180, -270),
]

points = []
for center, start_deg, end_deg in corners:
    points.extend(arc_points(center, r, start_deg, end_deg, ARC_SEGMENTS))

# shift the whole track so the bottom straight sits at y = 0, where the car
# spawns, instead of the track's center sitting at the origin
shift = (0.0, a + r)
points = [(x + shift[0], y + shift[1]) for x, y in points]

# ---------------------------------------------------------------------------
# extrude each polyline segment into a road box + two curb boxes
# ---------------------------------------------------------------------------
n_points = len(points)
half_track = TRACK_WIDTH / 2.0
curb_offset = half_track + CURB_WIDTH / 2.0

for i in range(n_points):
    x0, y0 = points[i]
    x1, y1 = points[(i + 1) % n_points]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        continue
    angle = math.degrees(math.atan2(dy, dx))
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    nx, ny = -dy / length, dx / length  # unit normal, perpendicular to the segment

    add_static_box(
        f"{TRACK}/road_{i}", (mx, my, ROAD_HEIGHT / 2.0),
        (length, TRACK_WIDTH, ROAD_HEIGHT), angle, road_mat, collision=False,
    )
    add_static_box(
        f"{TRACK}/curb_outer_{i}",
        (mx + nx * curb_offset, my + ny * curb_offset, CURB_HEIGHT / 2.0),
        (length + CURB_WIDTH, CURB_WIDTH, CURB_HEIGHT), angle, curb_mat, collision=True,
    )
    add_static_box(
        f"{TRACK}/curb_inner_{i}",
        (mx - nx * curb_offset, my - ny * curb_offset, CURB_HEIGHT / 2.0),
        (length + CURB_WIDTH, CURB_WIDTH, CURB_HEIGHT), angle, curb_mat, collision=True,
    )

stage.Save()

with open(f"{ROOT}/run_output/create_track_done.txt", "w") as f:
    f.write("done\n")

simulation_app.close()
