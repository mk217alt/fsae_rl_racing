"""Graphs of the lap telemetry written by lap_telemetry.py (4_results/lap_telemetry_100M*.csv/.json), comparing the
two cars in the style of Formula 1 telemetry:

    <run>_traces.png   speed, throttle, longitudinal acceleration and steering angle along the lap, both cars, every
                       flying lap drawn thin and the average of the flying laps thick, sectors shaded
    <run>_laps.png     longitudinal acceleration and steering angle of both cars lap by lap (lap 1 to 10), and the
                       number of steering reversals per lap
    <run>_track.png    the track to scale with the timing line, the three sectors, the corners and kerbs, the grid,
                       and a timing strip with the mean and best sector and lap times of both cars
    <run>_sectors.png  sector times of both cars lap by lap (one panel per sector, mean and best sector marked)
                       and the interval between the cars at the timing line

Usage: python.bat plot_lap_telemetry.py [run ...]      (run = lap_telemetry_100M, lap_telemetry_100M_swapped, ...;
                                                        default: every lap_telemetry_*.json in 4_results)
The thesis figures (thesis_document/build/figures.py) call the same functions with the thesis style.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))).replace("\\", "/")  # the project folder, wherever it is

import csv
import glob
import json
import math
import sys

try:
    import matplotlib
except ImportError:  # Isaac Sim's Python keeps matplotlib and Pillow in extension folders
    _isaac = _os.path.dirname(_os.path.dirname(_os.path.dirname(sys.executable)))
    for _p in glob.glob(_os.path.join(_isaac, "exts*", "*", "pip_prebundle")):
        sys.path.append(_p)
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon

RESULTS = f"{ROOT}/4_results"
CM = 1 / 2.54
LEAD_COLOR = "#2b5d8a"
FOLLOW_COLOR = "#c0622b"
INK = "#1b2430"
SECTOR_SHADE = ["#f1f3f5", "#ffffff", "#f1f3f5"]
SECTOR_LINE = ["#2f8f83", "#6c8f3d", "#b0852b"]  # sector 1, 2, 3 (distinct from the two car colours)


# ------------------------------------------------------------------ data
def load(run):
    """rows per car as numpy columns, the JSON report, the leader and the follower, and the track geometry"""
    report = json.load(open(f"{RESULTS}/{run}.json"))
    cols = {}
    with open(f"{RESULTS}/{run}.csv") as f:
        rd = csv.DictReader(f)
        raw = list(rd)
    for car in ("Car A", "Car B"):
        r = [x for x in raw if x["car"] == car]
        cols[car] = {k: np.array([float(x[k]) for x in r]) for k in rd.fieldnames if k != "car"}
    flying = {c: (cols[c]["lap"] >= 2) & (cols[c]["lap"] <= report["n_laps"]) for c in cols}
    leader = max(cols, key=lambda c: np.median(cols[c]["gap_m"][flying[c]]))
    follower = "Car A" if leader == "Car B" else "Car B"
    return {"report": report, "cols": cols, "flying": flying, "leader": leader, "follower": follower,
            "geometry": geometry(report, cols)}


def geometry(report, cols):
    """centerline in lap-distance coordinates (0 = timing line) and the corner spans"""
    pts = np.array(report["centerline_xy"])
    cum = np.array(report["centerline_cumulative_m"])
    length = report["track_length_m"]
    # timing line = Car B's grid spot (the first logged position of Car B), projected onto the centerline
    bx, by = cols["Car B"]["x_m"][0], cols["Car B"]["y_m"][0]
    best = (1e9, 0.0)
    n = len(pts)
    for i in range(n):
        p0, p1 = pts[i], pts[(i + 1) % n]
        d = p1 - p0
        t = float(np.clip(((bx - p0[0]) * d[0] + (by - p0[1]) * d[1]) / (d @ d), 0, 1))
        q = p0 + t * d
        dist2 = (bx - q[0]) ** 2 + (by - q[1]) ** 2
        if dist2 < best[0]:
            best = (dist2, cum[i] + t * (cum[i + 1] - cum[i]))
    p_line = best[1]
    seg_len = np.diff(cum)
    corners = []
    i = 0
    while i < n:
        if seg_len[i] < 3.0:  # arcs are split into short chords; straights are single long segments
            j = i
            while j + 1 < n and seg_len[j + 1] < 3.0:
                j += 1
            corners.append([(cum[i] - p_line) % length, (cum[j + 1] - p_line) % length])
            i = j + 1
        else:
            i += 1
    corners.sort()

    def xy_at(s):
        p = (p_line + s) % length
        k = int(np.searchsorted(cum, p, side="right") - 1)
        k = min(k, n - 1)
        t = (p - cum[k]) / (cum[k + 1] - cum[k])
        return pts[k] + t * (pts[(k + 1) % n] - pts[k])

    return {"length": length, "sector": report["sector_length_m"], "corners": corners, "xy_at": xy_at,
            "points": pts}


CAR_WORD = "Car"  # the thesis figures say "Vehicle"


def role_label(d, car):
    grid = d["report"]["grid"][car]
    role = "leader" if car == d["leader"] else "follower"
    return f"{CAR_WORD} {car[-1]} ({role}; grid: {grid.replace(' of the first corner', '')})"


def car_color(d, car):
    return LEAD_COLOR if car == d["leader"] else FOLLOW_COLOR


def shade_sectors(ax, d, labels=False):
    g = d["geometry"]
    for s in range(3):
        ax.axvspan(s * g["sector"], (s + 1) * g["sector"], color=SECTOR_SHADE[s], zorder=0, lw=0)
        if labels:
            ax.text((s + 0.5) * g["sector"], 1.02, f"Sector {s + 1}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=INK)
    for s in range(1, 3):
        ax.axvline(s * g["sector"], color="#8a939c", lw=0.6, ls="--", zorder=1)
    for c0, c1 in g["corners"]:
        ax.axvspan(c0, c1, ymin=0, ymax=0.035, color="#8a939c", lw=0, zorder=2)


def lap_matrix(d, car, key, grid_s):
    """one row per flying lap, the quantity resampled on a common lap-distance grid"""
    c = d["cols"][car]
    out = []
    for lap in range(2, d["report"]["n_laps"] + 1):
        m = c["lap"] == lap
        s, v = c["lap_dist_m"][m], c[key][m]
        if len(s) > 10:
            out.append(np.interp(grid_s, s, v))
    return np.array(out)


def steering_reversals(values):
    signs = np.sign(values)
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1]))


# ------------------------------------------------------------------ figures
def fig_traces(d, out, width_cm=16.0):
    """speed, throttle, acceleration and steering along the lap, both cars, flying laps overlaid + their average"""
    g = d["geometry"]
    grid_s = np.linspace(0, g["length"], 1070)
    panels = [("speed_mps", "Speed (m/s)", 1.0), ("throttle_cmd", "Throttle (%)", 100.0),
              ("accel_long_mps2", "Long. acceleration\n(m/s$^2$)", 1.0), ("steer_deg", "Steering angle (deg)", 1.0)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(width_cm * CM, 15.5 * CM), sharex=True,
                             gridspec_kw={"height_ratios": [1, 0.8, 1, 1.25]})
    for ax, (key, label, scale) in zip(axes, panels):
        shade_sectors(ax, d, labels=ax is axes[0])
        for car in (d["follower"], d["leader"]):
            mat = lap_matrix(d, car, key, grid_s) * scale
            for row in mat:
                ax.plot(grid_s, row, color=car_color(d, car), lw=0.35, alpha=0.28, zorder=3)
            ax.plot(grid_s, mat.mean(axis=0), color=car_color(d, car), lw=1.3, zorder=4,
                    label=role_label(d, car))
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#dde1e5", lw=0.5)
    axes[0].set_ylim(4.9, 6.25)
    axes[1].set_ylim(-5, 108)
    axes[3].set_ylim(-28, 28)
    axes[3].set_yticks([-25, 0, 25])
    axes[-1].set_xlabel("Distance from the timing line (m)")
    axes[-1].set_xlim(0, g["length"])
    for k, (c0, c1) in enumerate(g["corners"]):
        axes[-1].text((c0 + c1) / 2, -27, f"C{k + 1}", ha="center", va="bottom", fontsize=7, color="#5b646d")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.4, rect=(0, 0.035, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


ACCEL_LIM = 3.6


def fig_laps(d, out, width_cm=16.0):
    """acceleration and steering angle of both cars, lap by lap"""
    n = d["report"]["n_laps"]
    laps = np.arange(1, n + 1)
    fig, axes = plt.subplots(3, 1, figsize=(width_cm * CM, 11.6 * CM), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 0.9, 0.9]})
    offsets = {d["leader"]: 0.19, d["follower"]: -0.19}
    ax = axes[0]
    for car in (d["follower"], d["leader"]):
        c = d["cols"][car]
        data = [c["accel_long_mps2"][c["lap"] == lap] for lap in laps]
        color = car_color(d, car)
        bp = ax.boxplot(data, positions=laps + offsets[car], widths=0.32, whis=(5, 95), showfliers=False,
                        patch_artist=True, manage_ticks=False)
        for b in bp["boxes"]:
            b.set(facecolor=color, alpha=0.35, edgecolor=color, lw=0.7)
        for b in bp["whiskers"] + bp["caps"]:
            b.set(color=color, lw=0.7)
        for b in bp["medians"]:
            b.set(color=color, lw=1.2)
        for fn, marker, edge in ((np.min, "v", -ACCEL_LIM), (np.max, "^", ACCEL_LIM)):
            vals = np.array([fn(x) for x in data])
            ax.scatter(laps + offsets[car], np.clip(vals, -ACCEL_LIM + 0.15, ACCEL_LIM - 0.15), marker=marker, s=10,
                       color=color, zorder=5)
            for lap, v in zip(laps, vals):
                if abs(v) > ACCEL_LIM - 0.15:  # off the scale (lap 1): print the value under the marker
                    ax.text(lap + offsets[car], np.sign(v) * (ACCEL_LIM - 0.6), f"{v:+.1f}", fontsize=6.3,
                            color=color, ha="center", va="center")
    ax.set_ylim(-ACCEL_LIM, ACCEL_LIM)
    ax.set_ylabel("Long. acceleration\n(m/s$^2$)")
    ax.text(1, 1.02, "lap 1: standing start", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7, color="#5b646d")
    for car in (d["follower"], d["leader"]):
        c = d["cols"][car]
        mean_abs = [np.mean(np.abs(c["steer_deg"][c["lap"] == lap])) for lap in laps]
        rev = [steering_reversals(c["steer_deg"][c["lap"] == lap]) for lap in laps]
        axes[1].bar(laps + offsets[car], mean_abs, width=0.34, color=car_color(d, car), alpha=0.85)
        axes[2].bar(laps + offsets[car], rev, width=0.34, color=car_color(d, car), alpha=0.85,
                    label=role_label(d, car))
    axes[1].set_ylabel("Mean steering\nangle (|deg|)")
    axes[2].set_ylabel("Steering reversals\nper lap")
    for ax in axes:
        ax.grid(axis="y", color="#dde1e5", lw=0.5)
        ax.axvspan(0.5, 1.5, color="#f1f3f5", zorder=0, lw=0)
    axes[2].set_xticks(laps)
    axes[2].set_xlim(0.5, n + 0.5)
    axes[2].set_xlabel("Lap")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.5, rect=(0, 0.04, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


TRACK_WIDTH = 9.0
KERB_RED = "#c8102e"


def _frame(g, s):
    """point on the centerline at lap distance s, unit tangent, and unit normal to the left (the inside)"""
    p = np.asarray(g["xy_at"](s), dtype=float)
    q = np.asarray(g["xy_at"](s + 0.25), dtype=float)
    t = (q - p) / np.linalg.norm(q - p)
    return p, t, np.array([-t[1], t[0]])


def sector_names(g):
    names = []
    for s in range(3):
        cs = [f"C{k + 1}" for k, (c0, c1) in enumerate(g["corners"]) if s * g["sector"] <= (c0 + c1) / 2 < (s + 1) * g["sector"]]
        names.append("corners " + " and ".join(cs) if cs else "back straight")
    return names


def fig_track(d, out, width_cm=16.0):
    """the stadium track to scale: sectors, corners with kerbs, timing line and grid; below it a timing strip with
    the mean and best sector and lap times of both cars"""
    g, rep = d["geometry"], d["report"]
    L, sec = g["length"], g["sector"]
    half = TRACK_WIDTH / 2
    fig = plt.figure(figsize=(width_cm * CM, 11.2 * CM))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.0], hspace=0.04)
    ax = fig.add_subplot(gs[0])

    # asphalt, one polygon per sector, with the sector colour along the centerline
    for s in range(3):
        ss = np.linspace(s * sec, (s + 1) * sec, 200)
        fr = [_frame(g, v) for v in ss]
        outer = np.array([p - n * half for p, t, n in fr])
        inner = np.array([p + n * half for p, t, n in fr])
        ax.add_patch(Polygon(np.concatenate([outer, inner[::-1]]), closed=True, facecolor="#d3d8dd",
                             edgecolor="#d3d8dd", lw=0.6, zorder=1))
        center = np.array([p for p, t, n in fr])
        ax.plot(center[:, 0], center[:, 1], color=SECTOR_LINE[s], lw=4.0, solid_capstyle="butt", zorder=3)
        p, t, n = _frame(g, (s + 0.5) * sec)
        ax.text(*(p + n * 2.6), f"S{s + 1}", color=SECTOR_LINE[s], fontsize=9, weight="bold", ha="center",
                va="center", zorder=8)
    # track edges, and red-and-white kerbs on the edge of the asphalt through the corners
    ss = np.linspace(0, L, 900)
    fr = [_frame(g, v) for v in ss]
    for sign in (-1, 1):
        e = np.array([p + sign * n * half for p, t, n in fr])
        ax.plot(np.append(e[:, 0], e[0, 0]), np.append(e[:, 1], e[0, 1]), color="#6b737b", lw=0.8, zorder=2)
    for c0, c1 in g["corners"]:
        edges = np.arange(c0 - 1.0, c1 + 1.0, 1.0)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            for sign in (-1, 1):
                pa, _, na = _frame(g, a)
                pb, _, nb = _frame(g, b)
                seg = np.array([pa + sign * na * (half - 0.3), pb + sign * nb * (half - 0.3)])
                ax.plot(seg[:, 0], seg[:, 1], color=KERB_RED if i % 2 == 0 else "white", lw=3.0,
                        solid_capstyle="butt", zorder=4)
    # sector boundaries across the track
    for s in (1, 2):
        p, t, n = _frame(g, s * sec)
        seg = np.array([p - n * half, p + n * half])
        ax.plot(seg[:, 0], seg[:, 1], color="white", lw=1.8, ls=(0, (2, 1.2)), zorder=5)
    # chequered timing line at the grid
    p, t, n = _frame(g, 0.0)
    q = TRACK_WIDTH / 12
    for i in range(12):
        for r in range(2):
            o = p + n * (-half + i * q) + t * (r - 1) * q
            sq = np.array([o, o + n * q, o + n * q + t * q, o + t * q])
            ax.add_patch(Polygon(sq, closed=True, facecolor=INK if (i + r) % 2 else "white", edgecolor=INK, lw=0.2,
                                 zorder=6))
    ax.text(p[0], p[1] - half - 0.9, "timing line at the grid", ha="center", va="top", fontsize=7.5, color=INK)
    # the two cars on the grid (2.0 m x 1.0 m, to scale)
    for car in (d["follower"], d["leader"]):
        c = d["cols"][car]
        pc = np.array([c["x_m"][0], c["y_m"][0]])
        _, tc, nc = _frame(g, c["lap_dist_m"][0])
        box = np.array([pc - tc - nc * 0.5, pc + tc - nc * 0.5, pc + tc + nc * 0.5, pc - tc + nc * 0.5])
        ax.add_patch(Polygon(box, closed=True, facecolor=car_color(d, car), edgecolor="white", lw=0.6, zorder=7))
        ax.text(*(pc - tc * (2.5 + c["lap_dist_m"][0])), car[-1], ha="center", va="center", fontsize=7.5, color=car_color(d, car),
                weight="bold", zorder=7)
    # direction of travel
    for a, b in ((6.0, 14.0), (62.0, 70.0)):
        pa, _, na = _frame(g, a)
        pb, _, nb = _frame(g, b)
        ax.annotate("", xy=pb - nb * 2.2, xytext=pa - na * 2.2, zorder=8,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1, mutation_scale=10))
    # corner labels, outside the corners
    mid = np.mean([_frame(g, v)[0] for v in np.linspace(0, L, 60, endpoint=False)], axis=0)
    for k, (c0, c1) in enumerate(g["corners"]):
        pc = _frame(g, (c0 + c1) / 2)[0]
        v = (pc - mid) / np.linalg.norm(pc - mid)
        ax.text(*(pc + v * (half + 2.4)), f"C{k + 1}", ha="center", va="center", fontsize=8.5, color="#4b545c",
                weight="bold")
    ax.set_xlim(-33, 33)
    ax.set_ylim(-8.2, 27.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # timing strip: one box per sector and one for the lap, as on a Formula 1 timing screen
    st = fig.add_subplot(gs[1])
    st.set_xlim(0, 4)
    st.set_ylim(0, 1)
    st.axis("off")
    fl = {car: rep["cars"][car]["flying_laps"] for car in ("Car A", "Car B")}
    names = sector_names(g)
    boxes = [(f"SECTOR {s + 1}  ·  {sec:.1f} m", names[s].replace("corners ", ""), SECTOR_LINE[s],
              {car: (fl[car]["sector_mean_s"][s], fl[car]["best_sectors_s"][s]) for car in fl}) for s in range(3)]
    boxes.append((f"LAP  ·  {L:.1f} m", f"laps 2–{rep['n_laps']}", INK,
                  {car: (fl[car]["lap_mean_s"], fl[car]["best_lap_s"]) for car in fl}))
    for i, (title, sub, col, vals) in enumerate(boxes):
        x0, w = i + 0.03, 0.94
        st.add_patch(FancyBboxPatch((x0, 0.04), w, 0.9, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor="white", edgecolor=col, lw=1.1, transform=st.transData))
        st.add_patch(Polygon([(x0, 0.94), (x0 + w, 0.94), (x0 + w, 0.72), (x0, 0.72)], closed=True, facecolor=col,
                             edgecolor="none"))
        st.text(x0 + 0.04, 0.83, title, color="white", fontsize=6.9, weight="bold", va="center")
        st.text(x0 + 0.04, 0.60, sub, color="#4b545c", fontsize=6.4, va="center")
        st.text(x0 + w - 0.24, 0.60, "mean", color="#8a939c", fontsize=6.0, va="center", ha="right")
        st.text(x0 + w - 0.03, 0.60, "best", color="#8a939c", fontsize=6.0, va="center", ha="right")
        for row, car in enumerate((d["leader"], d["follower"])):
            y = 0.40 - 0.22 * row
            role = "leader" if car == d["leader"] else "follower"
            mean, best = vals[car]
            st.text(x0 + 0.04, y, f"{car[-1]}  {role}", color=car_color(d, car), fontsize=6.5, va="center")
            st.text(x0 + w - 0.24, y, f"{mean:.2f}", color=car_color(d, car), fontsize=6.8, va="center", ha="right",
                    weight="bold")
            st.text(x0 + w - 0.03, y, f"{best:.2f}", color=car_color(d, car), fontsize=6.8, va="center", ha="right")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_sectors(d, out, width_cm=16.0):
    """sector times of both cars lap by lap (one panel per sector) and the interval at the timing line"""
    g, rep = d["geometry"], d["report"]
    n = rep["n_laps"]
    laps = np.arange(2, n + 1)
    sectors = {car: np.array([l["sectors_s"] for l in rep["cars"][car]["laps"]]) for car in ("Car A", "Car B")}
    flying = np.concatenate([sectors[c][1:, :].ravel() for c in sectors])
    ylim = (np.floor((flying.min() - 0.06) * 20) / 20, np.ceil((flying.max() + 0.06) * 20) / 20)
    names = sector_names(g)
    fig, axs = plt.subplots(2, 2, figsize=(width_cm * CM, 11.8 * CM))
    for s, ax in enumerate(axs.ravel()[:3]):
        means = {}
        for car in (d["follower"], d["leader"]):
            v = sectors[car][1:, s]
            col = car_color(d, car)
            ax.plot(laps, v, "-o", ms=3.4, lw=1.3, color=col, zorder=3)
            ax.axhline(v.mean(), color=col, lw=0.8, ls="--", alpha=0.85, zorder=2)
            b = int(np.argmin(v))
            ax.plot(laps[b], v[b], marker="*", ms=10, color=col, mec="white", mew=0.6, zorder=4)
            means[car] = v.mean()
        hi = max(means, key=means.get)
        for car, m in means.items():
            ax.text(n + 0.45, m, f"{m:.2f}", color=car_color(d, car), fontsize=7, ha="left",
                    va="bottom" if car == hi else "top")
        lap1 = ",  ".join(f"{car[-1]} {sectors[car][0, s]:.2f} s" for car in (d["leader"], d["follower"]))
        ax.text(0.02, 0.97, f"lap 1 (standing start): {lap1}", transform=ax.transAxes, fontsize=6.4, va="top",
                color="#5b646d")
        ax.set_title(f"Sector {s + 1}  ·  {names[s]}", color=SECTOR_LINE[s], fontsize=8.5, weight="bold",
                     loc="left")
        ax.set_ylim(*ylim)
        ax.set_xlim(1.5, n + 1.3)
        ax.set_xticks(laps)
        ax.grid(axis="y", color="#dde1e5", lw=0.5)
        ax.set_ylabel("Sector time (s)")
    for ax in axs[1, :]:
        ax.set_xlabel("Lap")
    ax = axs[1, 1]
    lap_times = {car: [l["time_s"] for l in rep["cars"][car]["laps"]] for car in ("Car A", "Car B")}
    gap = np.cumsum(lap_times[d["follower"]]) - np.cumsum(lap_times[d["leader"]])
    all_laps = np.arange(1, len(gap) + 1)
    ax.bar(all_laps, gap, width=0.62, color=FOLLOW_COLOR, alpha=0.85, zorder=3)
    for x, v in zip(all_laps, gap):
        ax.text(x, v + 0.06, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6, color="#4b545c")
    ax.set_ylim(0, gap.max() * 1.2)
    ax.set_xlim(0.4, len(gap) + 0.6)
    ax.set_xticks(all_laps)
    ax.set_title(f"Interval at the timing line  ·  {d['follower'][-1]} behind {d['leader'][-1]}", fontsize=8.5,
                 weight="bold", loc="left", color=INK)
    ax.set_ylabel("Interval (s)")
    ax.grid(axis="y", color="#dde1e5", lw=0.5, zorder=0)
    handles = [Line2D([], [], color=car_color(d, c), marker="o", ms=3.4, lw=1.3, label=role_label(d, c))
               for c in (d["follower"], d["leader"])]
    handles += [Line2D([], [], color="#5b646d", lw=0.8, ls="--", label="mean of laps 2–" + str(n)),
                Line2D([], [], color="#5b646d", marker="*", ms=9, lw=0, label="best sector")]
    fig.tight_layout(h_pad=1.2, w_pad=1.4, rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7.3, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main(runs):
    for run in runs:
        d = load(run)
        for name, fn in (("track", fig_track), ("sectors", fig_sectors), ("traces", fig_traces), ("laps", fig_laps)):
            out = f"{RESULTS}/{run}_{name}.png"
            fn(d, out)
            print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 200})
    runs = sys.argv[1:] or sorted(_os.path.basename(p)[:-5] for p in glob.glob(f"{RESULTS}/lap_telemetry_*.json"))
    main(runs)
    print("PLOTS DONE")
