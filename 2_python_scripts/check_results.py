# check_results.py - recomputes the thesis numbers from the result files.
# Run it from anywhere:   python 2_python_scripts/check_results.py   (Python 3; NumPy only for part 6)
import csv, glob, json, os, statistics, zipfile

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the project folder

EV = "4_results/"
SC = EV + "start_condition_results/"

def rows(name):
    return list(csv.DictReader(open(SC + name)))

print("1) Race evaluations - incident-free heats, as in the thesis tables")
for f in sorted(glob.glob(EV + "two_car_race_*_*.json")):
    if "trace" in f or "summary" in f:
        continue
    heats = json.load(open(f))["heats"]
    clean = [h for h in heats if not any(c["first_event"] for c in h["cars"])]
    laps = [x for h in clean for c in h["cars"] for x in c["flying_laps_s"]]
    lc = sum(h["lead_changes"] for h in heats)
    name = os.path.basename(f)[13:-5]
    print(f"   {name:34} clean {len(clean):2}/{len(heats)}   lap "
          f"{statistics.mean(laps):.2f} +/- {statistics.pstdev(laps):.2f} s   lead changes {lc}")

print("2) All stadium heats of the self-play lineage (Thesis 6.6-6.7)")
labels = ["run49", "newtrack_run3", "newtrack_run12", "newtrack_overtake_run2",
          "newtrack_smooth_run1", "newtrack_smooth_run20", "newtrack_smooth_run35",
          "newtrack_compete_run6", "100M_milestone"]
s = [json.load(open(f"{EV}two_car_race_stadium_{l}.json"))["summary"] for l in labels]
print(f"   heats {sum(x['heats'] for x in s)}, lead changes {sum(x['lead_changes_total'] for x in s)}, "
      f"inside starter leads at the end {sum(x['left_start_leads_at_end'] for x in s)}")

print("3) Lock-in and passing tests (Thesis 6.8, 6.10)")
for n in ["start_offsets_competition_round4.csv", "start_offsets_competition_round6.csv",
          "start_offsets_100M.csv"]:
    r = rows(n)
    k = sum((float(x["dxA"]) > float(x["dxB"])) == (x["leader_label"] == "A") for x in r)
    print(f"   {n:40} car that started further forward leads: {k} of {len(r)}")
seeds = {}
for x in rows("mirrored_100M.csv"):
    seeds.setdefault(x["seed"], set()).add(x["leader_label"])
print(f"   mirrored_100M.csv: same physical start wins in "
      f"{sum(len(v) == 2 for v in seeds.values())} of {len(seeds)} seeds")
for n in ["passing_test_100M_before.csv", "passing_test_100M_after_staggered.csv"]:
    r = rows(n)
    print(f"   {n:40} car starting behind leads at the end: "
          f"{sum(x['leads_at_end'] == '1' for x in r)} of {len(r)}")
heats = json.load(open(EV + "two_car_race_stadium_newtrack_stagger_run3.json"))["heats"]
first = [c["first_event"][0] for h in heats for c in h["cars"] if c["first_event"]]
print(f"   staggered policy: {sum(x > 500 for x in first)} of {len(first)} first incidents "
      f"after step 500, median step {statistics.median(first):.0f}")

print("4) Training steps stored inside the model files")
for p in ["3_trained_models/car2_ppo_model.zip",
          "3_trained_models/car2_vec_ppo_model.zip",
          "3_trained_models/car2_racingline_ppo_model.zip",
          "3_trained_models/car_race_ppo_model.zip"]:
    n = json.loads(zipfile.ZipFile(p).read("data"))["num_timesteps"]
    print(f"   {os.path.basename(p):32} {n:>13,} steps")

print("5) Lap telemetry of the final policy, flying laps 2-10 (Thesis 6.9)")
for run in ["lap_telemetry_100M", "lap_telemetry_100M_swapped", "lap_telemetry_100M_a_ahead2m"]:
    rep = json.load(open(f"{EV}{run}.json"))
    label = ("positions exchanged" if rep["start_positions_swapped"] else
             f"Car A {rep['car_a_ahead_m']:g} m ahead" if rep.get("car_a_ahead_m") else "authored start")
    steps = {}
    for x in csv.DictReader(open(f"{EV}{run}.csv")):
        if 2 <= int(x["lap"]) <= rep["n_laps"]:
            steps.setdefault(x["car"], []).append(x)
    lead = max(steps, key=lambda c: statistics.median(float(x["gap_m"]) for x in steps[c]))
    for car in sorted(steps, key=lambda c: c != lead):
        f, s = rep["cars"][car]["flying_laps"], steps[car]
        full = 100 * sum(float(x["throttle_cmd"]) >= 0.99 for x in s) / len(s)
        reversals = []
        for lap in range(2, rep["n_laps"] + 1):
            signs = [(v > 0) - (v < 0) for v in (float(x["steer_deg"]) for x in s if int(x["lap"]) == lap)]
            signs = [v for v in signs if v]
            reversals.append(sum(a != b for a, b in zip(signs, signs[1:])))
        sm = f["sector_mean_s"]
        print(f"   {label if car == lead else '':20} {'leader  ' if car == lead else 'follower'} {car}  lap "
              f"{f['lap_mean_s']:.2f} +/- {f['lap_std_s']:.2f} s  S1 {sm[0]:.2f}  S2 {sm[1]:.2f}  S3 {sm[2]:.2f}  "
              f"full throttle {full:5.1f}%  steering reversals/lap {statistics.mean(reversals):3.0f}")

print("6) Behavioral-cloning demonstrations (Thesis 5.4; needs NumPy)")
try:
    import numpy as np
    for f in sorted(glob.glob("5_data/human_driving_data*.npz")):
        cos = np.load(f)["obs"][:, 2]          # cos(heading error)
        print(f"   {os.path.basename(f):28} mean cos {cos.mean():+.2f}, "
              f"facing backwards {100 * (cos < 0).mean():.0f}% of samples")
except ImportError:
    print("   NumPy not installed - skipped (pip install numpy)")
