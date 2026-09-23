# Head-to-head re-run: racing-line policy vs. two-car self-play policy (2026-09-23)

`head_to_head_eval.py` repeats the journal's §8 head-to-head (`race_mixed_policy_eval.py`) on the **fixed**
`race_deploy.usd` (analytic ground plane) against a named self-play checkpoint, **run49** (final original-track
round, 40.09 M steps). The earlier heats (2026-09-19 10:39 and 12:19) ran before the deploy stage's ground fix
(17:06) against an early ~5 M-step checkpoint and were dominated by chassis-collapse recoveries (623-1,091 per
car); their "racing-line covers 19% more distance" is withdrawn.

Both policies act deterministically; the racing-line car sees its 7-D observation, the self-play car its 10-D
observation with the opponent features. Distance = cumulative wrap-corrected progress along the centerline
(126.77 m).

## Fixed-start heats (5,000 control steps = 333 s)

| Heat | Net | Lane RL / SP | Racing-line | Self-play | Bad-state steps RL / SP | Steps < 1.2 m / < 2 m apart |
|---|---|---|---|---|---|---|
| 1 | on | A / B | 1,658.95 m | 1,672.52 m | 239 / 253 | 470 / 1,350 |
| 2 (swap) | on | B / A | 1,985.07 m | 1,982.24 m | 0 / 0 | 0 / 15 |
| 1 | off | A / B | 1,289.84 m | 1,291.18 m | 8 / 0 | 0 / 4,941 |
| 2 (swap) | off | B / A | 1,985.07 m | 1,982.24 m | 0 / 0 | 0 / 15 |

Totals over both lanes: net on RL 3,644.0 m vs. SP 3,654.8 m; net off RL 3,274.9 m vs. SP 3,273.4 m (dead heat).
Flying laps of both policies in clean running: 21.0-21.13 s (top speed 6 m/s -> 21.1 s per lap).
Heat 1 without the net: the cars run side by side, within 2 m, for 4,941 of 5,000 steps and both slow to ~32.6 s
laps. With the net, a height event during that contact (step ~917) started a ~500-step teleport loop (both cars
put back onto the centerline next to each other) - an artifact of the net. The car in lane B finished ahead in all
fixed-start heats.

## Random-start series (10 start segments x 2 lane assignments, 1,500 control steps = 100 s, net off)

`head_to_head_run49_series10x1500.json` (start segments drawn with seed 4000; training-style side-by-side start,
1.5 m either side of the centerline).

* Racing-line ahead in **17 of 20** heats, self-play in 3; lane B ahead in 7 of 20 (no lane effect here).
* 16 heats without an incident: racing-line ahead in 14, on average **+0.5% distance** (typically 2-8 m per ~585 m).
* The self-play car left the track in 3 heats (start segments 36, 30, 9; lane A), the racing-line car in 2
  (segments 9, 34), each time shortly after the cars had come close (first a height event, then off-track).
* Mean distance 553.6 m (racing-line) vs. 497.2 m (self-play); these means are dominated by the crashes.

## Reading

Both policies drive at essentially the car's top speed on this track. In clean running the racing-line policy is
marginally faster (~0.5%), plausibly because its centering weight is 0.1 against the self-play policy's 0.9. The
outcome of a single heat is decided by what happens when the cars meet, not by pace. The racing-line policy is
not decisively better, and the opponent-aware self-play policy does not beat it either.
