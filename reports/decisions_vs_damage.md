# Decisions vs. Damage — methodology note and Kyle Tucker case study

**Script:** `python/05_decisions_vs_damage.py`
**Tests:** `tests/test_05_decisions_vs_damage.py`
**Source idea:** Driveline Baseball's Kyle Tucker 2026 vs. 2023 breakdown (Oct 2026).

## The question

When a hitter's production drops, where did it break? Driveline's read on Tucker:
he still swings at the right pitches; he just isn't doing damage on them anymore.
A late torso coil and a struggle to get open produce an aggressive, flat bat path,
so pitches in the lower zone that he used to lift now become ground balls.

This module turns that read into a repeatable test for any hitter.

## The diagnostic chain

```
1 Decisions -> 2 Inputs -> 3 Delivery -> 4 Contact quality -> 5 Outcomes
```

| Layer | Question | Metrics |
|---|---|---|
| 1 Decisions | Is he swinging at the right pitches? | Swing%, zone swing%, chase%, zone-minus-chase gap, BB% |
| 2 Inputs | Is the engine the same? | Bat speed, fast-swing% (75+ mph), swing length |
| 3 Delivery | Is he getting the barrel to the ball the same way? | Attack angle, ideal attack angle% (5–20°), swing path tilt, attack direction, contact depth |
| 4 Contact quality | Is the energy transferring? | Zone contact%, whiff%, squared-up%, collision efficiency, blast%, LA sweet-spot% |
| 5 Outcomes | What happened to the ball? | EV, HardHit%, barrel%, GB%, launch angle, K%, wOBA, xwOBA, xwOBAcon |

Read it left to right. The **first layer that moved meaningfully in the wrong
direction** is where the problem starts. Everything to its right is consequence.

## Key definitions

| Metric | Definition |
|---|---|
| Max possible EV | `1.23 × bat speed + 0.23 × pitch speed` (Savant's collision model) |
| Squared-up | EV ≥ 80% of max possible EV |
| Blast | `100 × (EV / max EV) + bat speed ≥ 164` |
| Collision efficiency `q` | Solves `EV = q·pitch + (1+q)·bat` → `q = (EV − bat) / (pitch + bat)` |
| Pitch speed | Speed at the front of the plate from the 9-parameter fit (`vx0…az`), not release speed |
| Competitive swings | No bunts; drop the hitter's slowest 10% of swings in the window unless EV ≥ 90 (approximates Savant's rule) |
| Contact depth / attack direction | Sign set from this hitter's own data so **+ = out front** and **+ = pull side** |
| Zone grid | Statcast zones 1–9, catcher's view |

## When is a change real?

Two tests, both required:

1. **Statistically real.** Game-level (cluster) bootstrap of both windows. Each
   layer gets a 10% family-wise error budget, split across its metrics
   (Bonferroni). Without that, ~30 metrics at 90% would flag a few "changes" by
   chance in every comparison.
2. **Big enough to matter.** `|Δ|` must clear a minimum effect (`MIN_EFFECT` in
   the script): e.g. 1.0 mph bat speed, 2° attack angle, 2 pp chase, .015 xwOBA.

Verdicts: `better`, `worse`, `shifted` (descriptive metric moved, e.g. attack
angle), `trivial` (real but too small), `stable`, `missing`.

The `trivial` verdict is the lesson of the Tucker post: with 2,000+ tracked
swings a season, a 0.5 mph bat-speed drop can be statistically significant and
still explain nothing.

## Case study: Kyle Tucker (MLBAM 663656)

### Published reference (Driveline, 2023 vs. 2026)

| Layer | Metric | 2023 | 2026 | Read |
|---|---|---|---|---|
| Decisions | Chase% | 21.6% | 23.9% | Up 2.3 pp, still 87th pctl |
| Decisions | BB% | 11.9% | 12.4% | Held |
| Inputs | Bat speed | 71.9 | 71.4 | Flat |
| Contact | Squared-up% | 29.9% | 27.0% | Down |
| Contact | Whiff% | 18.5% | 20.8% | Up |
| Contact | LA sweet-spot% | 35.7% | 34.7% | Flat |
| Outcomes | Avg EV | 90.2 | 88.5 | Down 1.7 mph |
| Outcomes | HardHit% | 45.9% | 36.3% | Down ~10 pp |
| Outcomes | Barrel% | 10.6% | 6.4% | Down ~40% |
| Outcomes | xwOBA | .386 | .331 | Down .055 |
| Outcomes | K% | 13.6% | 16.7% | Up |

Driveline's grades (Swing Decision 72 → 72, Hitting 67 → 55, Contact 60 → 56,
Power 56 → 47) are proprietary and not replicated here.

### Hypotheses the script tests

- **H1 Decisions held.** Chase and zone-swing changes stay inside the minimum effect.
- **H2 Inputs held.** Bat speed is `stable` or `trivial`.
- **H3 Delivery changed.** Flatter attack angle and/or lower ideal-attack-angle%,
  and possibly a deeper contact point (late coil → late barrel).
- **H4 Contact quality fell.** Squared-up% and collision efficiency down.
- **H5 Low-zone damage collapsed.** Zone grid: bottom-row xwOBAcon down and
  bottom-row GB% up versus the base window (Driveline's heat-map slide).

If H1–H2 hold and the chain first breaks at Delivery, the analysis agrees with
Driveline. If it breaks at Decisions instead, the 2.3 pp chase increase is
carrying more weight than the post implies.

### Run it

```bash
pip install pybaseball pandas numpy matplotlib
python python/05_decisions_vs_damage.py            # defaults: Tucker, 2023 2024 2025 2026, base 2023 vs 2026
python python/05_decisions_vs_damage.py --base 2025 --compare 2026   # the 2025 vs 2026 question
```

Bat tracking is not public in the pitch CSV before 2024. With 2023 as the base,
Inputs/Delivery/Contact-quality tracking metrics are compared from the earliest
tracked window (2024) instead; the `base_window` column says which.

Outputs land in `output/tables/dvd_kyle_tucker_*.csv` and
`output/figures/dvd_kyle_tucker_{chain,zones}.png`. The reference file compares
computed values with Driveline's published numbers; expect small gaps from
definitions (Savant's competitive-swing filter, per-swing vs. per-contact
squared-up, and Driveline's 2023 tracking numbers most likely coming from 2023 H2).

### Results

_Pending a local run — Baseball Savant is not reachable from the build
environment. Paste the console diagnosis and the two figures here._

## Applying it beyond MLB

- **Campo Las Palmas prospects.** The chain works with any bat sensor
  (Blast, Diamond Kinetics) plus Trackman/Rapsodo: Decisions from pitch
  location, Inputs/Delivery from the sensor, Contact quality and Outcomes from
  ball flight. The same "first broken layer" read tells a coach whether to work
  on approach, strength, or swing path.
- **LIDOM data platform.** Run the script for every LIDOM roster player with MLB
  Statcast history and store the change table per player-season; the
  `first broken layer` becomes a scouting tag.

## Known limitations

1. **Fouls carry no EV in the public CSV**, so squared-up per swing counts only
   balls in play. It reads low against Savant's own number.
2. **Competitive swings are approximated.** Savant also removes check swings,
   which the pitch CSV does not mark.
3. **Plate speed is modelled** from the 9-parameter fit; Savant may use a
   different reference point for "pitch speed at contact".
4. **xwOBA** uses Savant's per-BBE `estimated_woba_using_speedangle`; Savant's
   published xwOBA also adjusts some batted balls for sprint speed.
5. **Multiple comparisons.** The per-layer error budget is 10%, so about one
   comparison in ten will still show a chance flag somewhere in a layer.
6. **Mechanics are inferred, not seen.** Attack angle, tilt and contact depth
   are consistent with "late coil, flat path", but only video confirms the cause.

## Testing

```bash
python -m pytest tests -q
python python/05_decisions_vs_damage.py --demo    # synthetic hitter with a planted Delivery break
```

The demo builds a hitter whose 2025–26 seasons keep bat speed and decisions but
flatten the attack angle and lose collision efficiency. The diagnosis should
read `Chain breaks first at: 3 Delivery`, with bat speed flagged as trivial.
