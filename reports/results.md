# Results

Written by `python analysis.py`.

## Data and split

- 400 appointments, 8 columns, 91 no-shows (22.8%).
- Training: 280 appointments (64 no-shows). Test: 120 appointments (27 no-shows). Stratified, seed 42.

## Model v1

- Classification tree, depth 2, at least 10 training appointments per leaf.
- Features offered: sms_reminder, prior_no_shows, lead_time_days, new_patient, distance_miles, age.
- Features the tree splits on, from the root: sms_reminder, prior_no_shows.
- No-shows weighted 2x (an empty slot costs $120, an overrun $60), so a leaf predicts no-show when more than 33% of it missed.

```
|--- sms_reminder <= 0.50
|   |--- prior_no_shows <= 0.50
|   |   |--- class: 0
|   |--- prior_no_shows >  0.50
|   |   |--- class: 1
|--- sms_reminder >  0.50
|   |--- prior_no_shows <= 1.50
|   |   |--- class: 0
|   |--- prior_no_shows >  1.50
|   |   |--- class: 1
```

| leaf (left to right) | training appointments | no-shows | no-show rate | decision |
|---|---|---|---|---|
| 1 | 56 | 10 | 17.9% | hold |
| 2 | 62 | 39 | 62.9% | overbook |
| 3 | 139 | 5 | 3.6% | hold |
| 4 | 23 | 10 | 43.5% | overbook |

## Test appointments: baseline vs v1

| | TN (held, came) | FP (overbooked, came) | FN (held, no-show) | TP (overbooked, no-show) | accuracy | precision | recall | value total | value per appointment |
|---|---|---|---|---|---|---|---|---|---|
| baseline (hold all) | 93 | 0 | 27 | 0 | 77.5% | n/a | 0.0% | -$3,240.00 | -$27.00 |
| v1 tree | 75 | 18 | 6 | 21 | 80.0% | 53.8% | 77.8% | -$1,800.00 | -$15.00 |

v1 accuracy: training 81.8%, test 80.0%. Baseline accuracy: training 77.1%, test 77.5%.

## Upcoming appointments

- D: leaf 2 (no text reminder, 1+ past no-shows) -> **overbook**
- E: leaf 3 (text reminder sent, 0 or 1 past no-shows) -> **hold**

## Value

- Per appointment: baseline -$27.00, v1 -$15.00, saving $12.00 (95% bootstrap range $2.50 to $21.50).
- Assumed volume: 20 appointments a day x 250 days = 5,000 a year.
- Per year: baseline -$135,000.00, v1 -$75,000.00, saving $60,000.00 (44% less lost to no-shows).

## How depth was picked (5-fold CV on the training rows only)

| depth | CV accuracy | CV value per appointment |
|---|---|---|
| 1 | 68.2% | -$24.00 |
| 2 | 81.8% | -$15.64 |
| 3 | 77.9% | -$18.00 |
| 4 | 76.4% | -$18.64 |
| 5 | 77.9% | -$18.43 |
| baseline | 77.1% | -$27.43 |
