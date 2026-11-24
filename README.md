# clinic-noshow-model

Can a clinic predict which appointments will be no-shows well enough to decide which slots to overbook, and what is that worth compared with holding every slot? This repo fits one classification tree to 400 past appointments and answers that with numbers on a held-out test set.

## Run

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Mac/Linux: .venv/bin/pip
.venv\Scripts\python analysis.py                # split, fit, evaluate, decide D and E, value -> reports/
.venv\Scripts\python figures.py                 # tree, confusion matrices, value chart -> figures/
```

## Files

| file | what it does |
|---|---|
| `data/clinic_appointments.csv` | 400 past appointments, one row each (the assignment's data) |
| `analysis.py` | 70/30 stratified split, the v1 tree, baseline vs v1 metrics, `decide()`, value per appointment and per year |
| `figures.py` | `sklearn.tree.plot_tree`, `ConfusionMatrixDisplay` and a matplotlib bar chart |
| `reports/results.md` | every result, plus the cross-validation check used to pick the depth |

## Choices

- **Split.** 30% of appointments held out as a test set (120 of 400), stratified so both parts keep the 22.8% no-show rate. Seed 42.
- **Features.** Text reminder, past no-shows, days booked ahead, new patient, distance and age. Appointment hour is left out because no-show rates by hour bounce between 14% and 28% with no trend.
- **Depth 2.** Picked by 5-fold cross-validation on the training rows only. Deeper trees did worse there.
- **No-shows weighted 2x.** A held slot that goes empty costs $120 and an overbooked slot where both patients come costs $60. With that weight, the tree predicts no-show for a group once more than 1 in 3 of its patients missed, which is where overbooking starts to pay.
- **Volume.** 20 appointments a day for 250 clinic days, so 5,000 a year.

## Results (test set, 120 appointments)

| | TN | FP | FN | TP | accuracy | precision | recall | value per appointment |
|---|---|---|---|---|---|---|---|---|
| baseline: hold every slot | 93 | 0 | 27 | 0 | 77.5% | n/a | 0.0% | -$27.00 |
| v1 tree | 75 | 18 | 6 | 21 | 80.0% | 53.8% | 77.8% | -$15.00 |

v1 accuracy is 81.8% on the training rows and 80.0% on the test rows. At 5,000 appointments a year, v1 loses $75,000 to no-shows instead of $135,000, a saving of about $60,000. The test set is small, so the saving per appointment has a 95% bootstrap range of $2.50 to $21.50.

AI use: an AI coding assistant helped write the code, including the chart code in `figures.py`. I ran every script myself, so the charts and numbers come from running this code.
