"""Clinic no-shows: fit a classification tree and decide which slots to overbook.

Run `python analysis.py`. It prints every result and writes reports/results.json
(read by figures.py) and reports/results.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "clinic_appointments.csv"
REPORTS = ROOT / "reports"

SEED = 42                                   # split seed
TEST_SHARE = 0.30                           # 120 of 400 appointments held out
FEATURES = ["sms_reminder", "prior_no_shows", "lead_time_days",
            "new_patient", "distance_miles", "age"]  # appointment_hour left out: no pattern by hour
MAX_DEPTH = 2                               # picked by 5-fold CV on the training rows
MIN_LEAF = 10                               # every group holds at least 10 training appointments
VALUE = {"tn": 0, "fp": -60, "fn": -120, "tp": 0}  # $ per appointment, from the cost table
NO_SHOW_WEIGHT = VALUE["fn"] / VALUE["fp"]  # 2.0: an empty slot costs twice an overrun
APPTS_PER_DAY, CLINIC_DAYS = 20, 250        # assumed volume: 5,000 appointments a year

# With no-shows counted twice, a group is called "no-show" once more than 1 in 3 of
# its appointments were missed. That is the break-even: overbooking pays above 1/3.
BREAK_EVEN = VALUE["fp"] / (VALUE["fp"] + VALUE["fn"])

UPCOMING = {
    "D": dict(lead_time_days=21, prior_no_shows=1, age=27, distance_miles=9.5,
              sms_reminder=0, new_patient=1, appointment_hour=15),
    "E": dict(lead_time_days=3, prior_no_shows=0, age=61, distance_miles=2.0,
              sms_reminder=1, new_patient=0, appointment_hour=9),
}


def load():
    df = pd.read_csv(DATA)
    y = (df["no_show"] == "no-show").astype(int)  # 1 = no-show
    return df, y


def make_tree(depth=MAX_DEPTH):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_LEAF,
                                  class_weight={0: 1, 1: NO_SHOW_WEIGHT}, random_state=0)


def scorecard(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    total = tn * VALUE["tn"] + fp * VALUE["fp"] + fn * VALUE["fn"] + tp * VALUE["tp"]
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(n),
        "accuracy": (tn + tp) / n,
        "precision": tp / (tp + fp) if tp + fp else None,  # None: never predicts a no-show
        "recall": tp / (tp + fn),
        "value_total": int(total),
        "value_per_appt": total / n,
    }


def decide(appointment, model):
    """Overbook the slot if the tree predicts a no-show, hold it otherwise."""
    row = pd.DataFrame([appointment])[FEATURES]
    return "overbook" if model.predict(row)[0] == 1 else "hold"


def condition(feature, threshold, goes_left):
    """One split in plain words, e.g. 'no text reminder' or '1+ past no-shows'."""
    cut = int(threshold)  # thresholds sit halfway between whole numbers
    if feature == "sms_reminder":
        return "no text reminder" if goes_left else "text reminder sent"
    if feature == "prior_no_shows":
        if goes_left:
            return {0: "no past no-shows", 1: "0 or 1 past no-shows"}.get(cut, f"{cut} or fewer past no-shows")
        return f"{cut + 1}+ past no-shows"
    return f"{feature} {'<=' if goes_left else '>'} {threshold:g}"


def tree_nodes(model, X_train, y_train):
    """The fitted tree as a list of nodes, with unweighted training counts per leaf."""
    t = model.tree_
    leaf_of_row = model.apply(X_train)
    nodes = []
    for i in range(t.node_count):
        node = {"id": i, "left": int(t.children_left[i]), "right": int(t.children_right[i])}
        if node["left"] == -1:
            in_leaf = y_train[leaf_of_row == i]
            node.update(leaf=True, n=int(len(in_leaf)), no_shows=int(in_leaf.sum()),
                        rate=float(in_leaf.mean()),
                        decision="overbook" if model.classes_[t.value[i][0].argmax()] == 1 else "hold")
        else:
            node.update(leaf=False, feature=FEATURES[t.feature[i]], threshold=float(t.threshold[i]))
        nodes.append(node)
    leaves = [n for n in nodes if n["leaf"]]  # node ids are depth-first, so this is left to right
    for number, n in enumerate(leaves, start=1):
        n["leaf_number"] = number
    _add_paths(model, nodes, 0, [])
    return nodes


def _add_paths(model, nodes, i, steps):
    """Store on each leaf the splits that lead to it, in plain words."""
    t = model.tree_
    if t.children_left[i] == -1:
        nodes[i]["path"] = steps
        return
    feature, thr = FEATURES[t.feature[i]], t.threshold[i]
    _add_paths(model, nodes, t.children_left[i], steps + [condition(feature, thr, True)])
    _add_paths(model, nodes, t.children_right[i], steps + [condition(feature, thr, False)])


def leaf_path(model, appointment):
    """The splits one appointment follows from the root to its leaf, in plain words."""
    row = pd.DataFrame([appointment])[FEATURES]
    t = model.tree_
    steps, i = [], 0
    while t.children_left[i] != -1:
        feature, thr = FEATURES[t.feature[i]], t.threshold[i]
        goes_left = row.iloc[0][feature] <= thr
        steps.append(condition(feature, thr, goes_left))
        i = t.children_left[i] if goes_left else t.children_right[i]
    return int(i), steps


def cv_depth_check(X_train, y_train):
    """5-fold CV value per appointment on the training rows, by depth (how depth 2 was picked)."""
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for depth in range(1, 6):
        pred = cross_val_predict(make_tree(depth), X_train, y_train, cv=folds)
        rows.append({"depth": depth, **scorecard(y_train, pred)})
    rows.append({"depth": "baseline", **scorecard(y_train, np.zeros(len(y_train), dtype=int))})
    return rows


def bootstrap_savings(y_test, pred_v1, reps=10_000, seed=0):
    """95% range for v1's saving per appointment over the baseline, resampling test rows."""
    per_appt_v1 = np.where(pred_v1 == 1, np.where(y_test == 1, VALUE["tp"], VALUE["fp"]),
                           np.where(y_test == 1, VALUE["fn"], VALUE["tn"]))
    per_appt_base = np.where(y_test == 1, VALUE["fn"], VALUE["tn"])
    saving = per_appt_v1 - per_appt_base
    rng = np.random.default_rng(seed)
    means = saving[rng.integers(0, len(saving), (reps, len(saving)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def split():
    """The 70/30 split, stratified so both parts keep the same no-show rate."""
    df, y = load()
    return train_test_split(df[FEATURES], y, test_size=TEST_SHARE, stratify=y, random_state=SEED)


def main():
    df, y = load()
    X_train, X_test, y_train, y_test = split()

    v1 = make_tree().fit(X_train, y_train)
    pred_test = v1.predict(X_test)
    baseline = scorecard(y_test, np.zeros(len(y_test), dtype=int))  # hold every slot
    model = scorecard(y_test, pred_test)
    train_accuracy = float((v1.predict(X_train) == y_train).mean())
    baseline_train_accuracy = float((y_train == 0).mean())

    appts_per_year = APPTS_PER_DAY * CLINIC_DAYS
    nodes = tree_nodes(v1, X_train, y_train)
    upcoming = {}
    for name, appt in UPCOMING.items():
        node_id, steps = leaf_path(v1, appt)
        leaf = next(n["leaf_number"] for n in nodes if n["id"] == node_id)
        upcoming[name] = {"appointment": appt, "leaf": leaf, "path": steps, "decision": decide(appt, v1)}

    results = {
        "data": {"rows": len(df), "columns": df.shape[1], "no_shows": int(y.sum()),
                 "no_show_rate": float(y.mean()),
                 "train_rows": len(X_train), "train_no_shows": int(y_train.sum()),
                 "test_rows": len(X_test), "test_no_shows": int(y_test.sum())},
        "model": {"features_offered": FEATURES, "max_depth": MAX_DEPTH, "min_leaf": MIN_LEAF,
                  "no_show_weight": NO_SHOW_WEIGHT, "break_even": BREAK_EVEN,
                  "features_used": list(dict.fromkeys(n["feature"] for n in nodes if not n["leaf"])),  # root first
                  "nodes": nodes,
                  "rules": export_text(v1, feature_names=FEATURES)},
        "test": {"baseline": baseline, "v1": model},
        "v1_train_accuracy": train_accuracy,
        "baseline_train_accuracy": baseline_train_accuracy,
        "upcoming": upcoming,
        "value": {"appts_per_day": APPTS_PER_DAY, "clinic_days": CLINIC_DAYS,
                  "appts_per_year": appts_per_year,
                  "baseline_per_year": baseline["value_per_appt"] * appts_per_year,
                  "v1_per_year": model["value_per_appt"] * appts_per_year,
                  "saving_per_appt": model["value_per_appt"] - baseline["value_per_appt"],
                  "saving_per_year": (model["value_per_appt"] - baseline["value_per_appt"]) * appts_per_year,
                  "saving_per_appt_95": bootstrap_savings(y_test.to_numpy(), pred_test),
                  "loss_cut": 1 - model["value_total"] / baseline["value_total"]},
        "cv_depth_check": cv_depth_check(X_train, y_train),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    (REPORTS / "results.md").write_text(report(results))
    print(report(results))


def money(x):
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def report(r):
    d, b, m, v = r["data"], r["test"]["baseline"], r["test"]["v1"], r["value"]
    lines = [
        "# Results",
        "",
        "Written by `python analysis.py`.",
        "",
        "## Data and split",
        "",
        f"- {d['rows']} appointments, {d['columns']} columns, {d['no_shows']} no-shows ({d['no_show_rate']:.1%}).",
        f"- Training: {d['train_rows']} appointments ({d['train_no_shows']} no-shows). "
        f"Test: {d['test_rows']} appointments ({d['test_no_shows']} no-shows). Stratified, seed {SEED}.",
        "",
        "## Model v1",
        "",
        f"- Classification tree, depth {r['model']['max_depth']}, at least {r['model']['min_leaf']} training appointments per leaf.",
        f"- Features offered: {', '.join(r['model']['features_offered'])}.",
        f"- Features the tree splits on, from the root: {', '.join(r['model']['features_used'])}.",
        f"- No-shows weighted {r['model']['no_show_weight']:g}x (an empty slot costs $120, an overrun $60), "
        f"so a leaf predicts no-show when more than {r['model']['break_even']:.0%} of it missed.",
        "",
        "```",
        r["model"]["rules"].rstrip(),
        "```",
        "",
        "| leaf (left to right) | training appointments | no-shows | no-show rate | decision |",
        "|---|---|---|---|---|",
    ]
    for n in r["model"]["nodes"]:
        if n["leaf"]:
            lines.append(f"| {n['leaf_number']} | {n['n']} | {n['no_shows']} | {n['rate']:.1%} | {n['decision']} |")
    lines += [
        "",
        "## Test appointments: baseline vs v1",
        "",
        "| | TN (held, came) | FP (overbooked, came) | FN (held, no-show) | TP (overbooked, no-show) "
        "| accuracy | precision | recall | value total | value per appointment |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in [("baseline (hold all)", b), ("v1 tree", m)]:
        lines.append(f"| {name} | {s['tn']} | {s['fp']} | {s['fn']} | {s['tp']} | {pct(s['accuracy'])} "
                     f"| {pct(s['precision'])} | {pct(s['recall'])} | {money(s['value_total'])} "
                     f"| {money(s['value_per_appt'])} |")
    lines += [
        "",
        f"v1 accuracy: training {r['v1_train_accuracy']:.1%}, test {m['accuracy']:.1%}. "
        f"Baseline accuracy: training {r['baseline_train_accuracy']:.1%}, test {b['accuracy']:.1%}.",
        "",
        "## Upcoming appointments",
        "",
    ]
    for name, u in r["upcoming"].items():
        lines.append(f"- {name}: leaf {u['leaf']} ({', '.join(u['path'])}) -> **{u['decision']}**")
    lo, hi = v["saving_per_appt_95"]
    lines += [
        "",
        "## Value",
        "",
        f"- Per appointment: baseline {money(b['value_per_appt'])}, v1 {money(m['value_per_appt'])}, "
        f"saving {money(v['saving_per_appt'])} (95% bootstrap range {money(lo)} to {money(hi)}).",
        f"- Assumed volume: {v['appts_per_day']} appointments a day x {v['clinic_days']} days = "
        f"{v['appts_per_year']:,} a year.",
        f"- Per year: baseline {money(v['baseline_per_year'])}, v1 {money(v['v1_per_year'])}, "
        f"saving {money(v['saving_per_year'])} ({v['loss_cut']:.0%} less lost to no-shows).",
        "",
        "## How depth was picked (5-fold CV on the training rows only)",
        "",
        "| depth | CV accuracy | CV value per appointment |",
        "|---|---|---|",
    ]
    for row in r["cv_depth_check"]:
        lines.append(f"| {row['depth']} | {pct(row['accuracy'])} | {money(row['value_per_appt'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
