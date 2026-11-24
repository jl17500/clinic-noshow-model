"""Charts: the v1 tree, test confusion matrices, and value per appointment.

Run `python figures.py` after analysis.py. Each chart opens in its own window and is saved to figures/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree

from analysis import FEATURES, condition, make_tree, split

ROOT = Path(__file__).parent
OUT = ROOT / "figures"
CLASSES = ["show", "no-show"]


def tree_chart(v1, X_train, y_train):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = plot_tree(v1, feature_names=FEATURES, class_names=CLASSES, filled=True, rounded=True,
                       impurity=False, fontsize=13, ax=ax)
    # plain-word labels; sklearn's own counts double each no-show (2x weight)
    t = v1.tree_
    in_node = v1.decision_path(X_train).toarray().astype(bool)
    boxes = [a for a in labels if "samples" in a.get_text()]  # node boxes, in node order
    for node, box in enumerate(boxes):
        n = int(in_node[:, node].sum())
        no_shows = int(y_train[in_node[:, node]].sum())
        counts = f"{no_shows} of {n} missed"
        if t.children_left[node] == -1:
            decision = "OVERBOOK" if v1.classes_[t.value[node][0].argmax()] == 1 else "HOLD"
            box.set_text(f"{counts}\n{decision}")
        else:
            question = condition(FEATURES[t.feature[node]], t.threshold[node], goes_left=True)
            box.set_text(f"{question[0].upper()}{question[1:]}?\n{counts}")
    for arrow_label in labels:
        word = arrow_label.get_text().strip()  # sklearn pads these with spaces
        if word in ("True", "False"):
            arrow_label.set_text("  Yes" if word == "True" else "No  ")
    ax.set_title(f"v1 decision tree, fit on the {len(y_train)} training appointments", fontsize=14)
    fig.savefig(OUT / "tree.png", dpi=200, bbox_inches="tight")


def confusion_chart(y_test, pred_v1):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, pred, title in [(axes[0], [0] * len(y_test), "Baseline: hold every slot"),
                            (axes[1], pred_v1, "v1 tree")]:
        ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1], display_labels=CLASSES,
                                                cmap="Blues", colorbar=False, ax=ax,
                                                text_kw={"fontsize": 16})
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.tick_params(labelsize=12)
    fig.suptitle(f"Confusion matrices on the {len(y_test)} test appointments", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "confusion.png", dpi=200, bbox_inches="tight")


def value_chart(results):
    base, v1 = results["test"]["baseline"], results["test"]["v1"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    values = [base["value_per_appt"], v1["value_per_appt"]]
    bars = ax.bar(["Baseline\n(hold every slot)", "v1 tree\n(overbook flagged)"], values,
                  color=["tab:gray", "tab:blue"], width=0.55)
    ax.bar_label(bars, labels=[f"-${-v:.2f}" for v in values], padding=4, fontsize=14)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(min(values) * 1.25, 0)
    ax.set_ylabel("Value per appointment ($)", fontsize=12)
    ax.set_title("Value per test appointment", fontsize=13)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "value.png", dpi=200, bbox_inches="tight")


def main():
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    X_train, X_test, y_train, y_test = split()
    v1 = make_tree().fit(X_train, y_train)
    OUT.mkdir(exist_ok=True)
    tree_chart(v1, X_train, y_train)
    confusion_chart(y_test, v1.predict(X_test))
    value_chart(results)
    print(f"saved tree.png, confusion.png and value.png in {OUT}")
    plt.show()  # one window per chart


if __name__ == "__main__":
    main()
