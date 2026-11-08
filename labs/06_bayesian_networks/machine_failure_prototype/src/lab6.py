import os
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Ellipse, FancyArrowPatch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "machine_failure.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "machine_failure.csv"
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GRAPH_PATH = os.path.join(OUTPUT_DIR, "bayesian_network_structure.png")
TARGET_COLUMN = "Machine failure"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

def run_eda(df: pd.DataFrame) -> None:
    print("\n=== EDA: DATASET SHAPE ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== EDA: COLUMN NAMES ===")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")

    print("\n=== EDA: DATA TYPES ===")
    print(df.dtypes)

    print("\n=== EDA: MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_percent
    })
    print(missing_table)

    print(f"\n=== EDA: TARGET DISTRIBUTION ({TARGET_COLUMN}) ===")
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    target_percent = (df[TARGET_COLUMN].value_counts(normalize=True, dropna=False).sort_index() * 100).round(2)
    target_table = pd.DataFrame({
        "count": target_counts,
        "percent": target_percent
    })
    print(target_table)

    print("\n=== EDA: NUMERIC FEATURES SUMMARY ===")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print(df[numeric_cols].describe().T)
    else:
        print("No numeric columns found.")

def load_and_preprocess_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    print("=== RAW FIRST 5 ROWS ===")
    print(df.head())

    run_eda(df)

    selected_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    ]

    df = df[selected_columns].copy()
    df = df.dropna()

    df["Air_temp_cat"] = pd.qcut(
        df["Air temperature [K]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    df["Process_temp_cat"] = pd.qcut(
        df["Process temperature [K]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    df["Speed_cat"] = pd.qcut(
        df["Rotational speed [rpm]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    df["Torque_cat"] = pd.qcut(
        df["Torque [Nm]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    df["Wear_cat"] = pd.qcut(
        df["Tool wear [min]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    bn_df = df[
        [
            "Type",
            "Air_temp_cat",
            "Process_temp_cat",
            "Speed_cat",
            "Torque_cat",
            "Wear_cat",
            "Machine failure",
        ]
    ].copy()

    bn_df["Type"] = bn_df["Type"].astype(str)
    bn_df["Air_temp_cat"] = bn_df["Air_temp_cat"].astype(str)
    bn_df["Process_temp_cat"] = bn_df["Process_temp_cat"].astype(str)
    bn_df["Speed_cat"] = bn_df["Speed_cat"].astype(str)
    bn_df["Torque_cat"] = bn_df["Torque_cat"].astype(str)
    bn_df["Wear_cat"] = bn_df["Wear_cat"].astype(str)
    bn_df["Machine failure"] = bn_df["Machine failure"].map({0: "no", 1: "yes"}).astype(str)

    print("\n=== PREPROCESSED FIRST 5 ROWS ===")
    print(bn_df.head())

    print("\n=== PREPROCESSED DATA TYPES ===")
    print(bn_df.dtypes)

    print("\n=== PREPROCESSED TARGET DISTRIBUTION ===")
    print(bn_df["Machine failure"].value_counts())
    print((bn_df["Machine failure"].value_counts(normalize=True) * 100).round(2))

    return bn_df

def build_model() -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork(
        [
            ("Type", "Torque_cat"),
            ("Type", "Speed_cat"),
            ("Air_temp_cat", "Process_temp_cat"),
            ("Torque_cat", "Machine failure"),
            ("Wear_cat", "Machine failure"),
            ("Speed_cat", "Machine failure"),
            ("Process_temp_cat", "Machine failure"),
        ]
    )
    return model

def fit_model(model: DiscreteBayesianNetwork, data: pd.DataFrame) -> None:
    model.fit(data, estimator=MaximumLikelihoodEstimator)


def print_cpds(model) -> None:
    print("\n=== CPD TABLES ===\n")

    for cpd in model.get_cpds():
        print(f"CPD for: {cpd.variable}")

        if not cpd.variables[1:]:
            states = cpd.state_names[cpd.variable]
            probs = cpd.values

            df = pd.DataFrame({
                cpd.variable: states,
                "probability": [round(float(p), 3) for p in probs]
            })

            print(df.to_string(index=False))
            print("\n" + "-" * 60 + "\n")
            continue

        evidence_vars = cpd.variables[1:]
        evidence_states = [cpd.state_names[var] for var in evidence_vars]

        columns = pd.MultiIndex.from_product(evidence_states, names=evidence_vars)
        index = cpd.state_names[cpd.variable]

        values = cpd.values.reshape(len(index), -1)
        df = pd.DataFrame(values, index=index, columns=columns).round(3)

        print(df.to_string())
        print("\n" + "-" * 60 + "\n")

def visualize_network(model: DiscreteBayesianNetwork, output_path: str) -> None:
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())

    pos = {
        "Air_temp_cat": (-3.2, 2.3),
        "Process_temp_cat": (-0.8, 2.3),
        "Torque_cat": (-3.0, 0.9),
        "Type": (-3.0, -1.2),
        "Speed_cat": (-1.0, -2.3),
        "Machine failure": (0.4, 0.1),
        "Wear_cat": (3.4, -0.7)
    }

    fig, ax = plt.subplots(figsize=(13, 8))

    ellipse_w = 1.9
    ellipse_h = 0.8
    rx = ellipse_w / 2
    ry = ellipse_h / 2

    for node, (x, y) in pos.items():
        ellipse = Ellipse(
            (x, y),
            width=ellipse_w,
            height=ellipse_h,
            facecolor="white",
            edgecolor="black",
            linewidth=1.4,
            zorder=2
        )
        ax.add_patch(ellipse)

    def shrink_point(p1, p2, rx, ry):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if dx == 0 and dy == 0:
            return p2

        t = 1 / max(abs(dx)/rx if dx != 0 else 0, abs(dy)/ry if dy != 0 else 0)
        return (p2[0] - dx * t, p2[1] - dy * t)

    for u, v in graph.edges():
        p1 = pos[u]
        p2 = pos[v]

        end = shrink_point(p1, p2, rx, ry)

        arrow = FancyArrowPatch(
            p1,
            end,
            arrowstyle='-|>',
            mutation_scale=18,
            linewidth=1.4,
            color="black",
            connectionstyle="arc3,rad=0.05",
            zorder=1
        )
        ax.add_patch(arrow)

    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=10,
            zorder=3
        )

    ax.set_title("Bayesian Network Structure", fontsize=14)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.0, 3.2)
    ax.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

def run_inference(model: DiscreteBayesianNetwork) -> None:
    infer = VariableElimination(model)

    queries = {
        "Q1: Base probability of machine failure": {
            "variables": ["Machine failure"],
            "evidence": None
        },
        "Q2: High wear and high torque": {
            "variables": ["Machine failure"],
            "evidence": {"Wear_cat": "high", "Torque_cat": "high"}
        },
        "Q3: Low wear and low torque": {
            "variables": ["Machine failure"],
            "evidence": {"Wear_cat": "low", "Torque_cat": "low"}
        },
        "Q4: High process temperature, high wear, high speed": {
            "variables": ["Machine failure"],
            "evidence": {
                "Process_temp_cat": "high",
                "Wear_cat": "high",
                "Speed_cat": "high"
            }
        },
        "Q5: Type H, high torque, high wear": {
            "variables": ["Machine failure"],
            "evidence": {
                "Type": "H",
                "Torque_cat": "high",
                "Wear_cat": "high"
            }
        }
    }

    print("\n=== INFERENCE RESULTS ===\n")
    for title, params in queries.items():
        result = infer.query(
            variables=params["variables"],
            evidence=params["evidence"]
        )
        print(title)
        print(result)
        print()

def main() -> None:
    bn_df = load_and_preprocess_data(DATA_PATH)

    model = build_model()
    fit_model(model, bn_df)

    print("\n=== MODEL EDGES ===")
    print(list(model.edges()))

    print_cpds(model)
    run_inference(model)
    visualize_network(model, GRAPH_PATH)

    print(f"\nGraph saved to: {GRAPH_PATH}")

if __name__ == "__main__":
    main()