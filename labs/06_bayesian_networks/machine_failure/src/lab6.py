import os
import pandas as pd
from tabulate import tabulate
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
GRAPH_PATH = os.path.join(OUTPUT_DIR, "bayesian_network_structure_machine_failure.png")
TARGET_COLUMN = "Machine failure"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

def print_block(title: str) -> None:
    print("\n" + "=" * 90)
    print(title.upper())
    print("=" * 90)

def print_table(df: pd.DataFrame, tablefmt: str = "fancy_grid") -> None:
    if df.empty:
        print("[empty table]")
        return
    print(tabulate(df, headers="keys", tablefmt=tablefmt, showindex=False))

def inspect_source_table(df: pd.DataFrame) -> None:
    print_block("dataset passport")

    overview = pd.DataFrame([
        ["Rows", df.shape[0]],
        ["Columns", df.shape[1]],
        ["Target column", TARGET_COLUMN if TARGET_COLUMN in df.columns else "not found"]
    ], columns=["Attribute", "Value"])

    print_table(overview)

    print_block("column registry")

    col_df = pd.DataFrame({
        "No": range(1, len(df.columns) + 1),
        "Column": df.columns,
        "Dtype": [str(dtype) for dtype in df.dtypes]
    })

    print_table(col_df)

    print_block("missing values audit")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isnull().sum().values,
        "Missing %": ((df.isnull().sum() / len(df)) * 100).round(2).values
    }).sort_values(by=["Missing", "Column"], ascending=[False, True])

    print_table(missing_df)

    if TARGET_COLUMN in df.columns:
        print_block("target balance")

        target_counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
        target_share = (df[TARGET_COLUMN].value_counts(dropna=False, normalize=True).sort_index() * 100).round(2)

        target_df = pd.DataFrame({
            "Target state": target_counts.index.astype(str),
            "Count": target_counts.values,
            "Percent": target_share.values
        })

        print_table(target_df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    print_block("numeric profile")

    if numeric_cols:
        numeric_df = df[numeric_cols].describe().T.reset_index().rename(columns={"index": "Feature"})
        print_table(numeric_df.round(3))
    else:
        print("No numeric columns found.")

    print_block("raw sample")
    print_table(df.head(10).astype(str), tablefmt="grid")

def prepare_bn_dataset(csv_path: str) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)

    inspect_source_table(raw_df)

    selected_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]

    work_df = raw_df[selected_columns].copy()
    work_df = work_df.dropna()

    work_df["Air_temp_cat"] = pd.qcut(
        work_df["Air temperature [K]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    work_df["Process_temp_cat"] = pd.qcut(
        work_df["Process temperature [K]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    work_df["Speed_cat"] = pd.qcut(
        work_df["Rotational speed [rpm]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    work_df["Torque_cat"] = pd.qcut(
        work_df["Torque [Nm]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    work_df["Wear_cat"] = pd.qcut(
        work_df["Tool wear [min]"],
        3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    bn_ready_df = work_df[
        [
            "Type",
            "Air_temp_cat",
            "Process_temp_cat",
            "Speed_cat",
            "Torque_cat",
            "Wear_cat",
            "Machine failure"
        ]
    ].copy()

    bn_ready_df["Type"] = bn_ready_df["Type"].astype(str).str.strip()
    bn_ready_df["Air_temp_cat"] = bn_ready_df["Air_temp_cat"].astype(str).str.strip().str.lower()
    bn_ready_df["Process_temp_cat"] = bn_ready_df["Process_temp_cat"].astype(str).str.strip().str.lower()
    bn_ready_df["Speed_cat"] = bn_ready_df["Speed_cat"].astype(str).str.strip().str.lower()
    bn_ready_df["Torque_cat"] = bn_ready_df["Torque_cat"].astype(str).str.strip().str.lower()
    bn_ready_df["Wear_cat"] = bn_ready_df["Wear_cat"].astype(str).str.strip().str.lower()
    bn_ready_df["Machine failure"] = bn_ready_df["Machine failure"].map({0: "no", 1: "yes"}).astype(str)

    print_block("prepared dataset preview")
    print_table(bn_ready_df.head(12).astype(str), tablefmt="grid")

    print_block("prepared dataset schema")

    schema_df = pd.DataFrame({
        "Column": bn_ready_df.columns,
        "Type": [str(dtype) for dtype in bn_ready_df.dtypes]
    })

    print_table(schema_df)

    print_block("state inventory")

    for column in bn_ready_df.columns:
        states_df = pd.DataFrame({
            "State": bn_ready_df[column].value_counts().index.astype(str),
            "Count": bn_ready_df[column].value_counts().values,
            "Percent": (bn_ready_df[column].value_counts(normalize=True).values * 100).round(2)
        })

        print(f"\n[{column}]")
        print_table(states_df, tablefmt="rounded_grid")

    return bn_ready_df

def assemble_dependency_graph() -> DiscreteBayesianNetwork:
    return DiscreteBayesianNetwork(
        [
            ("Type", "Torque_cat"),
            ("Type", "Speed_cat"),
            ("Air_temp_cat", "Process_temp_cat"),
            ("Torque_cat", "Machine failure"),
            ("Wear_cat", "Machine failure"),
            ("Speed_cat", "Machine failure"),
            ("Process_temp_cat", "Machine failure")
        ]
    )

def estimate_parameters(network: DiscreteBayesianNetwork, dataset: pd.DataFrame) -> None:
    network.fit(dataset, estimator=MaximumLikelihoodEstimator)

def render_probability_tables(network: DiscreteBayesianNetwork) -> None:
    print_block("conditional probability tables")

    for cpd in network.get_cpds():
        print(f"\nNode: {cpd.variable}")

        if len(cpd.variables) == 1:
            base_df = pd.DataFrame({
                "State": cpd.state_names[cpd.variable],
                "Probability": [round(float(x), 4) for x in cpd.values]
            })

            print_table(base_df, tablefmt="rounded_grid")
            continue

        evidence_vars = cpd.variables[1:]
        evidence_states = [cpd.state_names[var] for var in evidence_vars]
        grid_columns = pd.MultiIndex.from_product(evidence_states, names=evidence_vars)
        state_index = cpd.state_names[cpd.variable]

        cpd_df = pd.DataFrame(
            cpd.values.reshape(len(state_index), -1),
            index=state_index,
            columns=grid_columns
        ).round(4)

        cpd_df = cpd_df.reset_index().rename(columns={"index": cpd.variable})

        print(tabulate(cpd_df, headers="keys", tablefmt="fancy_grid", showindex=False))

def visualize_network(model: DiscreteBayesianNetwork, output_path: str) -> None:
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())

    pos = {
        "Air_temp_cat": (-3.6, 2.6),
        "Process_temp_cat": (-2.2, 0.9),
        "Type": (2.6, 2.6),
        "Torque_cat": (0.8, 0.9),
        "Speed_cat": (4.0, 0.9),
        "Wear_cat": (-3.4, -1.0),
        "Machine failure": (0.4, -2.2)
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    node_width = 2.45
    node_height = 0.78

    for node, (x, y) in pos.items():
        ellipse = Ellipse(
            (x, y),
            width=node_width,
            height=node_height,
            facecolor="#eef3f8",
            edgecolor="#1f2933",
            linewidth=1.8,
            zorder=2
        )
        ax.add_patch(ellipse)

    def get_boundary_point(start, end, width, height):
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if dx == 0 and dy == 0:
            return start

        rx = width / 2
        ry = height / 2

        scale = 1 / max(abs(dx) / rx if dx != 0 else 0, abs(dy) / ry if dy != 0 else 0)

        return (start[0] + dx * scale, start[1] + dy * scale)

    for u, v in graph.edges():
        start_center = pos[u]
        end_center = pos[v]

        start = get_boundary_point(start_center, end_center, node_width, node_height)
        end = get_boundary_point(end_center, start_center, node_width, node_height)

        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=1.7,
            color="#1f2933",
            shrinkA=0,
            shrinkB=0,
            zorder=1
        )

        ax.add_patch(arrow)

    labels = {
        "Type": "Type",
        "Air_temp_cat": "Air Temperature",
        "Process_temp_cat": "Process\nTemperature",
        "Speed_cat": "Rotational\nSpeed",
        "Torque_cat": "Torque",
        "Wear_cat": "Tool Wear",
        "Machine failure": "Machine\nFailure"
    }

    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            labels.get(node, node),
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
            zorder=3
        )

    ax.set_title(
        "Bayesian Network for Machine Failure Analysis",
        fontsize=16,
        fontweight="bold",
        pad=22,
        color="#111827"
    )

    ax.set_xlim(-5.2, 5.5)
    ax.set_ylim(-3.0, 3.4)
    ax.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

def execute_probabilistic_queries(network: DiscreteBayesianNetwork) -> None:
    solver = VariableElimination(network)

    scenario_bank = [
        {
            "Scenario": "Baseline probability",
            "Query variable": "Machine failure",
            "Evidence": "-",
            "Result": solver.query(variables=["Machine failure"])
        },
        {
            "Scenario": "High wear and high torque",
            "Query variable": "Machine failure",
            "Evidence": "wear=high, torque=high",
            "Result": solver.query(
                variables=["Machine failure"],
                evidence={
                    "Wear_cat": "high",
                    "Torque_cat": "high"
                }
            )
        },
        {
            "Scenario": "Low wear and low torque",
            "Query variable": "Machine failure",
            "Evidence": "wear=low, torque=low",
            "Result": solver.query(
                variables=["Machine failure"],
                evidence={
                    "Wear_cat": "low",
                    "Torque_cat": "low"
                }
            )
        },
        {
            "Scenario": "High process temperature, high wear, high speed",
            "Query variable": "Machine failure",
            "Evidence": "process temperature=high, wear=high, speed=high",
            "Result": solver.query(
                variables=["Machine failure"],
                evidence={
                    "Process_temp_cat": "high",
                    "Wear_cat": "high",
                    "Speed_cat": "high"
                }
            )
        },
        {
            "Scenario": "Type H, high torque, high wear",
            "Query variable": "Machine failure",
            "Evidence": "type=H, torque=high, wear=high",
            "Result": solver.query(
                variables=["Machine failure"],
                evidence={
                    "Type": "H",
                    "Torque_cat": "high",
                    "Wear_cat": "high"
                }
            )
        },
        {
            "Scenario": "Process temperature under high air temperature",
            "Query variable": "Process_temp_cat",
            "Evidence": "air temperature=high",
            "Result": solver.query(
                variables=["Process_temp_cat"],
                evidence={
                    "Air_temp_cat": "high"
                }
            )
        }
    ]

    print_block("inference report")

    summary_rows = []

    for case in scenario_bank:
        factor = case["Result"]
        variable = case["Query variable"]
        states = factor.state_names[variable]
        probs = [round(float(x), 4) for x in factor.values]
        dominant_idx = probs.index(max(probs))

        summary_rows.append([
            case["Scenario"],
            variable,
            case["Evidence"],
            states[dominant_idx],
            probs[dominant_idx]
        ])

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["Scenario", "Variable", "Evidence", "Most probable state", "Probability"]
    )

    print_table(summary_df)

    print_block("detailed inference tables")

    for case in scenario_bank:
        factor = case["Result"]
        variable = case["Query variable"]
        states = factor.state_names[variable]
        probs = [round(float(x), 4) for x in factor.values]

        print(f"\nScenario: {case['Scenario']}")
        print(f"Variable: {variable}")
        print(f"Evidence: {case['Evidence']}")

        detail_df = pd.DataFrame({
            "State": states,
            "Probability": probs
        })

        print_table(detail_df, tablefmt="rounded_grid")

def orchestrate_pipeline() -> None:
    dataset_for_bn = prepare_bn_dataset(DATA_PATH)

    bayes_net = assemble_dependency_graph()
    estimate_parameters(bayes_net, dataset_for_bn)

    print_block("network arcs")

    edges_df = pd.DataFrame(bayes_net.edges(), columns=["From", "To"])
    print_table(edges_df, tablefmt="rounded_grid")

    render_probability_tables(bayes_net)
    execute_probabilistic_queries(bayes_net)
    visualize_network(bayes_net, GRAPH_PATH)

    print_block("execution status")

    status_df = pd.DataFrame([
        ["Graph image", GRAPH_PATH],
        ["Target node", TARGET_COLUMN],
        ["Rows used", len(dataset_for_bn)],
        ["Nodes in network", len(bayes_net.nodes())],
        ["Edges in network", len(bayes_net.edges())]
    ], columns=["Item", "Value"])

    print_table(status_df, tablefmt="rounded_grid")

if __name__ == "__main__":
    orchestrate_pipeline()