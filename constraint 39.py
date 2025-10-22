from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 39: Flow conservation at an intermediate node (incoming = outgoing)
#-----------------------------------------------------------------------------------------

def run_case_flow(incoming_active, outgoing_active):
    model = cp_model.CpModel()

    # Sets
    K = [0]
    test_node = 1
    incoming_origins = [0, 2]   # possible predecessors
    outgoing_destinations = [0, 2]  # possible successors

    # Decision variables
    x = {}
    # Incoming arcs into test_node
    for idx, origin in enumerate(incoming_origins):
        x[(0, origin, test_node)] = model.NewBoolVar(f"x_0_{origin}_{test_node}")
        model.Add(x[(0, origin, test_node)] == incoming_active[idx])
    # Outgoing arcs from test_node
    for idx, dest in enumerate(outgoing_destinations):
        x[(0, test_node, dest)] = model.NewBoolVar(f"x_0_{test_node}_{dest}")
        model.Add(x[(0, test_node, dest)] == outgoing_active[idx])

    # Constraint 39: sum incoming == sum outgoing
    for k in K:
        incoming = sum(x[(k, origin, test_node)] for origin in incoming_origins)
        outgoing = sum(x[(k, test_node, dest)] for dest in outgoing_destinations)
        model.Add(incoming == outgoing)

    # Solve
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Functions
#-----------------------------------------------------------------------------------------
def test_flow_balanced_single():
    assert run_case_flow([1, 0], [1, 0]) == cp_model.OPTIMAL

def test_flow_balanced_double():
    assert run_case_flow([1, 1], [1, 1]) == cp_model.OPTIMAL

def test_flow_none():
    assert run_case_flow([0, 0], [0, 0]) == cp_model.OPTIMAL

def test_flow_in_only_one():
    assert run_case_flow([1, 0], [0, 0]) == cp_model.INFEASIBLE

def test_flow_out_only_one():
    assert run_case_flow([0, 0], [1, 0]) == cp_model.INFEASIBLE

def test_flow_unbalanced_more_in():
    assert run_case_flow([1, 1], [1, 0]) == cp_model.INFEASIBLE

def test_flow_unbalanced_more_out():
    assert run_case_flow([1, 0], [1, 1]) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("balanced single", [1, 0], [1, 0]),
        ("balanced double", [1, 1], [1, 1]),
        ("none", [0, 0], [0, 0]),
        ("incoming only", [1, 0], [0, 0]),
        ("outgoing only", [0, 0], [1, 0]),
        ("more incoming", [1, 1], [1, 0]),
        ("more outgoing", [1, 0], [1, 1]),
    ]
    for label, inc, out in scenarios:
        status = run_case_flow(inc, out)
        print(f"{label}: status={'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")