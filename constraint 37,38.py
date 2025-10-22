from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraints 37 & 38: At most one depot departure and at most one depot return (per tandem)
#-----------------------------------------------------------------------------------------

def run_case_depot(departures, returns):
    model = cp_model.CpModel()

    # Sets
    K = [0]        # single tandem
    depot = 0
    C = [1, 2]     # two customer nodes (ordered list for index alignment)

    # Decision variables: depot->customer and customer->depot arcs
    x = {}
    for j in C:
        x[(0, depot, j)] = model.NewBoolVar(f"x_0_{depot}_{j}")   # departure arcs
    for i in C:
        x[(0, i, depot)] = model.NewBoolVar(f"x_0_{i}_{depot}")   # return arcs

    # Fix scenario values (departures, returns are lists aligned with C order)
    for idx, j in enumerate(C):
        model.Add(x[(0, depot, j)] == departures[idx])
    for idx, i in enumerate(C):
        model.Add(x[(0, i, depot)] == returns[idx])

    # Constraint 37: Σ_j x_{k,0,j} ≤ 1
    for k in K:
        model.Add(sum(x[k, depot, j] for j in C) <= 1)

    # Constraint 38: Σ_i x_{k,i,0} ≤ 1
    for k in K:
        model.Add(sum(x[k, i, depot] for i in C) <= 1)

    # Solve
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Functions
#-----------------------------------------------------------------------------------------
def test_depot_none():
    assert run_case_depot([0, 0], [0, 0]) == cp_model.OPTIMAL

def test_depot_one_depart_one_return():
    assert run_case_depot([1, 0], [0, 1]) == cp_model.OPTIMAL

def test_depot_one_depart_only():
    assert run_case_depot([0, 1], [0, 0]) == cp_model.OPTIMAL

def test_depot_one_return_only():
    assert run_case_depot([0, 0], [1, 0]) == cp_model.OPTIMAL

def test_depot_two_departures():
    assert run_case_depot([1, 1], [0, 0]) == cp_model.INFEASIBLE

def test_depot_two_returns():
    assert run_case_depot([0, 0], [1, 1]) == cp_model.INFEASIBLE

def test_depot_two_depart_one_return():
    assert run_case_depot([1, 1], [0, 1]) == cp_model.INFEASIBLE

def test_depot_one_depart_two_returns():
    assert run_case_depot([1, 0], [1, 1]) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("none", [0, 0], [0, 0]),
        ("one depart & one return", [1, 0], [0, 1]),
        ("one depart only", [0, 1], [0, 0]),
        ("one return only", [0, 0], [1, 0]),
        ("two departures", [1, 1], [0, 0]),
        ("two returns", [0, 0], [1, 1]),
        ("two dep / one ret", [1, 1], [0, 1]),
        ("one dep / two ret", [1, 0], [1, 1]),
    ]
    for label, dep, ret in scenarios:
        status = run_case_depot(dep, ret)
        print(f"{label}: status={cp_model.OPTIMAL if status == cp_model.OPTIMAL else status}")