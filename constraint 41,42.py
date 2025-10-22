from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraints 41 & 42: MTZ Subtour Elimination (41) and Order Activation (42)
#-----------------------------------------------------------------------------------------

def run_case_subtour(arc12, arc21):
    model = cp_model.CpModel()

    # Sets
    K = [0]          # single tandem
    depot = 0
    C = {1, 2}       # customer nodes (MTZ applies here)
    VL = C           # launch nodes inside MTZ scope (excluding depot)
    VR = C           # same for destination set
    M = len(C)       # Big-M = number of customers

    # Decision variables: truck arcs between customers
    # x[k,i,j] = 1 if arc i->j used
    x = {}
    for i in VL:
        for j in VR:
            if i != j:
                x[(0, i, j)] = model.NewBoolVar(f"x_0_{i}_{j}")

    # Order variables: u[k,i] in [0, M]
    # Represent visit position; only meaningful if node has incoming arc
    u = { (0, i): model.NewIntVar(0, M, f"u_0_{i}") for i in C }

    # Fix arc activation according to test scenario
    model.Add(x[(0, 1, 2)] == arc12)
    model.Add(x[(0, 2, 1)] == arc21)

    # Constraint (41): MTZ subtour elimination
    for k in K:
        for i in VL:
            for j in VR:
                if i != j:
                    model.Add(u[k, i] - u[k, j] + 1 <= M * (1 - x[k, i, j]))

    # Constraint (42): activate u only if node has an incoming arc
    for k in K:
        for j in VR:
            incoming = sum(x[k, i, j] for i in VL if i != j)
            model.Add(u[k, j] <= M * incoming)

    # Solve
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Functions
#-----------------------------------------------------------------------------------------
def test_mtz_one_direction():
    # Only 1->2 active: no subtour
    assert run_case_subtour(1, 0) == cp_model.OPTIMAL

def test_mtz_none():
    # No arcs: trivially feasible
    assert run_case_subtour(0, 0) == cp_model.OPTIMAL

def test_mtz_subtour():
    # 1->2 and 2->1 both active forms a cycle; MTZ should forbid
    assert run_case_subtour(1, 1) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("one direction", 1, 0),
        ("none", 0, 0),
        ("subtour cycle", 1, 1),
    ]
    for label, a12, a21 in scenarios:
        status = run_case_subtour(a12, a21)
        print(f"{label}: status={'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")