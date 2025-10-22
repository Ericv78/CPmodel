from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 53: Depot arrival time bounded by horizon (a_{k,0} ≤ T)
#-----------------------------------------------------------------------------------------

def build_model_53(T):
    model = cp_model.CpModel()
    K = [0]
    depot = 0
    # Arrival time at depot for truck k
    a = {(0, depot): model.NewIntVar(0, 1000, "a_0_depot")}
    
    # Apply constraint 53
    for k in K:
        model.Add(a[k, depot] <= T)
    return model, a

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_case_53(T, arrival_val):
    model, a = build_model_53(T)
    model.Add(a[0, 0] == arrival_val)  # force scenario value
    model.Minimize(0)
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# test
#-----------------------------------------------------------------------------------------
def test_53_inside_horizon():
    assert run_case_53(10, 8) == cp_model.OPTIMAL

def test_53_boundary():
    assert run_case_53(10, 10) == cp_model.OPTIMAL

def test_53_zero_horizon_zero_arrival():
    assert run_case_53(0, 0) == cp_model.OPTIMAL

def test_53_exceeds_horizon():
    assert run_case_53(10, 12) == cp_model.INFEASIBLE

def test_53_zero_horizon_positive_arrival():
    assert run_case_53(0, 1) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("valid (8 ≤ 10)", 10, 8),
        ("boundary (10 ≤ 10)", 10, 10),
        ("infeasible (12 > 10)", 10, 12),
        ("zero horizon valid", 0, 0),
        ("zero horizon infeasible", 0, 1),
    ]
    for label, T, val in scenarios:
        status = run_case_53(T, val)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")