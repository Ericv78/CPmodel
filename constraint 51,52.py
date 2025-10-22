from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraints 51 & 52: Depot arrival times
#-----------------------------------------------------------------------------------------

def build_model_51_52():
    model = cp_model.CpModel()
    K = [0]
    depot = 0

    # Arrival time variables
    a       = {(0, depot): model.NewIntVar(0, 100, "a_0_0")}
    a_prime = {(0, depot): model.NewIntVar(0, 100, "a_prime_0_0")}

    # Apply constraints 51 & 52
    for k in K:
        model.Add(a[k, depot] == 0)
        model.Add(a_prime[k, depot] == 0)

    return model, a, a_prime

#-----------------------------------------------------------------------------------------
# Runner (optional forced values to test feasibility)
#-----------------------------------------------------------------------------------------
def run_case_51_52(force_a=None, force_ap=None):
    model, a, a_prime = build_model_51_52()
    if force_a is not None:
        model.Add(a[0, 0] == force_a)
    if force_ap is not None:
        model.Add(a_prime[0, 0] == force_ap)
    model.Minimize(0)
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Cases
#-----------------------------------------------------------------------------------------
def test_51_52_valid_zero():
    assert run_case_51_52(force_a=0, force_ap=0) == cp_model.OPTIMAL

def test_51_valid_only():
    assert run_case_51_52(force_a=0, force_ap=0) == cp_model.OPTIMAL

def test_52_valid_only():
    assert run_case_51_52(force_a=0, force_ap=0) == cp_model.OPTIMAL

def test_51_infeasible_nonzero_truck():
    assert run_case_51_52(force_a=5, force_ap=0) == cp_model.INFEASIBLE

def test_52_infeasible_nonzero_drone():
    assert run_case_51_52(force_a=0, force_ap=7) == cp_model.INFEASIBLE

def test_51_52_infeasible_both_nonzero():
    assert run_case_51_52(force_a=3, force_ap=4) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("valid both zero",            dict(force_a=0, force_ap=0)),
        ("valid (51) truck zero",      dict(force_a=0, force_ap=0)),  # same as above
        ("valid (52) drone zero",      dict(force_a=0, force_ap=0)),  # same as above
        ("infeasible truck nonzero",   dict(force_a=5, force_ap=0)),
        ("infeasible drone nonzero",   dict(force_a=0, force_ap=7)),
        ("infeasible both nonzero",    dict(force_a=3, force_ap=4)),
    ]
    for label, args in scenarios:
        status = run_case_51_52(**args)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")