from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 46: Valid drone sortie (launch i, serve j, rendezvous l)
# Enforce: j in VD, and i, j, l all distinct
#-----------------------------------------------------------------------------------------

def build_model():
    model = cp_model.CpModel()

    # Sets
    K   = [0]                # single tandem
    depot = 0
    VL  = [0, 1, 2, 3]       # launch candidates (include depot)
    C   = [1, 2, 4]          # customer nodes
    VR  = [3]                # rendezvous nodes
    VD  = {2, 4}             # drone‑eligible customers

    # Declare selected y_drone variables (only test subset)
    y_drone = {
        (0, 0, 2, 3): model.NewBoolVar("y_0_0_2_3"),  # valid
        (0, 0, 1, 3): model.NewBoolVar("y_0_0_1_3"),  # invalid (j not in VD)
        (0, 1, 1, 3): model.NewBoolVar("y_0_1_1_3"),  # invalid (i == j)
        (0, 3, 2, 3): model.NewBoolVar("y_0_3_2_3"),  # invalid (i == l)
        (0, 2, 4, 3): model.NewBoolVar("y_0_2_4_3"),  # valid
    }

    #Constraint 46
    for k in K:
        for i in VL:
            for j in C:
                for l in VR:
                    if (
                        i == j or i == l or
                        j == i or j == l or j not in VD or
                        l == i or l == j
                    ):
                        if (k,i,j,l) in y_drone:
                            model.Add(y_drone[k,i,j,l] == 0)

    return model, y_drone

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_case(assignments):
    model, y_drone = build_model()
    for key, val in assignments.items():
        if key not in y_drone:
            raise KeyError(f"Undeclared variable {key}")
        model.Add(y_drone[key] == val)
    model.Maximize(sum(y_drone.values()))
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Cases
#-----------------------------------------------------------------------------------------
def test_c46_valid_case1():
    assert run_case({(0, 0, 2, 3): 1}) == cp_model.OPTIMAL

def test_c46_valid_case2():
    assert run_case({(0, 2, 4, 3): 1}) == cp_model.OPTIMAL

def test_c46_invalid_j_not_in_VD():
    assert run_case({(0, 0, 1, 3): 1}) == cp_model.INFEASIBLE

def test_c46_invalid_i_equals_j():
    assert run_case({(0, 1, 1, 3): 1}) == cp_model.INFEASIBLE

def test_c46_invalid_i_equals_l():
    assert run_case({(0, 3, 2, 3): 1}) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("valid: j=2 in VD, distinct i,l", {(0, 0, 2, 3): 1}),
        ("valid: j=4 in VD, distinct i,l", {(0, 2, 4, 3): 1}),
        ("invalid: j=1 not in VD", {(0, 0, 1, 3): 1}),
        ("invalid: i == j", {(0, 1, 1, 3): 1}),
        ("invalid: i == l", {(0, 3, 2, 3): 1}),
    ]
    for label, assigns in scenarios:
        status = run_case(assigns)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")
