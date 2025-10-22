from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 40: Trucks cannot traverse between drone-only (damaged) nodes
# note: unlike the literature definition, VT is a set of node that trucks cannot traverse to.
# this is reflected in the constraint implentation below using VD.
#-----------------------------------------------------------------------------------------

def run_case_forbidden(truck_move_12, truck_move_21):
    model = cp_model.CpModel()

    # Sets
    K = [0]          # single tandem
    VD = {1, 2}      # drone-only(forbidden for truck traversal)

    # Decision variables: truck arcs between the two damaged nodes
    x = {
        (0, 1, 2): model.NewBoolVar("x_0_1_2"),
        (0, 2, 1): model.NewBoolVar("x_0_2_1"),
    }

    # Fix scenario values
    model.Add(x[0, 1, 2] == truck_move_12)
    model.Add(x[0, 2, 1] == truck_move_21)

    # Constraint 40: forbid truck arcs between drone-only nodes
    for k in K:
        for i in VD:
            for j in VD:
                if i != j:
                    model.Add(x[k, i, j] == 0)

    # Solve
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Functions
#-----------------------------------------------------------------------------------------
def test_constraint40_none():
    assert run_case_forbidden(0, 0) == cp_model.OPTIMAL

def test_constraint40_illegal_12():
    assert run_case_forbidden(1, 0) == cp_model.INFEASIBLE

def test_constraint40_illegal_21():
    assert run_case_forbidden(0, 1) == cp_model.INFEASIBLE

def test_constraint40_illegal_both():
    assert run_case_forbidden(1, 1) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("none", 0, 0),
        ("illegal 1->2", 1, 0),
        ("illegal 2->1", 0, 1),
        ("illegal both", 1, 1),
    ]
    for label, m12, m21 in scenarios:
        status = run_case_forbidden(m12, m21)
        print(f"{label}: status={'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")