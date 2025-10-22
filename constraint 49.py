from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 49: Drone sortie activation requires truck support (launch i and rendezvous l)
#-----------------------------------------------------------------------------------------

def build_model_49():
    model = cp_model.CpModel()

    K = [0]
    depot = 0
    VT = [1, 2]        # truck nodes
    VR = [3]           # rendezvous node
    VD = [2]           # drone‑servable customer (j = 2)
    num_nodes = 4      # indices: 0..3

    # Truck arcs subset
    x = {
        (0, 1, 2): model.NewBoolVar("x_0_1_2"),
        (0, 1, 3): model.NewBoolVar("x_0_1_3"),
        (0, 2, 3): model.NewBoolVar("x_0_2_3"),
    }

    # Drone sortie y(k,i,j,l)
    y = {
        (0, 1, 2, 3): model.NewBoolVar("y_0_1_2_3")
    }

    # Constraint 49 (fixed indentation and scope)
    for k in K:
        for i in VT + [depot]:
            for j in VD:
                for l in VR:
                    if (k, i, j, l) not in y:
                        continue
                    if i == j or i == l or j == l:
                        model.Add(y[k, i, j, l] == 0)
                        continue
                    y_var = y[k, i, j, l]
                    leave_i = sum(x[k, i, m] for m in range(num_nodes) if m != i and (k, i, m) in x)
                    enter_l = sum(x[k, m, l] for m in range(num_nodes) if m != l and (k, m, l) in x)
                    model.Add(y_var <= leave_i)
                    model.Add(y_var <= enter_l)
                    model.Add(2 * y_var <= leave_i + enter_l)

    return model, x, y

def run_case_49(assign_x, assign_y):
    model, x, y = build_model_49()
    for key, val in assign_x.items():
        model.Add(x[key] == val)
    for key, val in assign_y.items():
        model.Add(y[key] == val)
    model.Maximize(sum(y.values()))
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Cases
#-----------------------------------------------------------------------------------------

def test_49_valid_two_support_arcs():
    assert run_case_49({(0,1,2):1,(0,2,3):1,(0,1,3):0},{(0,1,2,3):1}) == cp_model.OPTIMAL

def test_49_valid_alt_support():
    assert run_case_49({(0,1,2):1,(0,1,3):1,(0,2,3):0},{(0,1,2,3):1}) == cp_model.OPTIMAL

def test_49_trivial_y_zero():
    assert run_case_49({(0,1,2):0,(0,1,3):0,(0,2,3):0},{(0,1,2,3):0}) == cp_model.OPTIMAL

def test_49_infeasible_only_depart():
    assert run_case_49({(0,1,2):1,(0,1,3):0,(0,2,3):0},{(0,1,2,3):1}) == cp_model.INFEASIBLE

def test_49_infeasible_only_arrive():
    assert run_case_49({(0,1,2):0,(0,1,3):0,(0,2,3):1},{(0,1,2,3):1}) == cp_model.INFEASIBLE

def test_49_infeasible_no_support():
    assert run_case_49({(0,1,2):0,(0,1,3):0,(0,2,3):0},{(0,1,2,3):1}) == cp_model.INFEASIBLE

if __name__ == "__main__":
    scenarios = [
        ("valid two support (1->2,2->3)", {(0,1,2):1,(0,2,3):1,(0,1,3):0}, {(0,1,2,3):1}),
        ("valid alt support (1->2,1->3)", {(0,1,2):1,(0,1,3):1,(0,2,3):0}, {(0,1,2,3):1}),
        ("trivial y=0", {(0,1,2):0,(0,1,3):0,(0,2,3):0}, {(0,1,2,3):0}),
        ("infeasible only depart", {(0,1,2):1,(0,1,3):0,(0,2,3):0}, {(0,1,2,3):1}),
        ("infeasible only arrive", {(0,1,2):0,(0,1,3):0,(0,2,3):1}, {(0,1,2,3):1}),
        ("infeasible no support", {(0,1,2):0,(0,1,3):0,(0,2,3):0}, {(0,1,2,3):1}),
    ]
    for label, ax, ay in scenarios:
        status = run_case_49(ax, ay)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")