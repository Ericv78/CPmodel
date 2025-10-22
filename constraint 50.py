from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 50: Drone rendezvous support
#-----------------------------------------------------------------------------------------

def build_model_50():
    model = cp_model.CpModel()

    # Sets
    K  = [0]          # single tandem
    depot = 0
    VL = [0, 1, 2]    # potential truck nodes (including depot)
    C  = [2]          # single customer j=2
    VR = [3]          # rendezvous node l=3

    # Truck arc variables entering l
    x = {
        (0, 1, 3): model.NewBoolVar("x_0_1_3"),
        (0, 2, 3): model.NewBoolVar("x_0_2_3"),
    }

    # Drone sortie variable (launch=0, serve=2, rendezvous=3)
    y = {
        (0, 0, 2, 3): model.NewBoolVar("y_0_0_2_3")
    }

    # Constraint 50
    for k in K:
        for j in C:
            for l in VR:
                key = (k, depot, j, l)
                if key in y and j != l:
                    rhs = sum(
                        x[(k, i, l)]
                        for i in VL
                        if i != j and i != l and (k, i, l) in x
                    )
                    model.Add(y[key] <= rhs)

    return model, x, y

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_50(assign_x, assign_y):
    model, x, y = build_model_50()
    for key, val in assign_x.items():
        model.Add(x[key] == val)
    for key, val in assign_y.items():
        model.Add(y[key] == val)
    model.Maximize(sum(y.values()))  # dummy objective
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Cases
#-----------------------------------------------------------------------------------------
def test_50_valid_one_truck_support():
    # One truck arc into l, drone returns => feasible
    assert run_50({(0,1,3):1, (0,2,3):0}, {(0,0,2,3):1}) == cp_model.OPTIMAL

def test_50_valid_two_truck_support():
    # Both arcs present, drone returns => feasible
    assert run_50({(0,1,3):1, (0,2,3):1}, {(0,0,2,3):1}) == cp_model.OPTIMAL

def test_50_trivial_y_zero():
    # No truck support needed when drone not used
    assert run_50({(0,1,3):0, (0,2,3):0}, {(0,0,2,3):0}) == cp_model.OPTIMAL

def test_50_infeasible_no_truck_support():
    # Drone returns but no truck arc into rendezvous
    assert run_50({(0,1,3):0, (0,2,3):0}, {(0,0,2,3):1}) == cp_model.INFEASIBLE

def test_50_infeasible_wrong_support_removed():
    # Force drone but remove needed arc present in other scenario (still no support)
    assert run_50({(0,1,3):0, (0,2,3):1}, {(0,0,2,3):1}) == cp_model.INFEASIBLE  # i=2 excluded by i!=j (j=2)

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("valid one support", {(0,1,3):1,(0,2,3):0}, {(0,0,2,3):1}),
        ("valid two support", {(0,1,3):1,(0,2,3):1}, {(0,0,2,3):1}),
        ("trivial y=0",       {(0,1,3):0,(0,2,3):0}, {(0,0,2,3):0}),
        ("infeasible none",   {(0,1,3):0,(0,2,3):0}, {(0,0,2,3):1}),
        ("infeasible wrong",  {(0,1,3):0,(0,2,3):1}, {(0,0,2,3):1}),
    ]
    for label, ax, ay in scenarios:
        status = run_50(ax, ay)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")