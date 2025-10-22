from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 36: Each affected area can be visited at most once by either a truck or drone
#-----------------------------------------------------------------------------------------

def run_case(truck_visit, drone_visit):
    model = cp_model.CpModel()

    # Sets
    K = [0]     #Single truck and drone pair
    VL = {0, 1} #Launch nodes
    C = {1}     #Customer nodes
    VR = {2}    #Rendezvous node

    # Decision variables
    x = {(0, 0, 1): model.NewBoolVar("x_0_0_1")}      #x[k,i,j]: truck 0 serves customer 1 from launch node 0
    y = {(0, 0, 1, 2): model.NewBoolVar("y_0_0_1_2")} #y[k,i,j,l]: drone 0 serves customer 1 from launch node 0 to rendezvous 2

    # Fix scenario values
    model.Add(x[0, 0, 1] == truck_visit)
    model.Add(y[0, 0, 1, 2] == drone_visit)

    # Constraint 36
    truck_part = sum(x[k, i, j] for k in K for j in C for i in VL if i != j)
    drone_part = sum(y[k, i, j, l] for k in K for j in C for i in VL if i != j for l in VR if l != i)
    model.Add(truck_part + drone_part <= 1)

    #Solve
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Test Functions
#-----------------------------------------------------------------------------------------
def test_constraint36_truck_only():
    assert run_case(1, 0) == cp_model.OPTIMAL

def test_constraint36_drone_only():
    assert run_case(0, 1) == cp_model.OPTIMAL

def test_constraint36_none():
    assert run_case(0, 0) == cp_model.OPTIMAL

def test_constraint36_both():
    assert run_case(1, 1) == cp_model.INFEASIBLE
    
#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("truck only", 1, 0),
        ("drone only", 0, 1),
        ("none", 0, 0),
        ("both", 1, 1),
    ]
    for label, t_visit, d_visit in scenarios:
        status = run_case(t_visit, d_visit)
        print(f"{label}: status={cp_model.OPTIMAL if status == cp_model.OPTIMAL else status}")