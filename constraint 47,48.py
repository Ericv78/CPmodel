from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraints 47 & 48 
#-----------------------------------------------------------------------------------------

def build_model_4748():
    model = cp_model.CpModel()

    # Sets (miniature)
    K  = [0]            # single tandem
    VL = [0, 1]         # launch candidates (depot 0, node 1)
    C  = [2, 3]         # customer nodes
    VR = [4, 5]         # rendezvous nodes

    # Drone sortie variables y[k,i,j,l]
    y = {
        (0, 0, 2, 4): model.NewBoolVar("y_0_0_2_4"),
        (0, 0, 3, 5): model.NewBoolVar("y_0_0_3_5"),
        (0, 1, 2, 4): model.NewBoolVar("y_0_1_2_4"),
        (0, 1, 3, 5): model.NewBoolVar("y_0_1_3_5"),
    }

    #Constraint 47
    for k in K:
        for i in VL:
            launch_trips = []
            for j in C:
                if j != i:
                    for l in VR:
                        if l != i and l != j and (k,i,j,l) in y:
                            launch_trips.append(y[k,i,j,l])
            if launch_trips:
                model.Add(sum(launch_trips) <= 1)

    #Constraint 48
    for k in K:
        for l in VR:
            rendezvous_trips = []
            for i in VL:
                if i != l:
                    for j in C:
                        if j != i and j != l and (k,i,j,l) in y:
                            rendezvous_trips.append(y[k,i,j,l])
            if rendezvous_trips:
                model.Add(sum(rendezvous_trips) <= 1)

    return model, y

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_4748(assignments):
    model, y = build_model_4748()
    for key, val in assignments.items():
        model.Add(y[key] == val)
    model.Maximize(sum(y.values()))  # dummy objective
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Tests for Constraint 47 (launch)
#-----------------------------------------------------------------------------------------
def test_47_valid_one_launch():
    assert run_4748({(0, 0, 2, 4): 1, (0, 0, 3, 5): 0}) == cp_model.OPTIMAL

def test_47_infeasible_two_launches_same_i():
    assert run_4748({(0, 0, 2, 4): 1, (0, 0, 3, 5): 1}) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Tests for Constraint 48 (rendezvous)
#-----------------------------------------------------------------------------------------
def test_48_valid_one_rendezvous():
    assert run_4748({(0, 1, 2, 4): 1, (0, 0, 3, 5): 0}) == cp_model.OPTIMAL

def test_48_infeasible_two_rendezvous_same_l():
    assert run_4748({(0, 1, 2, 4): 1, (0, 0, 2, 4): 1}) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("47 valid: one launch i=0", {(0, 0, 2, 4): 1, (0, 0, 3, 5): 0}),
        ("47 infeasible: two launches i=0", {(0, 0, 2, 4): 1, (0, 0, 3, 5): 1}),
        ("48 valid: one rendezvous l=4", {(0, 1, 2, 4): 1, (0, 0, 3, 5): 0}),
        ("48 infeasible: two rendezvous l=4", {(0, 1, 2, 4): 1, (0, 0, 2, 4): 1}),
    ]
    for label, assigns in scenarios:
        status = run_4748(assigns)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")
