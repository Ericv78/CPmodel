from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 61: Drone endurance (mirrored exactly from full model)
#-----------------------------------------------------------------------------------------

def build_model_61(E, t_prime, y_choice):
    model = cp_model.CpModel()

    # Sets
    K  = [0]          # single tandem
    VL = [0]          # launch candidates (depot only for test)
    C  = [1]          # one customer
    VR = [2]          # one rendezvous

    # Decision variable
    y_drone = {(0,0,1,2): model.NewBoolVar("y_0_0_1_2")}

    # Fix chosen sortie
    model.Add(y_drone[0,0,1,2] == y_choice)

    # --- Auto Big-M ---
    # Pick T as a safe upper bound: sum of all t_prime values + 1
    max_time = max(t_prime[i][j] for i in t_prime for j in t_prime[i])
    T = sum(t_prime[i][j] for i in t_prime for j in t_prime[i]) + max_time + 1

    # Constraint 61
    for k in K:
        for i in VL:
            for j in C:
                for l in VR:
                    if i != j and i != l and j != l and (k,i,j,l) in y_drone:
                        model.Add(
                            t_prime[i][j] + t_prime[j][l]
                            - T * (1 - y_drone[k,i,j,l])
                            <= E
                        )

    # Dummy objective
    model.Minimize(0)
    return model, y_drone

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_case_61(E, t_ij, t_jl, y_val):
    # Build t_prime dictionary
    t_prime = {0:{1:t_ij}, 1:{2:t_jl}}
    model, y_drone = build_model_61(E, t_prime, y_val)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return status

#-----------------------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------------------
def test_61_valid_within_endurance():
    # Flight time = 3+4=7 ≤ E=10
    assert run_case_61(E=10, t_ij=3, t_jl=4, y_val=1) == cp_model.OPTIMAL

def test_61_infeasible_exceeds_endurance():
    # Flight time = 6+7=13 > E=10
    assert run_case_61(E=10, t_ij=6, t_jl=7, y_val=1) == cp_model.INFEASIBLE

def test_61_relaxed_when_y0():
    # y=0 relaxes constraint regardless of times
    assert run_case_61(E=10, t_ij=100, t_jl=200, y_val=0) == cp_model.OPTIMAL

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("valid within endurance", dict(E=10, t_ij=3, t_jl=4, y_val=1)),
        ("infeasible exceeds endurance", dict(E=10, t_ij=6, t_jl=7, y_val=1)),
        ("relaxed when y=0", dict(E=10, t_ij=100, t_jl=200, y_val=0)),
    ]
    for label, args in scenarios:
        status = run_case_61(**args)
        s = "OPTIMAL" if status == cp_model.OPTIMAL else "INFEASIBLE"
        print(f"{label}: status={s}")
