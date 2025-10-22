from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraints 55 & 56: Drone timing with launch, service, rendezvous (mirrored exactly)
#-----------------------------------------------------------------------------------------

def build_model_55_56(T, t_prime):
    model = cp_model.CpModel()

    # Sets (miniature)
    K  = [0]            # single tandem
    VL = [0, 2]         # launch candidates (include depot 0 and node 2)
    C  = [1]            # one customer (service node)
    VR = [2]            # rendezvous nodes (use 2 to allow i!=j and j!=l)
    # Note: with C=[1], VL=[0,2], VR=[2], valid flights are (i in {0,2}, j=1, l=2), i!=j, j!=l, i!=l

    # Time variables
    a = {(0,i): model.NewIntVar(0, T, f"a_0_{i}") for i in set(VL) | set(VR)}
    a_prime = {(0,j): model.NewIntVar(0, T, f"a_prime_0_{j}") for j in C}

    # Drone sortie variables y_drone[k,i,j,l] (declare a subset sufficient for tests)
    y_drone = {
        (0, 0, 1, 2): model.NewBoolVar("y_0_0_1_2"),  # valid: i=0, j=1, l=2
        (0, 2, 1, 2): model.NewBoolVar("y_0_2_1_2"),  # invalid (i==l), will be zeroed by 46 but included for completeness
    }

    # Constraint 55
    for k in K:
        for i in VL:
            for j in C:
                if i == j:
                    continue
                flights_ijl = [y_drone[k,i,j,l]
                               for l in VR
                               if l != i and l != j and (k,i,j,l) in y_drone]
                if flights_ijl:
                    sum_ijl = sum(flights_ijl)
                    model.Add(a[k,i] + t_prime[i][j] - T * (1 - sum_ijl) <= a_prime[k,j])

    # Constraint 56
    for k in K:
        for j in C:
            for l in VR:
                if j == l:
                    continue
                flights_ijl = [y_drone[k,i,j,l]
                               for i in VL
                               if i != j and i != l and (k,i,j,l) in y_drone]
                if flights_ijl:
                    sum_ijl = sum(flights_ijl)
                    model.Add(a_prime[k,j] + t_prime[j][l] - T * (1 - sum_ijl) <= a[k,l])

    return model, a, a_prime, y_drone

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_case_55_56(T, t_prime, y_key, y_val, a_i_val, a_j_prime_val, a_l_val):
    model, a, a_prime, y_drone = build_model_55_56(T, t_prime)

    # Fix flight choice (must be a declared y_drone key)
    if y_key not in y_drone:
        raise KeyError(f"Undeclared flight variable {y_key}")
    model.Add(y_drone[y_key] == y_val)

    # Map indices from the chosen y_key into a, a_prime
    k, i, j, l = y_key

    # Fix times
    model.Add(a[(k, i)] == a_i_val)
    model.Add(a_prime[(k, j)] == a_j_prime_val)
    model.Add(a[(k, l)] == a_l_val)

    # Dummy objective
    model.Minimize(0)

    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------------------
def test_55_56_valid_consistent():
    # Use flight (0,0,1,2). t'_{0,1}=3, t'_{1,2}=4
    # 55: a_0 + 3 ≤ a'_1  -> 2 + 3 ≤ 6  OK
    # 56: a'_1 + 4 ≤ a_2  -> 6 + 4 ≤ 12 OK
    t_prime = {0:{1:3}, 1:{2:4}}
    assert run_case_55_56(T=20, t_prime=t_prime, y_key=(0,0,1,2), y_val=1,
                          a_i_val=2, a_j_prime_val=6, a_l_val=12) == cp_model.OPTIMAL

def test_55_56_valid_relaxed_y0():
    # y=0 relaxes both constraints via -T*(1 - 0) term
    t_prime = {0:{1:3}, 1:{2:4}}
    assert run_case_55_56(T=20, t_prime=t_prime, y_key=(0,0,1,2), y_val=0,
                          a_i_val=7, a_j_prime_val=1, a_l_val=1) == cp_model.OPTIMAL

def test_55_56_valid_boundary():
    # Equalities tight: 55: 0 + 5 ≤ 5, 56: 5 + 5 ≤ 10
    t_prime = {0:{1:5}, 1:{2:5}}
    assert run_case_55_56(T=15, t_prime=t_prime, y_key=(0,0,1,2), y_val=1,
                          a_i_val=0, a_j_prime_val=5, a_l_val=10) == cp_model.OPTIMAL

def test_55_infeasible_service_too_early():
    # 55 violated: 2 + 3 ≤ 4 is false
    t_prime = {0:{1:3}, 1:{2:4}}
    assert run_case_55_56(T=20, t_prime=t_prime, y_key=(0,0,1,2), y_val=1,
                          a_i_val=2, a_j_prime_val=4, a_l_val=12) == cp_model.INFEASIBLE

def test_56_infeasible_rendezvous_too_early():
    # 56 violated: 6 + 4 ≤ 9 is false
    t_prime = {0:{1:3}, 1:{2:4}}
    assert run_case_55_56(T=20, t_prime=t_prime, y_key=(0,0,1,2), y_val=1,
                          a_i_val=2, a_j_prime_val=6, a_l_val=9) == cp_model.INFEASIBLE

def test_55_56_infeasible_both():
    # Both violated: 55: 3+5 ≤ 7 OK, 56: 7+5 ≤ 11 is false -> overall infeasible
    t_prime = {0:{1:5}, 1:{2:5}}
    assert run_case_55_56(T=20, t_prime=t_prime, y_key=(0,0,1,2), y_val=1,
                          a_i_val=3, a_j_prime_val=7, a_l_val=11) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("valid consistent", dict(T=20, t_prime={0:{1:3},1:{2:4}}, y_key=(0,0,1,2), y_val=1,
                                  a_i_val=2, a_j_prime_val=6, a_l_val=12)),
        ("valid relaxed y=0", dict(T=20, t_prime={0:{1:3},1:{2:4}}, y_key=(0,0,1,2), y_val=0,
                                   a_i_val=7, a_j_prime_val=1, a_l_val=1)),
        ("valid boundary", dict(T=15, t_prime={0:{1:5},1:{2:5}}, y_key=(0,0,1,2), y_val=1,
                                a_i_val=0, a_j_prime_val=5, a_l_val=10)),
        ("infeasible service early", dict(T=20, t_prime={0:{1:3},1:{2:4}}, y_key=(0,0,1,2), y_val=1,
                                          a_i_val=2, a_j_prime_val=4, a_l_val=12)),
        ("infeasible rendezvous early", dict(T=20, t_prime={0:{1:3},1:{2:4}}, y_key=(0,0,1,2), y_val=1,
                                             a_i_val=2, a_j_prime_val=6, a_l_val=9)),
        ("infeasible both early", dict(T=20, t_prime={0:{1:5},1:{2:5}}, y_key=(0,0,1,2), y_val=1,
                                       a_i_val=3, a_j_prime_val=7, a_l_val=11)),
    ]
    for label, args in scenarios:
        status = run_case_55_56(**args)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")
