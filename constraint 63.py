from ortools.sat.python import cp_model

#-----------------------------------------------------------------------------------------
# Constraint 63: delay[k,i] >= a[k,i] - D[i] and delay[k,i] >= a_prime[k,i] - D[i]
#-----------------------------------------------------------------------------------------

def build_model_63(D_i):
    model = cp_model.CpModel()

    # Sets
    K = [0]        # one vehicle/team for test
    C = [1]        # one node for test

    # Variables
    # Arrival times (truck and drone) at node i
    a       = {(0,1): model.NewIntVar(0, 1000, "a_0_1")}
    a_prime = {(0,1): model.NewIntVar(0, 1000, "a_prime_0_1")}
    # Delay indexed by (k,i)
    delay   = {(0,1): model.NewIntVar(0, 1000, "delay_0_1")}

    # Constraint 63
    for k in K:
        for i in C:
            model.Add(delay[k, i] >= a[k, i] - D_i)
            model.Add(delay[k, i] >= a_prime[k, i] - D_i)

    return model, a, a_prime, delay

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_case_63(D_i, a_val, a_prime_val, delay_val):
    model, a, a_prime, delay = build_model_63(D_i)

    # Fix scenario
    model.Add(a[0,1] == a_val)
    model.Add(a_prime[0,1] == a_prime_val)
    model.Add(delay[0,1] == delay_val)

    # Dummy objective
    model.Minimize(0)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return status

#-----------------------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------------------
def test_63_on_time_both_zero_delay():
    # Truck at 8, Drone at 7, deadline 10 => delay = 0 satisfies both
    assert run_case_63(D_i=10, a_val=8, a_prime_val=7, delay_val=0) == cp_model.OPTIMAL

def test_63_exact_deadline_both_zero_delay():
    # Exactly at deadline => delay = 0
    assert run_case_63(D_i=10, a_val=10, a_prime_val=10, delay_val=0) == cp_model.OPTIMAL

def test_63_truck_late_drone_early_correct_delay():
    # Truck late by 5 (15-10), Drone early (9 <= 10) => delay must be >= 5
    assert run_case_63(D_i=10, a_val=15, a_prime_val=9, delay_val=5) == cp_model.OPTIMAL

def test_63_drone_late_truck_early_correct_delay():
    # Drone late by 6 (16-10), Truck early => delay must be >= 6
    assert run_case_63(D_i=10, a_val=8, a_prime_val=16, delay_val=6) == cp_model.OPTIMAL

def test_63_both_late_delay_equals_max():
    # Truck late by 3, Drone late by 7 => delay must be >= max(3,7)=7
    assert run_case_63(D_i=10, a_val=13, a_prime_val=17, delay_val=7) == cp_model.OPTIMAL

def test_63_late_delay_too_small_truck_dominates():
    # Truck late by 5, Drone early => delay 2 is too small
    assert run_case_63(D_i=10, a_val=15, a_prime_val=9, delay_val=2) == cp_model.INFEASIBLE

def test_63_late_delay_too_small_drone_dominates():
    # Drone late by 6, Truck early => delay 5 is too small (needs >=6)
    assert run_case_63(D_i=10, a_val=8, a_prime_val=16, delay_val=5) == cp_model.INFEASIBLE

def test_63_both_late_delay_too_small():
    # Truck late 3, Drone late 7 => delay 6 is too small (needs >=7)
    assert run_case_63(D_i=10, a_val=13, a_prime_val=17, delay_val=6) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("on time (truck=8, drone=7; D=10)",         dict(D_i=10, a_val=8,  a_prime_val=7,  delay_val=0)),
        ("exact deadline (10,10; D=10)",             dict(D_i=10, a_val=10, a_prime_val=10, delay_val=0)),
        ("truck late by 5, correct delay",           dict(D_i=10, a_val=15, a_prime_val=9,  delay_val=5)),
        ("drone late by 6, correct delay",           dict(D_i=10, a_val=8,  a_prime_val=16, delay_val=6)),
        ("both late, delay=max=7",                   dict(D_i=10, a_val=13, a_prime_val=17, delay_val=7)),
        ("truck late by 5, delay too small (2)",     dict(D_i=10, a_val=15, a_prime_val=9,  delay_val=2)),
        ("drone late by 6, delay too small (5)",     dict(D_i=10, a_val=8,  a_prime_val=16, delay_val=5)),
        ("both late, delay too small (6<7)",         dict(D_i=10, a_val=13, a_prime_val=17, delay_val=6)),
    ]
    for label, args in scenarios:
        status = run_case_63(**args)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")
