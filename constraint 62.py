from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 62: Sequential drone operations (exact mimic of provided loop)
# a'_k,l - T * (3 - Σ_{j∈C\{i,l}} y_{k,i,j,l} - Σ_{q∈C\{b}} Σ_{m∈VR\{b,q}} y_{k,b,q,m} - P_{k,l,b}) ≤ a'_k,b
# Activated (enforces a'_k,l ≤ a'_k,b) only when:
#   1) At least one drone sortie ends at l (first sum > 0)
#   2) At least one subsequent sortie launches from b (second sum > 0)
#   3) Truck arc (l,b) exists (P_{k,l,b} = 1)
# Otherwise relaxed by big-M T.
#-----------------------------------------------------------------------------------------

def build_model_62(T):
    model = cp_model.CpModel()

    # Sets
    K  = [0]
    VL = [0, 1]        # launch nodes
    VR = [2, 3]        # rendezvous nodes
    C  = [1, 2]        # customer nodes (distinct from some rendezvous)
    # We focus on triple (i=0, l=2, b=1) to activate sums.

    # Time variables for all l in VR and b in C (only those referenced matter)
    a_prime = {}
    for k in K:
        for node in set(VR + C):
            a_prime[(k, node)] = model.NewIntVar(0, T, f"a_prime_{k}_{node}")

    # Drone sortie variables y_{k,i,j,l} for all combinations (filtered later by membership tests)
    y_drone = {}
    for k in K:
        for i in VL:
            for j in C:
                for l in VR:
                    key = (k, i, j, l)
                    y_drone[key] = model.NewBoolVar(f"y_{k}_{i}_{j}_{l}")

    # Truck arc presence variables P_{k,l,b} for all l∈VR, b∈C
    P = {}
    for k in K:
        for l in VR:
            for b in C:
                P[(k, l, b)] = model.NewBoolVar(f"P_{k}_{l}_{b}")

    #Constraint 62 implementation
    for k in K:
        for i in VL:
            for l in VR:
                for b in C:
                    if i != b and i != l and l != b:
                        # First sum: Σ_{j ∈ C \ {i,l}} y_{k,i,j,l}
                        sum1_terms = [
                            y_drone[k, i, j, l]
                            for j in C
                            if j != i and j != l and (k, i, j, l) in y_drone
                        ]

                        # Second sum: Σ_{q ∈ C \ {b}} Σ_{m ∈ VR \ {b,q}} y_{k,b,q,m}
                        sum2_terms = [
                            y_drone[k, b, q, m]
                            for q in C if q != b
                            for m in VR if m != b and m != q and (k, b, q, m) in y_drone
                        ]

                        if sum1_terms or sum2_terms or (k, l, b) in P:
                            sum1 = sum(sum1_terms) if sum1_terms else 0
                            sum2 = sum(sum2_terms) if sum2_terms else 0
                            P_var = P[k, l, b] if (k, l, b) in P else 0

                            model.Add(
                                a_prime[k, l]
                                - T * (3 - sum1 - sum2 - P_var)
                                <= a_prime[k, b]
                            )

    return model, a_prime, y_drone, P

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_case_62(T, a_l, a_b, y_first, y_second, p_lb):
    model, a_prime, y_drone, P = build_model_62(T)

    # Set values for activation triple: first sortie (0,0,1,2), second (0,1,2,3), arc P(2,1)
    model.Add(a_prime[(0, 2)] == a_l)
    model.Add(a_prime[(0, 1)] == a_b)
    model.Add(y_drone[(0, 0, 1, 2)] == y_first)
    model.Add(y_drone[(0, 1, 2, 3)] == y_second)
    model.Add(P[(0, 2, 1)] == p_lb)

    # Leave all other variables free (can be 0)
    model.Minimize(0)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Compute activation expression for inspection
    sum1 = y_first               # only j=1 contributes
    sum2 = y_second              # only (b=1,q=2,m=3) contributes
    lhs = solver.Value(a_prime[(0, 2)]) - T * (3 - sum1 - sum2 - p_lb)
    rhs = solver.Value(a_prime[(0, 1)])
    print(f"LHS={lhs} RHS={rhs} (y_first={y_first}, y_second={y_second}, P={p_lb})")

    return status

#-----------------------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------------------
def test_62_relaxed_only_first():
    assert run_case_62(50, a_l=20, a_b=10, y_first=1, y_second=0, p_lb=1) == cp_model.OPTIMAL

def test_62_relaxed_only_second():
    assert run_case_62(50, a_l=25, a_b=5, y_first=0, y_second=1, p_lb=1) == cp_model.OPTIMAL

def test_62_relaxed_missing_arc():
    assert run_case_62(50, a_l=30, a_b=10, y_first=1, y_second=1, p_lb=0) == cp_model.OPTIMAL

def test_62_active_sequential_ok():
    assert run_case_62(50, a_l=20, a_b=25, y_first=1, y_second=1, p_lb=1) == cp_model.OPTIMAL

def test_62_active_violation():
    assert run_case_62(50, a_l=30, a_b=15, y_first=1, y_second=1, p_lb=1) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("relaxed only first", dict(T=50, a_l=20, a_b=10, y_first=1, y_second=0, p_lb=1)),
        ("relaxed only second", dict(T=50, a_l=25, a_b=5, y_first=0, y_second=1, p_lb=1)),
        ("relaxed missing arc", dict(T=50, a_l=30, a_b=10, y_first=1, y_second=1, p_lb=0)),
        ("active sequential ok", dict(T=50, a_l=20, a_b=25, y_first=1, y_second=1, p_lb=1)),
        ("active violation", dict(T=50, a_l=30, a_b=15, y_first=1, y_second=1, p_lb=1)),
    ]
    for label, args in scenarios:
        status = run_case_62(**args)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")