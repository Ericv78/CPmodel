from ortools.sat.python import cp_model

#-----------------------------------------------------------------------------------------
# Constraints 43 & 44: Sequence definition (precedence with Big-M)
#-----------------------------------------------------------------------------------------

def run_case_sequence(p_12, p_21):
    model = cp_model.CpModel()

    # Sets
    K = [0]          # single tandem
    depot = 0
    VL = {0,1,2}     # truck-accessible nodes (include depot)
    C = {1,2}        # customers
    M = 10           # Big-M

    # Arc indicator variables P[k,i,j] (only for customer-customer arcs)
    P = {
        (0,1,2): model.NewBoolVar("P_0_1_2"),
        (0,2,1): model.NewBoolVar("P_0_2_1"),
    }

    # Order variables u[k,i]
    u = {
        (0,1): model.NewIntVar(0, M, "u_0_1"),
        (0,2): model.NewIntVar(0, M, "u_0_2"),
    }

    # Fix arc choices
    model.Add(P[0,1,2] == p_12)
    model.Add(P[0,2,1] == p_21)

    #Constraints (43) and (44)
    for k in K:
        for i in VL:
            for j in C:
                if i != j and i != depot and j != depot:
                    model.Add(u[k,j] - u[k,i] <= M * P[k,i,j])           # (43)
                    model.Add(u[k,j] - u[k,i] >= M * (P[k,i,j] - 1) + 1) # (44)

    # Dummy objective
    model.Minimize(0)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    u1 = solver.Value(u[0,1]) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    u2 = solver.Value(u[0,2]) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    return status, u1, u2

#-----------------------------------------------------------------------------------------
# Test Cases
#-----------------------------------------------------------------------------------------
def test_seq_forward():
    status, u1, u2 = run_case_sequence(1,0)  # 1->2 active
    assert status == cp_model.OPTIMAL
    assert u2 > u1

def test_seq_reverse():
    status, u1, u2 = run_case_sequence(0,1)  # 2->1 active
    assert status == cp_model.OPTIMAL
    assert u1 > u2

def test_seq_none():
    status, u1, u2 = run_case_sequence(0,0)  # no ordering enforced
    assert status == cp_model.OPTIMAL

def test_seq_cycle():
    status, u1, u2 = run_case_sequence(1,1)  # both arcs => contradictory
    assert status == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Run scenarios
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("forward",1,0),
        ("reverse",0,1),
        ("none",0,0),
        ("cycle",1,1),
    ]
    for label,p12,p21 in scenarios:
        status,u1,u2 = run_case_sequence(p12,p21)
        s = "OPTIMAL" if status == cp_model.OPTIMAL else "INFEASIBLE"
        print(f"{label}: status={s}, u1={u1}, u2={u2}")
