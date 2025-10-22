from ortools.sat.python import cp_model
import pytest

#-----------------------------------------------------------------------------------------
# Constraint 54: Time propagation on used truck arcs (mirrored exactly)
#-----------------------------------------------------------------------------------------

def build_model_54(T, t):
    model = cp_model.CpModel()
    K = [0]
    VL = [0, 1]   # launch/visit nodes
    VR = [0, 1]   # rendezvous/visit nodes (same set here for test)

    # Time variables
    a = {(0,i): model.NewIntVar(0, T, f"a_0_{i}") for i in VL}

    # Arc variables
    x = {(0,i,j): model.NewBoolVar(f"x_0_{i}_{j}") for i in VL for j in VR if i != j}

    #Constraint 54
    for k in K:
        for i in VL:
            for j in VR:
                if i != j:
                    model.Add(a[k,i] + t[i][j] <= a[k,j] + T * (1 - x[k,i,j]))

    return model, a, x

#-----------------------------------------------------------------------------------------
# Runner
#-----------------------------------------------------------------------------------------
def run_case_54(T, t, x_val, a0_val, a1_val):
    model, a, x = build_model_54(T, t)
    # Fix scenario: only arc 0->1 matters
    model.Add(x[(0,0,1)] == x_val)
    model.Add(a[(0,0)] == a0_val)
    model.Add(a[(0,1)] == a1_val)
    model.Minimize(0)
    solver = cp_model.CpSolver()
    return solver.Solve(model)

#-----------------------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------------------
def test_54_arc_used_consistent():
    t = {0:{1:3}, 1:{0:3}}
    assert run_case_54(T=10, t=t, x_val=1, a0_val=2, a1_val=5) == cp_model.OPTIMAL

def test_54_arc_not_used_relaxed():
    t = {0:{1:3}, 1:{0:3}}
    assert run_case_54(T=10, t=t, x_val=0, a0_val=2, a1_val=0) == cp_model.OPTIMAL

def test_54_arc_used_inconsistent():
    t = {0:{1:3}, 1:{0:3}}
    assert run_case_54(T=10, t=t, x_val=1, a0_val=2, a1_val=4) == cp_model.INFEASIBLE

def test_54_arc_used_too_small():
    t = {0:{1:4}, 1:{0:4}}
    assert run_case_54(T=8, t=t, x_val=1, a0_val=3, a1_val=6) == cp_model.INFEASIBLE

#-----------------------------------------------------------------------------------------
# Feedback
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("valid used (2+3 ≤5)", dict(T=10, t={0:{1:3},1:{0:3}}, x_val=1, a0_val=2, a1_val=5)),
        ("valid not used relaxed", dict(T=10, t={0:{1:3},1:{0:3}}, x_val=0, a0_val=2, a1_val=0)),
        ("infeasible used (2+3 ≤4)", dict(T=10, t={0:{1:3},1:{0:3}}, x_val=1, a0_val=2, a1_val=4)),
        ("infeasible used (3+4 ≤6)", dict(T=8, t={0:{1:4},1:{0:4}}, x_val=1, a0_val=3, a1_val=6)),
    ]
    for label, args in scenarios:
        status = run_case_54(**args)
        print(f"{label}: {'OPTIMAL' if status == cp_model.OPTIMAL else 'INFEASIBLE'}")
