from ortools.sat.python import cp_model

def build_model_sync(T):
    model = cp_model.CpModel()
    K = [0]
    VL = [0]    # depot
    C  = [1]    # one customer
    VR = [2]    # rendezvous

    # Variables
    a       = {(0,0): model.NewIntVar(0, T, "a_0_0"),
               (0,2): model.NewIntVar(0, T, "a_0_2")}
    a_prime = {(0,0): model.NewIntVar(0, T, "a_prime_0_0"),
               (0,2): model.NewIntVar(0, T, "a_prime_0_2")}
    y_drone = {(0,0,1,2): model.NewBoolVar("y_0_0_1_2")}

    # Constraints 57 & 58
    for k in K:
        for i in VL:
            terms = [
                y_drone[k,i,j,l]
                for j in C if j != i
                for l in VR if l != i and l != j and (k,i,j,l) in y_drone
            ]
            if terms:
                sortie_sum = sum(terms)
                model.Add(a_prime[k,i] >= a[k,i] - T * (1 - sortie_sum))  # 57
                model.Add(a_prime[k,i] <= a[k,i] + T * (1 - sortie_sum))  # 58

    # Constraints 59 & 60
    for k in K:
        for l in VR:
            terms = [
                y_drone[k,i,j,l]
                for i in VL if i != l
                for j in C if j != i and j != l and (k,i,j,l) in y_drone
            ]
            if terms:
                sortie_sum = sum(terms)
                model.Add(a_prime[k,l] >= a[k,l] - T * (1 - sortie_sum))  # 59
                model.Add(a_prime[k,l] <= a[k,l] + T * (1 - sortie_sum))  # 60

    return model, a, a_prime, y_drone

def run_sync_case(T, force_y, a0, ap0, a2, ap2):
    model, a, a_prime, y = build_model_sync(T)
    model.Add(y[(0,0,1,2)] == force_y)
    model.Add(a[0,0] == a0)
    model.Add(a_prime[0,0] == ap0)
    model.Add(a[0,2] == a2)
    model.Add(a_prime[0,2] == ap2)
    solver = cp_model.CpSolver()
    return solver.Solve(model)

# --- Test cases ---
print("Valid: y=1, times match:",
      "OPTIMAL" if run_sync_case(50, 1, 5, 5, 12, 12)==cp_model.OPTIMAL else "INFEASIBLE")

print("Infeasible: y=1, launch mismatch:",
      "OPTIMAL" if run_sync_case(50, 1, 5, 7, 12, 12)==cp_model.OPTIMAL else "INFEASIBLE")

print("Infeasible: y=1, rendezvous mismatch:",
      "OPTIMAL" if run_sync_case(50, 1, 5, 5, 12, 14)==cp_model.OPTIMAL else "INFEASIBLE")

print("Trivial: y=0, mismatches allowed:",
      "OPTIMAL" if run_sync_case(50, 0, 5, 7, 12, 14)==cp_model.OPTIMAL else "INFEASIBLE")
