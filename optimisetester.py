from ortools.sat.python import cp_model
import numpy as np
model = cp_model.CpModel()

# ---------------- Input Data ----------------
V = [
    (0, 0),    # depot
    (15, 3),   # node 1 
    (18, 7),   # node 2 
    (4, 2),    # node 3 
    (5, 4),    # node 4 
    (3, 6),    # node 5 
    (20, 10),  # node 6 
    (6, 1)     # node 7
]
# ---------------- Demands (weights) ----------------
w = [
    0,   
    6,   
    7,   
    1,   
    2,   
    2,   
    8,   
    1    
]

# ---------------- Deadlines ----------------
D = {
    1: 100,  
    2: 110,  
    3: 35,   
    4: 45,   
    5: 55,   
    6: 120,  
    7: 40    
}

# ---------------- Parameters ----------------
horizon = 150
T = 150                        # Planning horizon (minutes)
E = 35                         # Maximum drone endurance (minutes)
N = 2                          # Number of truck-drone tandems
K = range(N)                   # Set of tandems
depot = 0
num_nodes = len(V)
C = set(range(1, num_nodes))   # Affected areas (excluding depot)
VL = set(range(num_nodes))     # Separation nodes (truck-accessible)
VR = set(range(1, num_nodes))  # Rendezvous nodes (excluding depot)
VT = {1, 2, 6}                 # truck-accessible affected areas
VD = C.difference(VT)          # remaining affected areas served by drone
n = len(C)

WT_max = 15   # Truck capacity
WD_max = 5    # Drone capacity
ct = 2      # Truck cost per minute
cd = 1      # Drone cost per minute
vt = 1.0      # Truck speed (km/min)
vd = 1.5      # Drone speed (km/min)

alpha_value = 5.0    # cost per minute of delay (same for all nodes)
beta_value  = 100.0  # penalty if unserved (same for all nodes)

alpha = {i: alpha_value for i in C}
beta  = {i: beta_value  for i in C}

# ---------------- Time Matrices ----------------
pts = np.array(V)
truck_dist_matrix = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=2)
t_float = truck_dist_matrix / vt                  # truck travel time (float)
euclidean_matrix = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
t_prime_float = euclidean_matrix / vd             # drone travel time (float)

# Convert to integer minutes
t = np.rint(t_float).astype(int)                  # truck time matrix (int)
t_prime = np.rint(t_prime_float).astype(int)      # drone time matrix (int)

# ---------------- Decision Variables ----------------
x = {}
y = {}
u = {}
y_drone = {}

for k in K:
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x[k, i, j] = model.NewBoolVar(f"x_{k}_{i}_{j}")
    for i in range(1, num_nodes):
        y[k, i] = model.NewBoolVar(f"y_{k}_{i}")
        u[k, i] = model.NewIntVar(0, num_nodes - 1, f"u_{k}_{i}")
        model.Add(u[k, i] >= y[k, i])
        model.Add(u[k, i] <= (num_nodes - 1) * y[k, i])


y_drone = {}
for k in K:
    for i in VL:          
        for j in C:       
            for l in VR:  
                y_drone[k, i, j, l] = model.NewBoolVar(f"y_drone_{k}_{i}_{j}_{l}")
                if (j not in VD) or (i == j) or (i == l) or (j == l):
                    model.Add(y_drone[k, i, j, l] == 0)

P = {}
for k in K:
    for i in VL:
        for j in C:
            if i != j:
                P[k, i, j] = model.NewBoolVar(f"P_{k}_{i}_{j}")

a = {}
a_prime = {}

for k in K:
    for i in range(num_nodes):
        a[k, i] = model.NewIntVar(0, horizon, f"a_{k}_{i}")         
        a_prime[k, i] = model.NewIntVar(0, horizon, f"a_prime_{k}_{i}")  

delay = {}
for k in K:
    for i in C:
        delay[k, i] = model.NewIntVar(0, horizon, f"delay_{k}_{i}")


# ---------------- Constraints ----------------
# (36) Each affected area visited at most once (truck or drone)
for j in C:
    truck_part = sum(x[k, i, j] for k in K for i in VL if i != j)
    drone_part = sum(y[k, i, j, l] for k in K for j in C for i in VL if i != j for l in VR if l != i)
    model.Add(truck_part + drone_part <= 1)

# (37, 38) Depot departure and return
for k in K:
    model.Add(sum(x[k, depot, j] for j in C) <= 1)  
    model.Add(sum(x[k, i, depot] for i in C) <= 1)  

# (39) Flow conservation
for k in K:
    for j in C:
        incoming = sum(x[k, i, j] for i in VL if i != j)
        outgoing = sum(x[k, j, l] for l in VR if l != j)
        model.Add(incoming - outgoing == 0)

# (40) Trucks cannot reach road-damaged areas
for k in K:
    for i in VD:
        for j in range(num_nodes):
            if i != j and (k,i,j) in x:
                model.Add(x[k,i,j] == 0)
                model.Add(x[k,j,i] == 0)

# (41,42) prevent the formation of subtours for the truck by ensuring that the truck does not traverse through previously visited arcs
M = len(C)  # maximum number of customer nodes
for k in K:
    for i in VL:
        for j in VR:
            if i != j and i != depot and j != depot:
                model.Add(u[k, i] - u[k, j] + 1 <= M * (1 - x[k, i, j]))

for k in K:
    for j in VR:
        incoming = sum(x[k, i, j] for i in VL if i != j)
        model.Add(u[k, j] <= M * incoming)

# (43,44) define the sequence of truck tours to prevent a node from being visited mulitple times within a single truck route
for k in K:
    for i in VL:
        for j in C:
            if i != j and i != depot and j != depot:
                model.Add(u[k, j] - u[k, i] <= M * P[k, i, j])
                model.Add(u[k, j] - u[k, i] >= M * (P[k, i, j] - 1) + 1)

# (45): enforces capacity limit for truck
for k in K:
    weighted_effort = []

    for i in C:
        # First term: truck arcs from i to rendezvous j
        for j in VR:
            if j != i:
                weighted_effort.append(w[j] * x[k, i, j])

    # Second term: drone arcs from i to j to l
    for j in VD:
        for i in VL:
            if i != j and i != k:
                for l in VR:
                    if l != i and l != j:
                        weighted_effort.append(w[j] * y_drone[k, i, j, l])

    model.Add(sum(weighted_effort) <= WT_max)

# (46) drones are restricted to serving affected areas within a set (V_d)
for k in K:
    for i in VL:
        for j in C:
            for l in VR:
                if (
                    i == j or i == l or
                    j == i or j == l or j not in VD or
                    l == i or l == j
                ):
                    model.Add(y_drone[k, i, j, l] == 0)

# (47,48) the drone can be launched and returned only once per node
for k in K:
    for i in VL:
        launch_trips = []
        for j in C:
            if j != i:
                for l in VR:
                    if l != i and l != j:
                        launch_trips.append(y_drone[k, i, j, l])
        model.Add(sum(launch_trips) <= 1)

for k in K:
    for l in VR:
        rendezvous_trips = []
        for i in VL:
            if i != l:
                for j in C:
                    if j != i and j != l:
                        rendezvous_trips.append(y_drone[k, i, j, l])
        model.Add(sum(rendezvous_trips) <= 1)

# (49)the drone can be launched and retrieved at different nodes along the truck route
for k in K:
    for i in VT.union({depot}):     
        for j in VD:                
            for l in VR:            
                if i == l or i == j or j == l:
                    continue
                if (k, i, j, l) not in y_drone:
                    continue  
                y_var = y_drone[k, i, j, l]
                sum_out_i = sum(x[k, i, t] for t in VT.union({depot}) if t != i)
                sum_in_l  = sum(x[k, t, l] for t in VT.union({depot}) if t != l)
                model.Add(2 * y_var <= sum_out_i + sum_in_l)

# (50) mandates that the associated truck must depart from any node to reach the rendezvous node l
for k in K:
    for j in C:
        for l in VR:
            if j != l:
                rhs = sum(x[k, i, l] for i in VL if i != j and i != l)
                model.Add(y_drone[k, 0, j, l] <= rhs)

# (51,52) initialize the arrival time of the truck and drone at the start of each route to zero, ensuring routes commence from the depot at the beginning
for k in K:
    model.Add(a[k, 0] == 0)
    model.Add(a_prime[k, 0] == 0)

# (53) ensures that the arrival time of the truck at the depot does not exceed the planning horizon T
for k in K:
    model.Add(a[k, depot] <= T)

# (54) ensures the continuity of truck arrival times, requiring that a truck’s arrival at node j is later than at node i if j is visited after i
for k in K:
    for i in VL:
        for j in VR:
            if i != j:
                model.Add(
                    a[k, i] + t[i][j] <= a[k, j] + T * (1 - x[k, i, j])
                )
            
# (55, 56) similarly guarantee drone arrival time continuity, ensuring that a drone’s arrival at subsequent nodes is sequential
# (55) a_i^k + t'_{ij} - T(1 - Σ_{l∈VR\{i,j}} y_{ijl}^k) ≤ a'_j^k   ∀k, i∈VL\{j}, j∈C\{i}
for k in K:
    for i in VL:
        for j in C:
            if i == j:
                continue
            # collect flights (i,j,l)
            flights_ijl = [y_drone[k, i, j, l]
                           for l in VR
                           if l != i and l != j and (k, i, j, l) in y_drone]
            if flights_ijl:
                # sum is 0 or 1 (due to launch/rendezvous uniqueness)
                sum_ijl = sum(flights_ijl)
                model.Add(a[k, i] + t_prime[i][j] - T * (1 - sum_ijl) <= a_prime[k, j])

# (56) a'_j^k + t'_{jl} - T(1 - Σ_{i∈VL\{j,l}} y_{ijl}^k) ≤ a_l^k   ∀k, j∈C\{l}, l∈VR\{j}
for k in K:
    for j in C:
        for l in VR:
            if j == l:
                continue
            flights_ijl = [y_drone[k, i, j, l]
                           for i in VL
                           if i != j and i != l and (k, i, j, l) in y_drone]
            if flights_ijl:
                sum_ijl = sum(flights_ijl)
                model.Add(a_prime[k, j] + t_prime[j][l] - T * (1 - sum_ijl) <= a[k, l])

# (57-60) synchronize the arrival times of trucks and drones, ensuring synchronized launch and rendezvous
# 57 and 58: launch synchronization
for k in K:
    for i in VL:
        terms = [
            y_drone[k, i, j, l]
            for j in C if j != i
            for l in VR if l != i and l != j and (k, i, j, l) in y_drone
        ]
        if terms:  # check the list, not the sum
            sortie_sum = sum(terms)
            model.Add(a_prime[k, i] >= a[k, i] - T * (1 - sortie_sum))  # 57
            model.Add(a_prime[k, i] <= a[k, i] + T * (1 - sortie_sum))  # 58

# 59 and 60: rendezvous synchronization
for k in K:
    for l in VR:
        terms = [
            y_drone[k, i, j, l]
            for i in VL if i != l
            for j in C if j != i and j != l and (k, i, j, l) in y_drone
        ]
        if terms:
            sortie_sum = sum(terms)
            model.Add(a_prime[k, l] >= a[k, l] - T * (1 - sortie_sum))  # 59
            model.Add(a_prime[k, l] <= a[k, l] + T * (1 - sortie_sum))  # 60

# (61) ensures that the total flight time of the drone does not exceed its endurance E
for k in K:
    for i in VL:
        for j in C:
            for l in VR:
                if i != j and i != l and j != l and (k, i, j, l) in y_drone:
                    model.Add(
                        t_prime[i][j]        # t'_{ij}
                        + t_prime[j][l]      # t'_{jl}
                        - T * (1 - y_drone[k, i, j, l])
                        <= E
                    )

# (62) prevents trucks from launching drones that are still delivering, ensuring sequential operations
for k in K:
    for i in VL:
        for l in VR:
            for b in C:
                if i != b and i != l and l != b:
                    # First sum: Σ_{j ∈ C \ {i,l}} y_{i j l}^k
                    sum1_terms = [
                        y_drone[k, i, j, l]
                        for j in C
                        if j != i and j != l and (k, i, j, l) in y_drone
                    ]

                    # Second sum: Σ_{q ∈ C \ {b,m}} Σ_{m ∈ VR \ {b,q}} y_{b q m}^k
                    sum2_terms = [
                        y_drone[k, b, q, m]
                        for q in C if q != b
                        for m in VR if m != b and m != q and (k, b, q, m) in y_drone
                    ]

                    # Only add if at least one term exists
                    if sum1_terms or sum2_terms or (k, l, b) in P:
                        sum1 = sum(sum1_terms) if sum1_terms else 0
                        sum2 = sum(sum2_terms) if sum2_terms else 0
                        P_var = P[k, l, b] if (k, l, b) in P else 0

                        model.Add(
                            a_prime[k, l]
                            - T * (3 - sum1 - sum2 - P_var)
                            <= a_prime[k, b]
                        )

#63 calculates the delay time of truck k or drone k at node i.
for k in K:
    for i in C:
        model.Add(delay[k, i] >= a[k, i] - D[i])       # truck lateness
        model.Add(delay[k, i] >= a_prime[k, i] - D[i])  # drone lateness

# ---------------- Objective Function ----------------
truck_cost = sum(t[i][j] * ct * x[k, i, j]
                 for k in K
                 for i in range(num_nodes)
                 for j in range(num_nodes) if i != j)

drone_cost = sum((t_prime[i][j] + t_prime[j][l]) * cd * y_drone[k, i, j, l]
                 for k in K
                 for i in VL
                 for j in C
                 for l in VR
                 if i != j and j != l and i != l)

delay_penalty = sum(alpha[i] * delay[k, i] for k in K for i in C)

# Truck service credit: Σ_{k∈K} Σ_{j∈VR\{i}} x_{i j}^k
truck_service_terms = {
    i: [x[k, i, j] for k in K for j in VR
        if j != i and (k, i, j) in x]
    for i in C
}

# Drone service credit: Σ_{k∈K} Σ_{j∈C\{i,l}} Σ_{l∈VR\{i,j}} y_{i j l}^k
drone_service_terms = {
    i: [y_drone[k, i, j, l] for k in K for j in C for l in VR
        if j != i and l != i and l != j and (k, i, j, l) in y_drone]
    for i in C
}

# Unserved penalty: Σ_{i∈C} β_i (1 − Σtruck − Σdrone)
unserved_penalty = sum(
    beta[i] * (1 - (sum(truck_service_terms[i]) + sum(drone_service_terms[i])))
    for i in C
)

# Final objective
model.Minimize(truck_cost + drone_cost + delay_penalty + unserved_penalty)

# ---------------- Solve ----------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_search_workers = 8
status = solver.Solve(model)

# ---------------- Solution Printer ----------------
# ---------------- Solution Printer ----------------
def print_solution_min(status):
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Status:", solver.StatusName(status))
        print("No solution.")
        return
    print("Status:", solver.StatusName(status))
    print("Objective:", solver.ObjectiveValue())

    # ---- Truck arcs used (added) ----
    print("\nTruck arcs used:")
    for k in K:
        used = [f"{i}->{j}" for i in range(num_nodes) for j in range(num_nodes)
                if i != j and (k, i, j) in x and solver.Value(x[k, i, j]) == 1]
        print(f"  Truck {k}: {used if used else 'None'}")

    # ---- Truck deliveries (nodes in VT) ----
    print("\nTruck deliveries:")
    found_truck_delivery = False
    for k in K:
        for j in VT:
            incoming_arcs = []
            for i in VT.union({depot}):
                if i != j and (k, i, j) in x and solver.Value(x[k, i, j]) == 1:
                    incoming_arcs.append(i)
            if len(incoming_arcs) == 1:
                found_truck_delivery = True
                t_arr = solver.Value(a[k, j]) if (k, j) in a else "N/A"
                print(f"  Truck {k} delivers to node {j} at time {t_arr}")
    if not found_truck_delivery:
        print("  None")

    # ---- Drone flights ----
    print("\nDrone flights:")
    found_drone_flight = False
    for (k, i, j, l), var in y_drone.items():
        if solver.Value(var) == 1:
            found_drone_flight = True
            launch_t = solver.Value(a[k, i]) if (k, i) in a else "N/A"
            service_t = solver.Value(a_prime[k, j]) if (k, j) in a_prime else "N/A"
            rend_t = solver.Value(a[k, l]) if (k, l) in a else "N/A"
            print(f"  Tandem {k}: launch {i} t={launch_t}, deliver {j} t={service_t}, rendezvous {l} t={rend_t}")
    if not found_drone_flight:
        print("  None")

    # ---- Rendezvous events ----
    print("\nRendezvous events:")
    rendezvous = set()
    for (k, i, j, l), var in y_drone.items():
        if solver.Value(var) == 1:
            rendezvous.add((k, l))
    if rendezvous:
        for k, l in sorted(rendezvous):
            rt = solver.Value(a[k, l]) if (k, l) in a else "N/A"
            print(f"  Tandem {k} rendezvous at node {l} time={rt}")
    else:
        print("  None")

# Call after solving
print_solution_min(status)