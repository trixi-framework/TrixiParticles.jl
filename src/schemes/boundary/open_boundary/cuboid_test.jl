using LinearAlgebra

particle_position = [2, 0]

P_1 = [0, 0]#, 0]
P_2 = [1, 0]#, 0]
P_3 = [0, 3]#, 3]
P_4 = [0, 0]#, 3]

boundary_zone = (P_1, P_2, P_3)

#### directions
u = boundary_zone[2] - boundary_zone[1]
v = boundary_zone[3] - boundary_zone[1]
#w = boundary_zone[4] - boundary_zone[1]

particle_distance = particle_position - boundary_zone[1]

condition_1 = 0 <= dot(particle_distance, u) <= dot(u, u)
condition_2 = 0 <= dot(particle_distance, v) <= dot(v, v)
#condition_3 = 0 <= dot(particle_distance, w) <= dot(w, w)

in_domain = condition_1 && condition_2 && condition_3
