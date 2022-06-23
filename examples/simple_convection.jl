using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (6, 6, 30)
u0 = Array{Float64, 2}(undef, 7, prod(n_particles_per_dimension))

for z in 1:n_particles_per_dimension[3],
        y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] * n_particles_per_dimension[3] +
        (y - 1) * n_particles_per_dimension[3] + z

    # Coordinates
    u0[1, particle] = x / 2
    u0[2, particle] = y / 2
    u0[3, particle] = z / 2 + 3

    # Velocity
    u0[4, particle] = 0
    u0[5, particle] = 0
    u0[6, particle] = -1

    # Mass
    u0[7, particle] = 1
end

tspan = (0.0, 20.0)
ode = Pixie.semidiscretize(u0, tspan)

sol = solve(ode, Euler(), dt=0.05);
