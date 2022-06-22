using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = 2
u0 = Array{Float64, 2}(undef, 7, n_particles_per_dimension^3)

for z in 1:n_particles_per_dimension,
        y in 1:n_particles_per_dimension,
        x in 1:n_particles_per_dimension
    particle = (x - 1) * n_particles_per_dimension^2 + (y - 1) * n_particles_per_dimension + z

    # Coordinates
    u0[1, particle] = x
    u0[2, particle] = y
    u0[3, particle] = z

    # Velocity
    u0[4, particle] = 1
    u0[5, particle] = 1
    u0[6, particle] = 0

    # Mass
    u0[7, particle] = 1
end

tspan = (0.0, 1.0)
ode = Pixie.semidiscretize(u0, tspan)

sol = solve(ode, Euler(), dt=0.1);
