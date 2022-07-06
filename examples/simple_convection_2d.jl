using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (15, 3)
u0 = Array{Float64, 2}(undef, 4, prod(n_particles_per_dimension))
mass = ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    u0[1, particle] = x / 2 + 3
    u0[2, particle] = y / 2

    # Velocity
    u0[3, particle] = -1
    u0[4, particle] = 0
end

tspan = (0.0, 20.0)
ode = Pixie.semidiscretize(u0, mass, tspan)

alive_callback = Pixie.AliveCallback(alive_interval=100)

sol = solve(ode, RDPK3SpFSAL49(), saveat=0.2, callback=alive_callback);
