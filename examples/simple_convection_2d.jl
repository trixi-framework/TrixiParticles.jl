using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (2, 1)
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = ones(Float64, prod(n_particles_per_dimension))
particle_densities = 3 * ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x * 0.1
    particle_coordinates[2, particle] = y * 0.1

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
end

state_equation = Pixie.StateEquationIdealGas(10.0, 1000.0, 1.0e5)
semi = Pixie.SPHSemidiscretization{2}(particle_masses,
                                      Pixie.ContinuityDensity(), state_equation)

tspan = (0.0, 5.0)
ode = Pixie.semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)

alive_callback = Pixie.AliveCallback(alive_interval=100)

sol = solve(ode, Euler(), dt=0.002, saveat=0.002, callback=alive_callback);
