using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (20, 50)
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = ones(Float64, prod(n_particles_per_dimension))
particle_densities = 3 * ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = (x - 1 - 0.5 * (n_particles_per_dimension[1] - 1)) * 0.1
    particle_coordinates[2, particle] = y * 0.1 + 1.0

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
end

n_boundaries_per_dimension = (400,)
spacing = 0.02

boundary_coordinates = Array{Float64, 2}(undef, 2, prod(n_boundaries_per_dimension))
boundary_spacings = spacing * ones(Float64, prod(n_boundaries_per_dimension))
boundary_masses = ones(Float64, prod(n_boundaries_per_dimension))

for y in 1:n_boundaries_per_dimension[1]
    boundary_particle = y

    boundary_coordinates[1, boundary_particle] = spacing * (y - 1 - 0.5 * (n_boundaries_per_dimension[1] - 1))
    boundary_coordinates[2, boundary_particle] = -0.1
end


state_equation = Pixie.StateEquationIdealGas(10.0, 3.0, 10.0, background_pressure=10.0)
semi = Pixie.SPHSemidiscretization(particle_masses, boundary_coordinates,
                                   boundary_masses, boundary_spacings,
                                   Pixie.ContinuityDensity(), state_equation)

tspan = (0.0, 5.0)
ode = Pixie.semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)

alive_callback = Pixie.AliveCallback(alive_interval=100)

sol = solve(ode, RDPK3SpFSAL49(), saveat=0.02, callback=alive_callback);
