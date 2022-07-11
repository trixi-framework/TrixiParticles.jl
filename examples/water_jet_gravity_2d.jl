using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (6, 30)
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x / 4
    particle_coordinates[2, particle] = y / 4 + 3

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
end

n_boundaries_per_dimension = (1000,)
spacing = 0.01

boundary_coordinates = Array{Float64, 2}(undef, 2, prod(n_boundaries_per_dimension))
boundary_spacings = spacing * ones(Float64, prod(n_boundaries_per_dimension))
boundary_masses = 100000 * ones(Float64, prod(n_boundaries_per_dimension))

for y in 1:n_boundaries_per_dimension[1]
    boundary_particle = y

    boundary_coordinates[1, boundary_particle] = spacing * (-0.5 * n_boundaries_per_dimension[1] + y) + 1
    boundary_coordinates[2, boundary_particle] = -1.0
end


semi = Pixie.SPHSemidiscretization(particle_masses, boundary_coordinates,
                                   boundary_masses, boundary_spacings)

tspan = (0.0, 12.0)
ode = Pixie.semidiscretize(semi, particle_coordinates, particle_velocities, tspan)

alive_callback = Pixie.AliveCallback(alive_interval=100)

sol = solve(ode, RDPK3SpFSAL49(), saveat=0.04, callback=alive_callback);
