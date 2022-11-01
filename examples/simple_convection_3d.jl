using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (10, 10, 10)
particle_coordinates = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
particle_masses = ones(Float64, prod(n_particles_per_dimension))

for z in 1:n_particles_per_dimension[3],
        y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] * n_particles_per_dimension[3] +
        (y - 1) * n_particles_per_dimension[3] + z

    # Coordinates
    particle_coordinates[1, particle] = (x - 1 - 0.5 * (n_particles_per_dimension[1] - 1)) * 0.1
    particle_coordinates[2, particle] = y * 0.1
    particle_coordinates[3, particle] = z * 0.1

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
    particle_velocities[3, particle] = 0
end

smoothing_length = 0.12
semi = SPHFluidSemidiscretization{3}(particle_masses, SummationDensity(),
                                StateEquationCole(10.0, 7, 1000.0, 1.0, background_pressure=1.0),
                                SchoenbergCubicSplineKernel{3}(), smoothing_length)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, particle_coordinates, particle_velocities, tspan)

alive_callback = AliveCallback(alive_interval=20)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0)

callbacks = CallbackSet(alive_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks);
