using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (20, 20)
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension))

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

smoothing_length = 0.12
semi = Pixie.SPHSemidiscretization{2}(particle_masses, Pixie.SummationDensity(),
                                      Pixie.StateEquationTait(10.0, 7, 1000.0, 1.0, background_pressure=1.0),
                                      Pixie.CubicSplineKernel{2}(), smoothing_length,
                                      viscosity=Pixie.ArtificialViscosityMonaghan(1.0, 2.0))

tspan = (0.0, 5.0)
ode = Pixie.semidiscretize(semi, particle_coordinates, particle_velocities, tspan)

alive_callback = Pixie.AliveCallback(alive_interval=100)

sol = solve(ode, RDPK3SpFSAL49(), saveat=0.02, callback=alive_callback);
