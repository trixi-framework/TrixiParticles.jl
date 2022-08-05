using Pixie
using OrdinaryDiffEq

# Particle data
n_particles_per_dimension = (10, 30)
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

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

# Boundary particle data
n_boundaries_per_dimension = (400,)
spacing = 0.02

boundary_coordinates = Array{Float64, 2}(undef, 2, prod(n_boundaries_per_dimension))
boundary_spacings = spacing * ones(Float64, prod(n_boundaries_per_dimension))
boundary_masses = 10 * ones(Float64, prod(n_boundaries_per_dimension))

for y in 1:n_boundaries_per_dimension[1]
    boundary_particle = y

    boundary_coordinates[1, boundary_particle] = spacing * (y - 1 - 0.5 * (n_boundaries_per_dimension[1] - 1))
    boundary_coordinates[2, boundary_particle] = -0.1
end

K = 1000.0
boundary_conditions = Pixie.BoundaryConditionMonaghanKajtar(K, boundary_coordinates,
                                                            boundary_masses, boundary_spacings)

# Create semidiscretization
state_equation = Pixie.StateEquationTait(10.0, 7, 1000.0, 1.0, background_pressure=1.0)
# state_equation = Pixie.StateEquationIdealGas(10.0, 3.0, 10.0, background_pressure=10.0)

smoothing_length = 0.12
semi = Pixie.SPHSemidiscretization{2}(particle_masses,
                                      Pixie.ContinuityDensity(), state_equation,
                                      Pixie.CubicSplineKernel{2}(), smoothing_length,
                                      viscosity=Pixie.ArtificialViscosityMonaghan(1.0, 2.0),
                                      boundary_conditions=boundary_conditions,
                                      gravity=(0.0, -9.81))

tspan = (0.0, 5.0)
ode = Pixie.semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)

alive_callback = Pixie.AliveCallback(alive_interval=100)

sol = solve(ode, RDPK3SpFSAL49(), dt=1e-5, saveat=0.02, callback=alive_callback);
