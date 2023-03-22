# To test FSI, this example simulates a beam which is clamped on both ends and is bent by
# the load of a water column. Thereafter, the y-deflection of the beam is compared with the
# analytical deflection curve.
#
# The analytical approach assumes that the beam is made from homogeneous material, is much
# longer than any dimension of its cross-section and also the beam does not bend severely
# or coming close to deforming or fracturing (for more details see the "analytical solution"
# section of this example).
#
# In addition to these assumptions, we replaced the uniform load of the beam
# under gravity (by its own weight) by the load of the water column under gravity.
# Therefore, the load is not uniformly distributed anymore, since the height
# of the water column is higher towards the center of the beam.
#
# Because of these assumptions/modifications, we cannot expect
# a good agreement with the analytical solution.

using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Solid

length_beam = 1.0
thickness_beam = length_beam / 10
n_particles_y = 15
solid_density = 1500.0

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness_beam / (n_particles_y - 1)

# Lamé constants
E = 1.4e6
nu = 0.0 # e.g. for a rod: Diameter does not change when its length will change

solid_smoothing_length = sqrt(2) * solid_particle_spacing
solid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

n_particles_per_dimension = (round(Int, length_beam / solid_particle_spacing - 1),
                             n_particles_y)

# shift the beam outwards since coordinates start in negative coordinate directions
y_position = -(n_particles_y - 1) * solid_particle_spacing

beam = RectangularShape(solid_particle_spacing, n_particles_per_dimension,
                        (solid_particle_spacing, y_position),
                        density=solid_density)

n_boundary_layers = 5

# The beam is clamped between two rigid "pillars". Calculate the start coordinates of the fixed
# particles for the left pillar (A) and the right pillar (B).
# Since coordinates start in negative coordinate directions,
# shift x and y coordinates outwards and downwards respectively.
x_start_clamp_A = -(n_boundary_layers - 1) * solid_particle_spacing
x_start_clamp_B = (beam.n_particles_per_dimension[1] + 1) * solid_particle_spacing

# Start point of the pillars in y-direction.
y_start_clamp = -(2 * n_particles_y - 1) * solid_particle_spacing

clamp_A = RectangularShape(solid_particle_spacing, (n_boundary_layers, 10 * n_particles_y),
                           (x_start_clamp_A, y_start_clamp),
                           density=solid_density)

clamp_B = RectangularShape(solid_particle_spacing, (n_boundary_layers, 10 * n_particles_y),
                           (x_start_clamp_B, y_start_clamp),
                           density=solid_density)

clamp_coordinates = hcat(clamp_A.coordinates, clamp_B.coordinates)
particle_coordinates = hcat(beam.coordinates, clamp_coordinates)
particle_velocities = zeros(size(particle_coordinates))
particle_masses = vcat(beam.masses, clamp_A.masses, clamp_B.masses)
particle_densities = vcat(beam.densities, clamp_A.densities, clamp_B.densities)

n_fixed_particles = size(clamp_coordinates, 2)

# ==========================================================================================
# ==== Fluid

water_density = 1000.0
water_height = thickness_beam * solid_density / water_density
water_width = 1.0
fluid_particle_spacing = solid_particle_spacing * 2

fluid_smoothing_length = 1.2 * fluid_particle_spacing
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

sound_speed = 10 * sqrt(9.81 * water_height)

state_equation = StateEquationCole(sound_speed, 7, water_density, 100000.0,
                                   background_pressure=100000.0)
viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

# Remove one particle in x-direction, so that the water column fits in between the two
# pillars, even though the load of the water is not correct anymore. The simulation is not
# physically correct anyway (see comment above).
n_particles_per_dimension = (round(Int, water_width / fluid_particle_spacing) - 1,
                             round(Int, water_height / fluid_particle_spacing))

fluid = RectangularShape(fluid_particle_spacing, n_particles_per_dimension,
                         (fluid_particle_spacing, fluid_particle_spacing),
                         density=water_density)

# ==========================================================================================
# ==== Boundary models

hydrodynamic_densites = water_density * ones(size(particle_densities))
hydrodynamic_masses = hydrodynamic_densites * solid_particle_spacing^2

boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densites,
                                                   hydrodynamic_masses, state_equation,
                                                   AdamiPressureExtrapolation(),
                                                   fluid_smoothing_kernel,
                                                   fluid_smoothing_length)

# beta = 3
# K = 9.81 * water_height
# boundary_model_solid = BoundaryModelMonaghanKajtar(K, beta, fluid_particle_spacing / beta)

# ==========================================================================================
# ==== Containers

solid_container = SolidParticleContainer(particle_coordinates, particle_velocities,
                                         particle_masses, particle_densities,
                                         solid_smoothing_kernel, solid_smoothing_length,
                                         E, nu, boundary_model_solid,
                                         n_fixed_particles=n_fixed_particles,
                                         acceleration=(0.0, 0.0),
                                         penalty_force=PenaltyForceGanzenmueller(alpha=0.1))

fluid_container = FluidParticleContainer(fluid.coordinates,
                                         zeros(size(fluid.coordinates)),
                                         fluid.masses, fluid.densities,
                                         ContinuityDensity(), state_equation,
                                         fluid_smoothing_kernel, fluid_smoothing_length,
                                         water_density,
                                         viscosity=viscosity,
                                         acceleration=(0.0, gravity))

# rigid_solid_container = BoundaryParticleContainer(particle_coordinates, boundary_model_solid)

# ==========================================================================================
# ==== Simulation

tspan = (0.0, 2.0)

semi = Semidiscretization(solid_container, fluid_container,# rigid_solid_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=10.0)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# ==========================================================================================
# ==== analytical solution

# In the following, the y deflection of the beam is compared with the analytical
# deflection curve (one of the most widely used formulas in structural engineering).
# Note that this is not the correct analytical soluation for this example,
# but just an approximation (see comment at the top of this file).
#
# The equation is derived in e.g.:
# Lubliner, J. & Papadopoulos, P., "Introduction to Solid Mechanics", Ch. 8,
# DOI: 10.1007/978-3-319-18878-2

function plot_analytical(solid_container)
    # distributed transverse force
    q_0 = gravity * solid_density * thickness_beam^2

    # second moment of area about z-axis (beam bends in the plane perpendicular to the z-axis)
    I_z = thickness_beam^4 / 12

    L = length_beam

    # d⁴v/dx⁴ = q_0 / (E * I_z)
    # (This "v" should not be confused with the same symbol used for velocity. To denote displacement
    # in solid mechanics the symbol u, v and w are used for x, y and z-direction, respectively)
    v(x) = q_0 * L^4 / (24 * E * I_z) * ((x / L)^4 - 2 * (x / L)^3 + (x / L)^2)

    # The upper edge of the beam is at position y = 0.0
    centroidal_fiber_position = (n_particles_y - 1) / 2 * solid_particle_spacing

    # In a homogeneous linearly elastic beam in pure bending, the neutral fiber (zero stress)
    # coincides with the centroidal fiber
    neutral_fiber_position = [solid_container.current_coordinates[:, i]
                              for i in TrixiParticles.each_moving_particle(solid_container)
                              if isapprox(solid_container.initial_coordinates[2, i],
                                          -centroidal_fiber_position)]

    neutral_fiber_position = hcat(neutral_fiber_position...)
    neutral_fiber_position[2, :] .+= centroidal_fiber_position

    plot(neutral_fiber_position[1, :], neutral_fiber_position[2, :], xlims=(0, L),
         xlabel="x", ylabel="y", lab="sph")
    plot!(v, lab="analytical")
end
