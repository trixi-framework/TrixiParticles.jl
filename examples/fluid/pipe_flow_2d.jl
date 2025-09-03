# ==========================================================================================
# 2D Pipe Flow Simulation with Open Boundaries (Inflow/Outflow)
#
# This example simulates fluid flow through a 2D pipe (channel) with an inflow
# boundary condition on one end and an outflow boundary condition on the other.
# Solid walls form the top and bottom of the pipe.
# The simulation demonstrates the use of open boundary conditions in TrixiParticles.jl.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.05

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 8

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
domain_size = (1.0, 0.4)

flow_direction = [1.0, 0.0]
reynolds_number = 100
const prescribed_velocity = 2.0

boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                 domain_size[2])

fluid_density = 1000.0

# For this particular example, it is necessary to have a background pressure.
# Otherwise the suction at the outflow is to big and the simulation becomes unstable.
pressure = 1000.0

sound_speed = 20 * prescribed_velocity

state_equation = nothing

pipe = RectangularTank(particle_spacing, domain_size, boundary_size, fluid_density,
                       pressure=pressure, n_layers=boundary_layers,
                       faces=(false, false, true, true))

# Shift pipe walls in negative x-direction for the inflow
pipe.boundary.coordinates[1, :] .-= particle_spacing * open_boundary_layers

NDIMS = ndims(pipe.fluid)

n_buffer_particles = 5 * pipe.n_particles_per_dimension[2]^(NDIMS - 1)

# ==========================================================================================
# ==== Fluid
wcsph = false

smoothing_length = 1.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{NDIMS}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = prescribed_velocity * domain_size[2] / reynolds_number

viscosity = ViscosityAdami(nu=kinematic_viscosity)

fluid_system = EntropicallyDampedSPHSystem(pipe.fluid, smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity=viscosity,
                                           density_calculator=fluid_density_calculator,
                                           buffer_size=n_buffer_particles)

# Alternatively the WCSPH scheme can be used
if wcsph
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                       exponent=1, background_pressure=pressure)
    alpha = 8 * kinematic_viscosity / (smoothing_length * sound_speed)
    viscosity = ArtificialViscosityMonaghan(; alpha, beta=0.0)

    fluid_system = WeaklyCompressibleSPHSystem(pipe.fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity=viscosity,
                                               buffer_size=n_buffer_particles)
end

# ==========================================================================================
# ==== Open Boundary

function velocity_function2d(pos, t)
    # Use this for a time-dependent inflow velocity
    # return SVector(0.5prescribed_velocity * sin(2pi * t) + prescribed_velocity, 0)

    return SVector(prescribed_velocity, 0.0)
end

open_boundary_model = BoundaryModelLastiwka()

boundary_type_in = InFlow()
plane_in = ([0.0, 0.0], [0.0, domain_size[2]])
inflow = BoundaryZone(; plane=plane_in, plane_normal=flow_direction, open_boundary_layers,
                      density=fluid_density, particle_spacing,
                      boundary_type=boundary_type_in)

reference_velocity_in = velocity_function2d
reference_pressure_in = pressure
reference_density_in = fluid_density
open_boundary_in = OpenBoundarySPHSystem(inflow; fluid_system,
                                         boundary_model=open_boundary_model,
                                         buffer_size=n_buffer_particles,
                                         reference_density=reference_density_in,
                                         reference_pressure=reference_pressure_in,
                                         reference_velocity=reference_velocity_in)

boundary_type_out = OutFlow()
plane_out = ([domain_size[1], 0.0], [domain_size[1], domain_size[2]])
outflow = BoundaryZone(; plane=plane_out, plane_normal=(-flow_direction),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       boundary_type=boundary_type_out)

reference_velocity_out = velocity_function2d
reference_pressure_out = pressure
reference_density_out = fluid_density
open_boundary_out = OpenBoundarySPHSystem(outflow; fluid_system,
                                          boundary_model=open_boundary_model,
                                          buffer_size=n_buffer_particles,
                                          reference_density=reference_density_out,
                                          reference_pressure=reference_pressure_out,
                                          reference_velocity=reference_velocity_out)
# ==========================================================================================
# ==== Boundary
viscosity_boundary = ViscosityAdami(nu=1e-4)
boundary_model = BoundaryModelDummyParticles(pipe.boundary.density, pipe.boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             viscosity=viscosity_boundary,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(pipe.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(pipe.boundary.coordinates .- particle_spacing, dims=2)
max_corner = maximum(pipe.boundary.coordinates .+ particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{NDIMS}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                    update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary_in, open_boundary_out,
                          boundary_system, neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")
particle_shifting = ParticleShiftingCallback()

extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(),
                        particle_shifting, extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
