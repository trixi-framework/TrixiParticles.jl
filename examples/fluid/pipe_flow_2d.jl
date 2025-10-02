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
particle_spacing = 0.02

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 6

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
domain_size = (1.0, 0.4)

reynolds_number = 100
const prescribed_velocity = (1.0, 0.0)
flow_direction = [1.0, 0.0]

open_boundary_size = (particle_spacing * open_boundary_layers, domain_size[2])

fluid_density = 1000.0

sound_speed = 10 * maximum(abs.(prescribed_velocity))

pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density,
                       n_layers=boundary_layers, velocity=prescribed_velocity,
                       faces=(false, false, true, true))

min_coords_inlet = (-open_boundary_layers * particle_spacing, 0.0)
inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density, n_layers=boundary_layers,
                        min_coordinates=min_coords_inlet,
                        faces=(false, false, true, true))

min_coords_outlet = (pipe.fluid_size[1], 0.0)
outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density, n_layers=boundary_layers,
                         min_coordinates=min_coords_outlet,
                         faces=(false, false, true, true))

NDIMS = ndims(pipe.fluid)

n_buffer_particles = 10 * pipe.n_particles_per_dimension[2]^(NDIMS - 1)

# ==========================================================================================
# ==== Fluid
wcsph = true

smoothing_length = 1.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{NDIMS}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = maximum(prescribed_velocity) * domain_size[2] / reynolds_number

viscosity = ViscosityAdami(nu=kinematic_viscosity)

# Alternatively the WCSPH scheme can be used
if wcsph
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                       exponent=1)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

    fluid_system = WeaklyCompressibleSPHSystem(pipe.fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               density_diffusion=density_diffusion,
                                               smoothing_length, viscosity=viscosity,
                                               shifting_technique=ParticleShiftingTechnique(v_max_factor=1.5),
                                               buffer_size=n_buffer_particles)
else
    # Alternatively the EDAC scheme can be used
    state_equation = nothing

    fluid_system = EntropicallyDampedSPHSystem(pipe.fluid, smoothing_kernel,
                                               smoothing_length, sound_speed,
                                               viscosity=viscosity,
                                               density_calculator=fluid_density_calculator,
                                               shifting_technique=ParticleShiftingTechnique(),
                                               buffer_size=n_buffer_particles)
end

# ==========================================================================================
# ==== Open Boundary

function velocity_function2d(pos, t)
    # Use this for a time-dependent inflow velocity
    # return SVector(0.5prescribed_velocity * sin(2pi * t) + prescribed_velocity, 0)

    return SVector(prescribed_velocity)
end

open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())

reference_velocity_in = velocity_function2d
reference_pressure_in = nothing
reference_density_in = nothing
boundary_type_in = InFlow()
face_in = ([0.0, 0.0], [0.0, domain_size[2]])
inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                      open_boundary_layers, density=fluid_density, particle_spacing,
                      reference_density=reference_density_in,
                      reference_pressure=reference_pressure_in,
                      reference_velocity=reference_velocity_in,
                      initial_condition=inlet.fluid, boundary_type=boundary_type_in)

reference_velocity_out = nothing
reference_pressure_out = nothing
reference_density_out = nothing
boundary_type_out = OutFlow()
face_out = ([min_coords_outlet[1], 0.0], [min_coords_outlet[1], domain_size[2]])
outflow = BoundaryZone(; boundary_face=face_out, face_normal=(-flow_direction),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       reference_density=reference_density_out,
                       reference_pressure=reference_pressure_out,
                       reference_velocity=reference_velocity_out,
                       initial_condition=outlet.fluid, boundary_type=boundary_type_out)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
wall = union(pipe.boundary, inlet.boundary, outlet.boundary)
viscosity_boundary = viscosity
boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             viscosity=viscosity_boundary,
                                             smoothing_kernel, smoothing_length)

boundary_system = WallBoundarySystem(wall, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(wall.coordinates .- particle_spacing, dims=2)
max_corner = maximum(wall.coordinates .+ particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{NDIMS}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                    update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, boundary_system,
                          neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
