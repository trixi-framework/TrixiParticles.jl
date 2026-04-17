# ==========================================================================================
# 2D Poiseuille Flow Simulation (Weakly Compressible SPH)
#
# Based on:
#   Zhan, X., et al. "Dynamical pressure boundary condition for weakly compressible smoothed particle hydrodynamics"
#   Physics of Fluids, Volume 37
#   https://doi.org/10.1063/5.0254575
#
# This example sets up a 2D Poiseuille flow simulation in a rectangular channel
# including open boundary conditions.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
channel_height = 0.001 # distance between top and bottom walls
channel_length = 0.004 # distance between inlet and outlet

particle_spacing = channel_height / 30

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
open_boundary_layers = 10

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 1.0)
use_wcsph = true

domain_size = (channel_length, channel_height)

open_boundary_size = (open_boundary_layers * particle_spacing, domain_size[2])

fluid_density = 1000.0
reynolds_number = 50
imposed_pressure_drop = 0.1
outlet_pressure = 0.1
inlet_pressure = outlet_pressure + imposed_pressure_drop
const dynamic_viscosity = sqrt(fluid_density * channel_height^3 * imposed_pressure_drop /
                               (8 * channel_length * reynolds_number))

v_max = channel_height^2 * imposed_pressure_drop / (8 * dynamic_viscosity * channel_length)

sound_speed_factor = 100
sound_speed = sound_speed_factor * v_max

flow_direction = (1.0, 0.0)

channel = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density,
                          pressure=(pos) -> outlet_pressure +
                                            imposed_pressure_drop *
                                            (1 - (pos[1] / channel_length)),
                          n_layers=boundary_layers, faces=(false, false, true, true),
                          coordinates_eltype=Float64)

# The analytical solution uses the full inflow-to-outflow extent x in [0, channel_length].
# Since the pressure is prescribed at the free surface of each `BoundaryZone` instead of at
# its `boundary_face`, the inlet and outlet free surfaces lie at x = 0 and
# x = channel_length (= L), respectively. Thus, the boundary zones are effectively part of
# the computational domain in this setup.
inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density, n_layers=boundary_layers, pressure=inlet_pressure,
                        min_coordinates=(0.0, 0.0), faces=(false, false, true, true),
                        coordinates_eltype=Float64)

outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density, n_layers=boundary_layers,
                         min_coordinates=(channel.fluid_size[1] - open_boundary_size[1],
                                          0.0),
                         faces=(false, false, true, true),
                         coordinates_eltype=Float64)

fluid = setdiff(channel.fluid, inlet.fluid, outlet.fluid)

n_buffer_particles = 10 * channel.n_particles_per_dimension[2]^2

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = dynamic_viscosity / fluid_density

viscosity = ViscosityAdami(nu=kinematic_viscosity)

background_pressure = 7 * sound_speed_factor / 10 * fluid_density * v_max^2
shifting_technique = TransportVelocityAdami(; background_pressure)

if use_wcsph
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                       exponent=1)
    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               buffer_size=n_buffer_particles,
                                               shifting_technique=shifting_technique,
                                               density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                               smoothing_length, viscosity=viscosity)
else
    state_equation = nothing

    fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel,
                                               smoothing_length,
                                               sound_speed, viscosity=viscosity,
                                               density_calculator=fluid_density_calculator,
                                               shifting_technique=shifting_technique,
                                               buffer_size=n_buffer_particles)
end

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelDynamicalPressureZhang()

inlet_boundary_type = BidirectionalFlow()
inlet_face = ([open_boundary_size[1], 0.0], [open_boundary_size[1], channel.fluid_size[2]])
inlet_reference_velocity = nothing
inlet_reference_pressure = 0.2
inlet_boundary_zone = BoundaryZone(; boundary_face=inlet_face, face_normal=flow_direction,
                                   open_boundary_layers, density=fluid_density,
                                   particle_spacing,
                                   reference_velocity=inlet_reference_velocity,
                                   reference_pressure=inlet_reference_pressure,
                                   initial_condition=inlet.fluid,
                                   boundary_type=inlet_boundary_type)

outlet_boundary_type = BidirectionalFlow()
outlet_face = ([channel.fluid_size[1] - open_boundary_size[1], 0.0],
               [channel.fluid_size[1] - open_boundary_size[1], channel.fluid_size[2]])
outlet_reference_velocity = nothing
outlet_reference_pressure = 0.1
outlet_flow_direction = (.-(flow_direction))
outlet_boundary_zone = BoundaryZone(; boundary_face=outlet_face,
                                    face_normal=outlet_flow_direction,
                                    open_boundary_layers, density=fluid_density,
                                    particle_spacing,
                                    reference_velocity=outlet_reference_velocity,
                                    reference_pressure=outlet_reference_pressure,
                                    initial_condition=outlet.fluid,
                                    boundary_type=outlet_boundary_type)

open_boundary = OpenBoundarySystem(inlet_boundary_zone, outlet_boundary_zone; fluid_system,
                                   boundary_model=open_boundary_model,
                                   calculate_flow_rate=true)

# ==========================================================================================
# ==== Boundary
wall_boundary = union(channel.boundary)

wall_boundary_model = BoundaryModelDummyParticles(wall_boundary; fluid_system=fluid_system,
                                                  viscosity=viscosity)
boundary_system = WallBoundarySystem(wall_boundary, wall_boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(wall_boundary.coordinates .- 2 * particle_spacing, dims=2)
max_corner = maximum(wall_boundary.coordinates .+ 2 * particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{2}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary,
                          boundary_system, neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="", output_directory="out")

extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
