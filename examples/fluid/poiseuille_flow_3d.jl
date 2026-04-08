# ==========================================================================================
# 3D Hagen-Poiseuille Flow Simulation (Weakly Compressible SPH)
#
# Based on:
#   Zhan, X., et al. "Dynamical pressure boundary condition for weakly compressible smoothed particle hydrodynamics"
#   Physics of Fluids, Volume 37
#   https://doi.org/10.1063/5.0254575
#
# This example sets up a 3D Hagen-Poiseuille flow simulation in a circular pipe
# including open boundary conditions.
# ==========================================================================================
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
channel_length = 0.004
channel_radius = 0.0005
channel_diameter = 2 * channel_radius

particle_spacing_factor = 15
particle_spacing = channel_diameter / particle_spacing_factor

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
open_boundary_layers = 10

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 1.0)

fluid_density = 1000.0
reynolds_number = 50
imposed_pressure_drop = 0.1
dynamic_viscosity = sqrt(fluid_density * channel_radius^2 * imposed_pressure_drop /
                         (4 * reynolds_number))

v_max = channel_radius^2 * imposed_pressure_drop / (4 * dynamic_viscosity * channel_length)

# To accurately capture the initial transient this value has to be increased to around 100 which doubles the runtime
sound_speed_factor = 50
sound_speed = sound_speed_factor * v_max

flow_direction = (1.0, 0.0, 0.0)
outlet_flow_direction = (-flow_direction[1], -flow_direction[2], -flow_direction[3])

n_particles_length = ceil(Int, channel_length / particle_spacing)

wall_cross_section = SphereShape(particle_spacing, channel_radius, (0.0, 0.0),
                                 fluid_density, n_layers=boundary_layers,
                                 layer_outwards=true, sphere_type=RoundSphere())

# Extend 2D coordinates to 3D by adding x-coordinates
wall_cross_section_coordinates = hcat(particle_spacing / 2 *
                                      ones(nparticles(wall_cross_section)),
                                      wall_cross_section.coordinates')'

wall_boundary = extrude_geometry(wall_cross_section_coordinates; particle_spacing,
                                 direction=collect(flow_direction),
                                 density=fluid_density,
                                 n_extrude=n_particles_length)

fluid_cross_section = SphereShape(particle_spacing, channel_radius, (0.0, 0.0),
                                  fluid_density, sphere_type=RoundSphere())

# Extend 2D coordinates to 3D by adding x-coordinates
fluid_cross_section_coordinates = hcat(particle_spacing / 2 *
                                       ones(nparticles(fluid_cross_section)),
                                       fluid_cross_section.coordinates')'

inlet_particles = extrude_geometry(fluid_cross_section_coordinates; particle_spacing,
                                   direction=collect(flow_direction),
                                   density=fluid_density,
                                   n_extrude=open_boundary_layers)

fluid_offset = [open_boundary_layers * particle_spacing, 0, 0]
fluid_particles = extrude_geometry(fluid_cross_section_coordinates .+ fluid_offset;
                                   particle_spacing,
                                   direction=collect(flow_direction),
                                   density=fluid_density,
                                   n_extrude=(n_particles_length - 2 * open_boundary_layers))

outlet_offset = [(n_particles_length - open_boundary_layers) * particle_spacing, 0, 0]
outlet_particles = extrude_geometry(fluid_cross_section_coordinates .+ outlet_offset;
                                    particle_spacing,
                                    direction=collect(flow_direction),
                                    n_extrude=open_boundary_layers,
                                    density=fluid_density)

n_buffer_particles = 10 * nparticles(fluid_cross_section)

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * particle_spacing
smoothing_kernel = WendlandC2Kernel{3}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = dynamic_viscosity / fluid_density
viscosity = ViscosityAdami(nu=kinematic_viscosity)

background_pressure = 7 * sound_speed_factor / 10 * fluid_density * v_max^2
shifting_technique = TransportVelocityAdami(; background_pressure)

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

fluid_system = WeaklyCompressibleSPHSystem(fluid_particles, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           buffer_size=n_buffer_particles,
                                           shifting_technique=shifting_technique,
                                           density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                           smoothing_length, viscosity=viscosity)

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelDynamicalPressureZhang()

inlet_boundary_type = BidirectionalFlow()
inlet_face = ([
                  open_boundary_layers * particle_spacing,
                  -channel_diameter,
                  -channel_diameter
              ],
              [
                  open_boundary_layers * particle_spacing,
                  -channel_diameter,
                  channel_diameter
              ],
              [open_boundary_layers * particle_spacing, channel_diameter, channel_diameter])
inlet_reference_velocity = nothing
inlet_reference_pressure = 0.2

inlet_zone = BoundaryZone(; boundary_face=inlet_face, face_normal=flow_direction,
                          open_boundary_layers, density=fluid_density, particle_spacing,
                          reference_velocity=inlet_reference_velocity,
                          reference_pressure=inlet_reference_pressure,
                          initial_condition=inlet_particles,
                          boundary_type=inlet_boundary_type)

outlet_boundary_type = BidirectionalFlow()
outlet_face = ([outlet_offset[1], -channel_diameter, -channel_diameter],
               [outlet_offset[1], -channel_diameter, channel_diameter],
               [outlet_offset[1], channel_diameter, channel_diameter])
outlet_reference_velocity = nothing
outlet_reference_pressure = 0.1

outlet_zone = BoundaryZone(; boundary_face=outlet_face,
                           face_normal=outlet_flow_direction,
                           open_boundary_layers, density=fluid_density, particle_spacing,
                           reference_velocity=outlet_reference_velocity,
                           reference_pressure=outlet_reference_pressure,
                           initial_condition=outlet_particles,
                           boundary_type=outlet_boundary_type)

open_boundary = OpenBoundarySystem(inlet_zone, outlet_zone; fluid_system,
                                   boundary_model=open_boundary_model,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
boundary_model = BoundaryModelDummyParticles(wall_boundary.density, wall_boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             viscosity=viscosity,
                                             smoothing_kernel, smoothing_length)

boundary_system = WallBoundarySystem(wall_boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(wall_boundary.coordinates .- 2 * particle_spacing, dims=2)
max_corner = maximum(wall_boundary.coordinates .+ 2 * particle_spacing, dims=2)

neighborhood_search = GridNeighborhoodSearch{3}(;
                                                cell_list=FullGridCellList(; min_corner,
                                                                           max_corner),
                                                update_strategy=ParallelUpdate())

semi_discretization = Semidiscretization(fluid_system, open_boundary,
                                         boundary_system,
                                         neighborhood_search=neighborhood_search,
                                         parallelization_backend=PolyesterBackend())

ode_problem = semidiscretize(semi_discretization, tspan)

info_callback = InfoCallback(interval=20)
saving_callback = SolutionSavingCallback(dt=0.01, prefix="", output_directory="out")
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback)

sol = solve(ode_problem, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need tuning to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need tuning to prevent boundary penetration)
            dtmax=1e-2,  # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks)
