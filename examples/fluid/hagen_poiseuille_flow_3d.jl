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
const flow_length = 0.004
const pipe_radius = 0.0005
pipe_diameter = 2 * pipe_radius

particle_spacing_factor = 30
particle_spacing = pipe_diameter / particle_spacing_factor

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
open_boundary_layers = 10

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 1.0)
domain_size = (flow_length, pipe_diameter)

open_boundary_size = (open_boundary_layers * particle_spacing, domain_size[2])

fluid_density = 1000.0
reynolds_number = 50
const pressure_drop = 0.1
const dynamic_viscosity = sqrt(fluid_density * pipe_radius^2 * pressure_drop /
                               (4 * reynolds_number))

v_max = pipe_radius^2 * pressure_drop / (4 * dynamic_viscosity * flow_length)

sound_speed_factor = 100
sound_speed = sound_speed_factor * v_max

flow_direction = (1.0, 0.0, 0.0)

n_particles_length = ceil(Int, flow_length / particle_spacing)
wall_2d = SphereShape(particle_spacing, pipe_radius, (0.0, 0.0),
                      fluid_density, n_layers=boundary_layers, layer_outwards=true,
                      sphere_type=RoundSphere())
# Extend 2d coordinates to 3d by adding x-coordinates
wall_2d_coordinates = hcat(particle_spacing / 2 * ones(nparticles(wall_2d)),
                           wall_2d.coordinates')'

pipe_wall = extrude_geometry(wall_2d_coordinates; particle_spacing,
                             direction=collect(flow_direction), density=fluid_density,
                             n_extrude=n_particles_length)

circle = SphereShape(particle_spacing, pipe_radius, (0.0, 0.0), fluid_density,
                     sphere_type=RoundSphere())
# Extend 2d coordinates to 3d by adding x-coordinates
circle_coordinates = hcat(particle_spacing / 2 * ones(nparticles(circle)),
                          circle.coordinates')'

inlet = extrude_geometry(circle_coordinates; particle_spacing,
                         direction=collect(flow_direction), density=fluid_density,
                         n_extrude=open_boundary_layers)
offset_fluid = [open_boundary_layers * particle_spacing, 0, 0]
fluid = extrude_geometry(circle_coordinates .+ offset_fluid; particle_spacing,
                         direction=collect(flow_direction), density=fluid_density,
                         n_extrude=(n_particles_length - 2 * open_boundary_layers))

offset_outlet = [(n_particles_length - open_boundary_layers) * particle_spacing, 0, 0]
outlet = extrude_geometry(circle_coordinates .+ offset_outlet;
                          particle_spacing, direction=collect(flow_direction),
                          n_extrude=open_boundary_layers, density=fluid_density)

n_buffer_particles = 10 * nparticles(circle)

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
fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           buffer_size=n_buffer_particles,
                                           shifting_technique=shifting_technique,
                                           density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                           smoothing_length, viscosity=viscosity)

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelDynamicalPressureZhang()

boundary_type_in = BidirectionalFlow()
face_in = ([open_boundary_layers * particle_spacing, -pipe_diameter, -pipe_diameter],
           [open_boundary_layers * particle_spacing, -pipe_diameter, pipe_diameter],
           [open_boundary_layers * particle_spacing, pipe_diameter, pipe_diameter])
reference_velocity_in = nothing
reference_pressure_in = 0.2
inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                      open_boundary_layers, density=fluid_density, particle_spacing,
                      reference_velocity=reference_velocity_in,
                      reference_pressure=reference_pressure_in,
                      initial_condition=inlet, boundary_type=boundary_type_in)

boundary_type_out = BidirectionalFlow()
face_out = ([offset_outlet[1], -pipe_diameter, -pipe_diameter],
            [offset_outlet[1], -pipe_diameter, pipe_diameter],
            [offset_outlet[1], pipe_diameter, pipe_diameter])
reference_velocity_out = nothing
reference_pressure_out = 0.1
outflow = BoundaryZone(; boundary_face=face_out, face_normal=(.-(flow_direction)),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       reference_velocity=reference_velocity_out,
                       reference_pressure=reference_pressure_out,
                       initial_condition=outlet, boundary_type=boundary_type_out)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   buffer_size=n_buffer_particles)
# ==========================================================================================
# ==== Boundary

boundary_model = BoundaryModelDummyParticles(pipe_wall.density, pipe_wall.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             viscosity=viscosity,
                                             smoothing_kernel, smoothing_length)

boundary_system = WallBoundarySystem(pipe_wall, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(pipe_wall.coordinates .- 2 * particle_spacing, dims=2)
max_corner = maximum(pipe_wall.coordinates .+ 2 * particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{3}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary,
                          boundary_system, neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=20)
saving_callback = SolutionSavingCallback(dt=0.01, prefix="", output_directory="out")

extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
