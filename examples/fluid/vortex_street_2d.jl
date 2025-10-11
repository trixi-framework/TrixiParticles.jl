# ==========================================================================================
# 2D Vortex Street
#
# Flow past a circular cylinder (vortex street), Tafuni et al. (2018).
# Other literature using this validation:
# Vacandio et al. (2013), Marrone et al. (2013), Calhoun (2002), Liu et al. (1998)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
factor_d = 0.08 # Resolution in the paper is `0.01` (5M particles)

const cylinder_diameter = 0.1
particle_spacing = factor_d * cylinder_diameter

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 8

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 5.0)

# Boundary geometry and initial fluid particle positions
domain_size = (25 * cylinder_diameter, 20 * cylinder_diameter)

flow_direction = [1.0, 0.0]
reynolds_number = 200
const prescribed_velocity = 1.0
const fluid_density = 1000.0
# Maximum velocity observed in the vortex street is typically around 1.5
v_max = 1.5
sound_speed = 10 * v_max

boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                 domain_size[2])

pipe = RectangularTank(particle_spacing, domain_size, boundary_size, fluid_density,
                       n_layers=boundary_layers, velocity=[prescribed_velocity, 0.0],
                       faces=(false, false, true, true))

# Shift pipe walls in negative x-direction for the inflow
pipe.boundary.coordinates[1, :] .-= particle_spacing * open_boundary_layers

n_buffer_particles = 20 * pipe.n_particles_per_dimension[2]^(ndims(pipe.fluid) - 1)

cylinder_center = (5 * cylinder_diameter, domain_size[2] / 2)
cylinder = SphereShape(particle_spacing, cylinder_diameter / 2,
                       cylinder_center, fluid_density, sphere_type=RoundSphere())

fluid = setdiff(pipe.fluid, cylinder)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = prescribed_velocity * cylinder_diameter / reynolds_number

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)
viscosity = ViscosityAdami(nu=kinematic_viscosity)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

# shifting_technique = TransportVelocityAdami(background_pressure=5 * fluid_density *
#                                                                 sound_speed^2)

shifting_technique = ParticleShiftingTechnique(; sound_speed_factor=0.2, v_max_factor=0)

fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           density_diffusion=density_diffusion,
                                           smoothing_length, viscosity=viscosity,
                                           pressure_acceleration=tensile_instability_control,
                                           shifting_technique=shifting_technique,
                                           buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelMirroringTafuni()
# open_boundary_model = BoundaryModelDynamicalPressureZhang()

inflow = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, domain_size[2]]),
                      face_normal=flow_direction, open_boundary_layers,
                      reference_velocity=(pos, t) -> SVector(prescribed_velocity, 0.0),
                      density=fluid_density, particle_spacing)

outflow = BoundaryZone(;
                       boundary_face=([pipe.fluid_size[1], 0.0],
                                      [pipe.fluid_size[1], pipe.fluid_size[2]]),
                       face_normal=(-flow_direction), particle_spacing,
                       density=fluid_density, open_boundary_layers)

# While the outlet velocity is not explicitly prescribed,
# initializing it in the x-direction helps to suppress initial pressure waves.
outflow.initial_condition.velocity[1, :] .= prescribed_velocity

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   pressure_acceleration=TrixiParticles.inter_particle_averaged_pressure,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
boundary_model = BoundaryModelDummyParticles(pipe.boundary.density, pipe.boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             smoothing_kernel, smoothing_length)

boundary_system_wall = WallBoundarySystem(pipe.boundary, boundary_model)

boundary_model_cylinder = BoundaryModelDummyParticles(cylinder.density, cylinder.mass,
                                                      AdamiPressureExtrapolation(),
                                                      state_equation=state_equation,
                                                      viscosity=viscosity,
                                                      smoothing_kernel, smoothing_length)

boundary_system_cylinder = WallBoundarySystem(cylinder, boundary_model_cylinder)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(pipe.boundary.coordinates .- 5 * particle_spacing, dims=2)
max_corner = maximum(pipe.boundary.coordinates .+ 5 * particle_spacing, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner)

neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, boundary_system_wall,
                          boundary_system_cylinder; neighborhood_search=neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

saving_callback = SolutionSavingCallback(; dt=0.02, prefix="", output_directory="out")

extra_callback = nothing

callbacks = CallbackSet(info_callback, UpdateCallback(), saving_callback, extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
