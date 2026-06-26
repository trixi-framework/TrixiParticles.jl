# ==========================================================================================
# 2D Vortex Street
#
# Flow past a circular cylinder (vortex street), Tafuni et al. (2018).
# Other literature using this validation:
# Vacondio et al. (2013), Marrone et al. (2013), Calhoun (2002), Liu et al. (1998)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
factor_d = 0.1 # Resolution in the paper is `0.01` (5M particles)

cylinder_diameter = 0.1
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
tspan = (0.0, 4.0)

# Boundary geometry and initial fluid particle positions
# For better results the domain_size should be increased to
# domain_size = (25 * cylinder_diameter, 20 * cylinder_diameter)
domain_size = (20 * cylinder_diameter, 10 * cylinder_diameter)
open_boundary_size = (particle_spacing * open_boundary_layers, domain_size[2])

flow_direction = SVector(1.0, 0.0)
reynolds_number = 200
prescribed_velocity = 1.0
fluid_density = 1000.0
# Maximum velocity observed in the vortex street is typically around 1.5
v_max = 1.5
sound_speed = 10 * v_max

pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density;
                       n_layers=boundary_layers,
                       velocity=prescribed_velocity * flow_direction,
                       faces=(false, false, true, true),
                       coordinates_eltype=Float64)

min_coords_inlet = (-open_boundary_layers * particle_spacing, 0.0)
inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density; n_layers=boundary_layers,
                        velocity=prescribed_velocity * flow_direction,
                        min_coordinates=min_coords_inlet,
                        faces=(false, false, true, true),
                        coordinates_eltype=Float64)

min_coords_outlet = (pipe.fluid_size[1], 0.0)
outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density; n_layers=boundary_layers,
                         velocity=prescribed_velocity * flow_direction,
                         min_coordinates=min_coords_outlet,
                         faces=(false, false, true, true),
                         coordinates_eltype=Float64)

n_buffer_particles = 20 * pipe.n_particles_per_dimension[2]

cylinder_center = (5 * cylinder_diameter, domain_size[2] / 2)
cylinder = SphereShape(particle_spacing, cylinder_diameter / 2,
                       cylinder_center, fluid_density; sphere_type=RoundSphere())

fluid = setdiff(pipe.fluid, cylinder)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = maximum(prescribed_velocity) * cylinder_diameter / reynolds_number

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)
viscosity = ViscosityAdami(nu=kinematic_viscosity)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

shifting_technique = ParticleShiftingTechnique(; sound_speed_factor=0.2, v_max_factor=0)

fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           density_calculator=fluid_density_calculator,
                                           state_equation, density_diffusion, viscosity,
                                           pressure_acceleration=tensile_instability_control,
                                           shifting_technique,
                                           buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())

face_in = ([0.0, 0.0], [0.0, domain_size[2]])
inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                      open_boundary_layers, density=fluid_density, particle_spacing,
                      reference_velocity=prescribed_velocity * flow_direction,
                      initial_condition=inlet.fluid, boundary_type=InFlow())

face_out = ([min_coords_outlet[1], 0.0], [min_coords_outlet[1], domain_size[2]])
outflow = BoundaryZone(; boundary_face=face_out, face_normal=(-flow_direction),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       initial_condition=outlet.fluid, boundary_type=OutFlow())

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   pressure_acceleration=TrixiParticles.inter_particle_averaged_pressure,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
wall = union(pipe.boundary, inlet.boundary, outlet.boundary)
boundary_model_wall = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                  AdamiPressureExtrapolation(),
                                                  smoothing_kernel, smoothing_length;
                                                  state_equation)

boundary_system_wall = WallBoundarySystem(wall, boundary_model_wall)

boundary_model_cylinder = BoundaryModelDummyParticles(cylinder.density, cylinder.mass,
                                                      AdamiPressureExtrapolation(),
                                                      smoothing_kernel, smoothing_length;
                                                      state_equation, viscosity)

boundary_system_cylinder = WallBoundarySystem(cylinder, boundary_model_cylinder)

# ==========================================================================================
# ==== Simulation
boundary = union(wall, cylinder)
min_corner = minimum(boundary.coordinates .- particle_spacing, dims=2)
max_corner = maximum(boundary.coordinates .+ particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{2}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, boundary_system_wall,
                          boundary_system_cylinder; neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # May need tuning to prevent boundary penetration
            reltol=1e-4, # May need tuning to prevent boundary penetration
            save_everystep=false, callback=callbacks);
