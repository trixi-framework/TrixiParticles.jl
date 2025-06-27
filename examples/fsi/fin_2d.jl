using TrixiParticles
using OrdinaryDiffEqLowStorageRK
# using OrdinaryDiffEqSymplecticRK

# ==========================================================================================
# ==== Resolution
n_particles_y = 6

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 3.0)

fin_length = 0.6
fin_thickness = 10e-3
flexural_rigidity = 6.0
poisson_ratio = 0.3
modulus = 12 * (1 - poisson_ratio^2) * flexural_rigidity / (fin_thickness^3)

fiber_volume_fraction = 0.6
fiber_density = 1800.0
epoxy_density = 1250.0
density = fiber_volume_fraction * fiber_density +
          (1 - fiber_volume_fraction) * epoxy_density

clamp_radius = 0.05

tank_size = (2.0, 1.0)
center = (tank_size[2] / 2, tank_size[2] / 2)
initial_fluid_size = tank_size
initial_velocity = (1.0, 0.0)

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = fin_thickness / (n_particles_y - 1)
fluid_particle_spacing = particle_spacing

smoothing_length_solid = sqrt(2) * particle_spacing
smoothing_length_fluid = 2 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

# Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius
# fixed_particles = SphereShape(particle_spacing, clamp_radius + particle_spacing / 2,
#                               (center[1] - clamp_radius - particle_spacing, center[2] + fin_thickness / 2), density,
#                             #   cutout_min=(center[1], center[2] - fin_thickness / 2),
#                             #   cutout_max=center .+ (clamp_radius, fin_thickness / 2),
#                               tlsph=true)

# fixed_beam = RectangularShape(particle_spacing, (round(Int, 0.05 / particle_spacing) + 1,# + n_particles_clamp_x,
#                              n_particles_y),
#                               center, density=density, tlsph=true)

# fixed_particles = union(fixed_particles, fixed_beam)

# n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)

using TrixiParticles
using PythonCall
using CondaPkg

CondaPkg.add("svgpathtools")
CondaPkg.add("ezdxf")
svgpath = pyimport("svgpathtools")

svg_path = "M509.299 100.016C507.966 91.5871 505.915 85.7111 503.145 82.3879C498.991 77.4031 479.883 60.7871 475.521 58.2947C471.16 55.8023 455.167 49.1559 448.936 47.702C442.705 46.2481 355.471 30.463 339.893 28.8014C329.508 27.6937 287.761 23.5397 214.651 16.3394C199.727 15.7768 185.488 15.1656 171.934 14.5056C151.602 13.5157 106.318 5.19544 82.4982 0.830064C58.6781 -3.53532 3.26262 9.37214 0.279574 38.2664C-2.54326 65.6089 16.5085 89.4186 34.9345 99.2843C49.9801 107.34 58.7166 107.403 71.5338 114.275C101.912 130.564 100.849 169.8 123.353 175.939C145.856 182.078 146.637 180.9 155.986 179.279C162.22 178.199 199.267 168.32 267.129 149.644L359.355 117.398L405.801 102.863C446.849 101.008 472.412 100.061 482.489 100.022C497.605 99.9635 507.524 100.062 509.299 100.016Z"

path = svgpath.parse_path(svg_path)
t_range = range(0, 1, length=50 * length(path))
points = [(pyconvert(Float64, p.real), -pyconvert(Float64, p.imag))
          for p in (path.point(t) for t in t_range)]

# ezdxf = pyimport("ezdxf")
# doc = ezdxf.new(dxfversion="R2010")
# msp = doc.modelspace()
# msp.add_polyline2d(points)
# doc.saveas("output.dxf")

points_matrix = reinterpret(reshape, Float64, points)
scaling = 0.3 / maximum(points_matrix, dims=2)[1]
points_matrix .*= scaling
points_matrix .+= (-0.3, -points_matrix[2, 1]) .+ center .- (0.0, 1e-4)
# # Clamp the blade in one layer of particles by moving the foot down by a particle spacing
# points_matrix .-= (0.0, particle_spacing)
geometry = TrixiParticles.Polygon(points_matrix)

# trixi2vtk(geometry)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    winding_number_factor=0.4,
                                                    hierarchical_winding=true)

# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density=density,
                             grid_offset=center, point_in_geometry_algorithm)

# Beam and clamped particles
length_clamp = round(Int, 0.15 / particle_spacing) * particle_spacing # m
n_particles_per_dimension = (round(Int, (fin_length + length_clamp) / particle_spacing) + 2,# + n_particles_clamp_x,
                             n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for solids.
# We therefore need to pass `tlsph=true`.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (center[1] - length_clamp, center[2]), density=density, tlsph=true)

fixed_particles = setdiff(shape_sampled, beam)

# solid = union(beam, fixed_particles)

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1
fluid_density = 1000.0
tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(false, false, true, true), velocity=initial_velocity)
# fluid = setdiff(tank.fluid, solid)

# ==========================================================================================
# ==== Packing
foot_sdf = SignedDistanceField(geometry, particle_spacing;
                               max_signed_distance=4 * particle_spacing,
                               use_for_boundary_packing=true)

boundary_packing = sample_boundary(foot_sdf; boundary_density=density,
                                   boundary_thickness=4 * particle_spacing)
boundary_packing = setdiff(boundary_packing, beam)

background_pressure = 1.0
smoothing_length_packing = 0.8 * particle_spacing
foot_packing_system = ParticlePackingSystem(fixed_particles; smoothing_length=smoothing_length_packing,
                                            signed_distance_field=foot_sdf, background_pressure)

fluid_packing_system = ParticlePackingSystem(boundary_packing; smoothing_length=smoothing_length_packing,
                                             signed_distance_field=foot_sdf, is_boundary=true, background_pressure,
                                             boundary_compress_factor=0.8)

blade_packing_system = ParticlePackingSystem(beam; smoothing_length=smoothing_length_packing,
                                             fixed_system=true, signed_distance_field=nothing, background_pressure)

min_corner = minimum(tank.boundary.coordinates, dims=2) .- fluid_particle_spacing / 2
max_corner = maximum(tank.boundary.coordinates, dims=2) .+ fluid_particle_spacing / 2
periodic_box = PeriodicBox(; min_corner, max_corner)
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box, cell_list, update_strategy=ParallelUpdate())

semi_packing = Semidiscretization(foot_packing_system, fluid_packing_system,
                                  blade_packing_system; neighborhood_search)

ode_packing = semidiscretize(semi_packing, (0.0, 10.0))

sol_packing = solve(ode_packing, RDPK3SpFSAL35();
            save_everystep=false,
            callback=CallbackSet(InfoCallback(interval=50),
                                #  SolutionSavingCallback(interval=50, prefix="packing"),
                                 UpdateCallback()),
            dtmax=1e-2)

packed_foot = InitialCondition(sol_packing, foot_packing_system, semi_packing)
solid = union(beam, packed_foot)
fluid = setdiff(tank.fluid, solid)

n_fixed_particles = nparticles(solid) - nparticles(beam)

fluid_packing_system = ParticlePackingSystem(fluid; smoothing_length=smoothing_length_packing,
                                             signed_distance_field=nothing, background_pressure)

packing_boundary = union(solid, tank.boundary)
boundary_packing_system = ParticlePackingSystem(packing_boundary; smoothing_length=smoothing_length_packing,
                                                fixed_system=true, signed_distance_field=nothing, background_pressure)

semi_packing = Semidiscretization(fluid_packing_system, boundary_packing_system;
                                  neighborhood_search)

ode_packing = semidiscretize(semi_packing, (0.0, 5.0))

sol_packing = solve(ode_packing, RDPK3SpFSAL35();
            save_everystep=false,
            callback=CallbackSet(InfoCallback(interval=50),
                                #  SolutionSavingCallback(interval=50, prefix="packing"),
                                 UpdateCallback()),
            dtmax=1e-2)

fluid = InitialCondition(sol_packing, fluid_packing_system, semi_packing)

# Movement function
frequency = 1.3 # Hz
amplitude = 0.18 # m
rotation_deg = 25 # degrees
rotation_phase_offset = 0.12 # periods
translation_vector = SVector(0.0, amplitude)
rotation_angle = rotation_deg * pi / 180

boundary_movement = TrixiParticles.oscillating_movement(frequency,
                                                        SVector(0.0, amplitude),
                                                        rotation_angle, center;
                                                        rotation_phase_offset, ramp_up=0.5)

sound_speed = 20.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, background_pressure=0.0)

# ==========================================================================================
# ==== Solid
boundary_density_calculator = AdamiPressureExtrapolation()
viscosity_fluid = ViscosityAdami(nu=1e-4)
viscosity_fin = ViscosityAdami(nu=1e-4)

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites = fluid_density * ones(size(solid.density))
hydrodynamic_masses = hydrodynamic_densites * particle_spacing^2

boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densites,
                                                   hydrodynamic_masses,
                                                   state_equation=state_equation,
                                                   boundary_density_calculator,
                                                   smoothing_kernel, smoothing_length_fluid,
                                                   viscosity=viscosity_fin)

# k_solid = 1.0
# beta_solid = fluid_particle_spacing / particle_spacing
# boundary_model_solid = BoundaryModelMonaghanKajtar(k_solid, beta_solid,
#                                                    particle_spacing,
#                                                    hydrodynamic_masses)

solid_system = TotalLagrangianSPHSystem(solid, smoothing_kernel, smoothing_length_solid,
                                        modulus, poisson_ratio;
                                        n_fixed_particles, movement=boundary_movement,
                                        boundary_model=boundary_model_solid,
                                        penalty_force=PenaltyForceGanzenmueller(alpha=0.1))

# ==========================================================================================
# ==== Fluid
fluid_density_calculator = ContinuityDensity()
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
# density_diffusion = DensityDiffusionAntuono(fluid, delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length_fluid, viscosity=viscosity_fluid,
                                           density_diffusion=density_diffusion,
                                           particle_shifting=TrixiParticles.ParticleShiftingSun2019(0.1 * sound_speed),
                                           pressure_acceleration=tensile_instability_control)
# fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
#                                            sound_speed, viscosity=ViscosityAdami(; nu),
#                                            transport_velocity=TransportVelocityAdami(10 * sound_speed^2 * fluid_density))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length_fluid)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(tank.boundary.coordinates, dims=2) .- fluid_particle_spacing / 2
max_corner = maximum(tank.boundary.coordinates, dims=2) .+ fluid_particle_spacing / 2
periodic_box = PeriodicBox(; min_corner, max_corner)
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box, cell_list, update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, boundary_system, solid_system; neighborhood_search,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01, prefix="")

split_dt = 2e-5
split_integration = SplitIntegrationCallback(RDPK3SpFSAL35(), adaptive=false, dt=split_dt,
                                             maxiters=10^8)
stepsize_callback = StepsizeCallback(cfl=0.5)
shifting_callback = ParticleShiftingCallback()
callbacks = CallbackSet(info_callback, saving_callback, shifting_callback,
                        split_integration, stepsize_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
# sol = solve(ode, RDPK3SpFSAL35(),
#             abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
#             reltol=1e-6, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
#             dtmax=1e-2, # Limit stepsize to prevent crashing
#             save_everystep=false, callback=callbacks, maxiters=10^8);

dt_fluid = 1.25e-4
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=dt_fluid, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks, maxiters=10^8);
