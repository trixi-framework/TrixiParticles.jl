using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.005

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
# Boundary geometry and initial fluid particle positions
tank_size = (1.0, 0.5)
# tank_size = (0.6, 0.6)
initial_fluid_size = tank_size

Re = 10000
initial_velocity = (0.0, 0.0)
nu = 1 * 1 / Re

tspan = (0.0, 2.0)

fluid_density = 1000.0
sound_speed = 50.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, background_pressure=0.0)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(false, false, true, true), velocity=initial_velocity)

center = tank_size ./ 2
hollow_sphere = SphereShape(fluid_particle_spacing, 0.1, center, fluid_density,
                            n_layers=4, sphere_type=RoundSphere())

filled_sphere = SphereShape(fluid_particle_spacing, 0.1, center, fluid_density,
                            sphere_type=RoundSphere())

# hollow_sphere = RectangularShape(fluid_particle_spacing, round.(Int, (0.1, 0.3) ./ fluid_particle_spacing), center .- 0.15,
#                                  density=fluid_density)
# filled_sphere = hollow_sphere

# using PythonCall
# using CondaPkg

# CondaPkg.add("svgpathtools")
# CondaPkg.add("ezdxf")
# svgpath = pyimport("svgpathtools")

# svg_path = "M509.299 100.016C507.966 91.5871 505.915 85.7111 503.145 82.3879C498.991 77.4031 479.883 60.7871 475.521 58.2947C471.16 55.8023 455.167 49.1559 448.936 47.702C442.705 46.2481 355.471 30.463 339.893 28.8014C329.508 27.6937 287.761 23.5397 214.651 16.3394C199.727 15.7768 185.488 15.1656 171.934 14.5056C151.602 13.5157 106.318 5.19544 82.4982 0.830064C58.6781 -3.53532 3.26262 9.37214 0.279574 38.2664C-2.54326 65.6089 16.5085 89.4186 34.9345 99.2843C49.9801 107.34 58.7166 107.403 71.5338 114.275C101.912 130.564 100.849 169.8 123.353 175.939C145.856 182.078 146.637 180.9 155.986 179.279C162.22 178.199 199.267 168.32 267.129 149.644L359.355 117.398L405.801 102.863C446.849 101.008 472.412 100.061 482.489 100.022C497.605 99.9635 507.524 100.062 509.299 100.016Z"

# path = svgpath.parse_path(svg_path)
# t_range = range(0, 1, length=50 * length(path))
# points = [(pyconvert(Float64, p.real), -pyconvert(Float64, p.imag))
#           for p in (path.point(t) for t in t_range)]

# # ezdxf = pyimport("ezdxf")
# # doc = ezdxf.new(dxfversion="R2010")
# # msp = doc.modelspace()
# # msp.add_polyline2d(points)
# # doc.saveas("output.dxf")

# center = tank_size ./ 2
# points_matrix = reinterpret(reshape, Float64, points)
# scaling = 0.3 / maximum(points_matrix, dims=2)[1]
# points_matrix .*= scaling
# points_matrix .+= (-0.15, -points_matrix[2, 1])

# geometry = TrixiParticles.Polygon(points_matrix)

# # trixi2vtk(geometry)

# point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
#                                                     winding_number_factor=0.4,
#                                                     hierarchical_winding=true)

# # Returns `InitialCondition`
# shape_sampled = ComplexShape(geometry; particle_spacing=fluid_particle_spacing, density=fluid_density,
#                              store_winding_number=true,
#                              point_in_geometry_algorithm)

# # angle = pi / 4
# # using StaticArrays
# # rotation_matrix = @SMatrix [cos(angle) -sin(angle)
# #                             sin(angle) cos(angle)]
# # shape_sampled.initial_condition.coordinates .= rotation_matrix * shape_sampled.initial_condition.coordinates
# shape_sampled.initial_condition.coordinates .+= center

# hollow_sphere = shape_sampled.initial_condition
# filled_sphere = hollow_sphere

# n_particles = round(Int, 0.12 / fluid_particle_spacing)
# cylinder = RectangularShape(fluid_particle_spacing, (n_particles, n_particles), (0.2 - 1.0, 0.24 - 1.0), density=fluid_density)

fluid = setdiff(tank.fluid, filled_sphere)

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

viscosity_fluid = ViscosityAdami(; nu)
# viscosity_fluid = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
viscosity_wall = ViscosityAdami(; nu)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
fluid_density_calculator = ContinuityDensity()
fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           density_diffusion=density_diffusion,
                                           pressure_acceleration=TrixiParticles.tensile_instability_control,
                                           particle_shifting=TrixiParticles.ParticleShiftingSun2019(5.0),
                                           smoothing_length, viscosity=viscosity_fluid)

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation(pressure_offset=10000.0)
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

# Movement function
# https://en.wikipedia.org/wiki/Triangle_wave#Harmonics
# triangle_series(t, N) = 8 / pi^2 * sum(i -> (-1)^i / (2i + 1)^2 * sin(2pi * (2i + 1) * t), 0:(N-1))
# movement_function(x, t) = x + SVector(0.4 * triangle_series(2 * t, 5), 0.0)
# is_moving(t) = true
# boundary_movement = BoundaryMovement(movement_function, is_moving)

boundary_movement = TrixiParticles.oscillating_movement(1.0,
                                                        SVector(0.4, 0.0),
                                                        0.0, center;
                                                        ramp_up=0.5)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

boundary_model_cylinder = BoundaryModelDummyParticles(hollow_sphere.density,
                                                      hollow_sphere.mass,
                                                      state_equation=state_equation,
                                                      boundary_density_calculator,
                                                      smoothing_kernel, smoothing_length,
                                                      viscosity=viscosity_wall)

boundary_system_cylinder = BoundarySPHSystem(hollow_sphere, boundary_model_cylinder,
                                             movement=boundary_movement)

# boundary_system_cylinder = TotalLagrangianSPHSystem(hollow_sphere, smoothing_kernel, smoothing_length,
#                                         1e5, 0.0;
#                                         n_fixed_particles=nparticles(hollow_sphere), movement=boundary_movement,
#                                         boundary_model=boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(fluid.coordinates, dims=2) .- fluid_particle_spacing / 2
max_corner = maximum(fluid.coordinates, dims=2) .+ fluid_particle_spacing / 2
periodic_box = PeriodicBox(; min_corner, max_corner)
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box, cell_list)

semi = Semidiscretization(fluid_system, boundary_system_cylinder;
                          neighborhood_search)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=0.01, prefix="")
shifting_callback = ParticleShiftingCallback()

stepsize_callback = StepsizeCallback(cfl=1.5)

callbacks = CallbackSet(info_callback, saving_callback, shifting_callback,
                        stepsize_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
# fluid_dt = 1e-3
# sol = solve(ode, RDPK3SpFSAL49(),
#             # abstol=1.0e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
#             # reltol=1.0e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
#             adaptive=false, dt=fluid_dt,
#             save_everystep=false, callback=callbacks);

time_step = 1.0
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=time_step, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);

# plane = interpolate_plane([0.0, -0.25], [1.0, 0.75], 0.0025, semi, fluid_system, sol)
