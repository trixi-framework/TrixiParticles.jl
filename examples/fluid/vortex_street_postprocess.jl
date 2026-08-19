using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
const fluid_particle_spacing = 0.2

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
# Boundary geometry and initial fluid particle positions
tank_size = (65.0, 20.0)
initial_fluid_size = tank_size

Re = 200
initial_velocity = (1.0, 0.0)
nu = 1 * 1 / Re

strouhal_number = 0.198 * (1 - 19.7 / Re)
frequency = strouhal_number * initial_velocity[1] / 1

tspan = (0.0, 50.0)

fluid_density = 1000.0
sound_speed = 10initial_velocity[1]
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(false, false, true, true), velocity=initial_velocity)

const radius = 2.0
const center = SVector(5.0, 10.0)
hollow_sphere = SphereShape(fluid_particle_spacing, radius, Tuple(center), fluid_density,
                            n_layers=4, sphere_type=RoundSphere())

filled_sphere = SphereShape(fluid_particle_spacing, radius, Tuple(center), fluid_density,
                            sphere_type=RoundSphere())

# n_particles = round(Int, 0.12 / fluid_particle_spacing)
# cylinder = RectangularShape(fluid_particle_spacing, (n_particles, n_particles), (0.2 - 1.0, 0.24 - 1.0), density=fluid_density)

fluid = setdiff(tank.fluid, filled_sphere)

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

viscosity = ViscosityAdami(; nu)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
fluid_density_calculator = ContinuityDensity()
fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           density_diffusion=density_diffusion,
                                           pressure_acceleration=TrixiParticles.tensile_instability_control,
                                        #    transport_velocity=TransportVelocityAdami(50_000.0),
                                           smoothing_length, viscosity=viscosity)

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

boundary_model_cylinder = BoundaryModelDummyParticles(hollow_sphere.density,
                                                      hollow_sphere.mass,
                                                      state_equation=state_equation,
                                                      boundary_density_calculator,
                                                      smoothing_kernel, smoothing_length,
                                                      viscosity=viscosity)

boundary_system_cylinder = BoundarySPHSystem(hollow_sphere, boundary_model_cylinder)

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[0.0, -1.0], max_corner=[65.0, 21.0])
cell_list = FullGridCellList(min_corner=[0.0, -1.0], max_corner=[65.0, 21.0])
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box, cell_list)

semi = Semidiscretization(fluid_system, boundary_system, boundary_system_cylinder;
                          neighborhood_search)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=1.0, prefix="tvf_old")
shifting_callback = ParticleShiftingCallback()

# ==========================================================================================
# ==== Postprocessing
circle = SphereShape(fluid_particle_spacing, (2 * radius + fluid_particle_spacing) / 2,
                     Tuple(center), fluid_density, n_layers=1,
                     sphere_type=RoundSphere())

# Points for pressure interpolation, located at the wall interface
const data_points = copy(circle.coordinates)

calculate_lift_force(system, v_ode, u_ode, semi, t) = nothing
function calculate_lift_force(system::TrixiParticles.FluidSystem, v_ode, u_ode, semi, t)
    force = zero(SVector{ndims(system), eltype(system)})

    values = interpolate_points(data_points, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                clip_negative_pressure=false)
    pressure = Array(values.pressure)

    for i in axes(data_points, 2)
        point = TrixiParticles.current_coords(data_points, system, i)

        # F = ∑ -p_i * A_i * n_i
        force -= pressure[i] * fluid_particle_spacing .*
                 TrixiParticles.normalize(point - center)
    end

    return 2 * force[2] / (fluid_density * 1^2 * 2 * radius)
end

calculate_drag_force(system, v_ode, u_ode, semi, t) = nothing
function calculate_drag_force(system::TrixiParticles.FluidSystem, v_ode, u_ode, semi, t)
    force = zero(SVector{ndims(system), eltype(system)})

    values = interpolate_points(data_points, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                clip_negative_pressure=false)
    pressure = Array(values.pressure)

    for i in axes(data_points, 2)
        point = TrixiParticles.current_coords(data_points, system, i)

        # F = ∑ -p_i * A_i * n_i
        force -= pressure[i] * fluid_particle_spacing .*
                 TrixiParticles.normalize(point - center)
    end

    return 2 * force[1] / (fluid_density * 1^2 * 2 * radius)
end

pp_callback = PostprocessCallback(; dt=0.5,
                                  f_l=calculate_lift_force, f_d=calculate_drag_force,
                                  filename="resulting_force_pst",
                                  write_csv=true, write_file_interval=10)

callbacks = CallbackSet(info_callback, saving_callback, shifting_callback,
                        UpdateCallback(), pp_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1.0e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1.0e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            save_everystep=false, callback=callbacks);

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt=1.0, # This is overwritten by the stepsize callback
#             save_everystep=false, callback=callbacks);

# plane = interpolate_plane([0.0, -0.25], [1.0, 0.75], 0.0025, semi, fluid_system, sol)
