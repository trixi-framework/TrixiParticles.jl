using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# 2D Periodic Poiseuille Flow with Carreau-Yasuda Viscosity
#
# This example simulates pressure-driven channel flow with the
# `ViscosityCarreauYasuda` non-Newtonian viscosity model. The driving pressure
# gradient is represented by an equivalent body acceleration.
# ==========================================================================================

# ==========================================================================================
# ==== Resolution
ny = 50

# ==========================================================================================
# ==== Experiment Setup
t_end_factor = 0.1
eps_factor = 1.0
sound_speed_factor = 60.0
initial_condition_mode = "analytical"
power_law_index = 1.0

channel_height = 1.0
channel_length = 6.0 * channel_height
particle_spacing = channel_height / ny
boundary_layers = 5

fluid_density = 1000.0
nu0 = 1.0e-3
nu_inf = 0.0
lambda_exponent = 2.0
reynolds_number = 200.0

reference_velocity = reynolds_number * nu0 / channel_height
pressure_gradient = 8.0 * fluid_density * reference_velocity^2 /
                    (reynolds_number * channel_height)
acceleration_x = pressure_gradient / fluid_density
carreau_time_constant = channel_height / max(reference_velocity, eps())

t_end = t_end_factor * channel_height / max(reference_velocity, eps())
tspan = (0.0, t_end)

if !(initial_condition_mode in ("newtonian", "analytical", "zero"))
    throw(ArgumentError("initial condition mode must be \"newtonian\", " *
                        "\"analytical\", or \"zero\""))
end

# ==========================================================================================
# ==== Analytical Solution
function linear_interpolation_clamped(x, y, interpolation_point)
    interpolation_point <= first(x) && return first(y)
    interpolation_point >= last(x) && return last(y)

    i = searchsortedlast(x, interpolation_point)
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    return y0 + (y1 - y0) * (interpolation_point - x0) / (x1 - x0)
end

function carreau_yasuda_kinematic_viscosity(shear_rate, nu0, nu_inf,
                                            time_constant, lambda_exponent,
                                            power_law_index)
    return nu_inf + (nu0 - nu_inf) *
           (1.0 + (time_constant * shear_rate)^lambda_exponent)^((power_law_index -
                                                                    1.0) /
                                                                   lambda_exponent)
end

function solve_shear_rate_from_stress(shear_stress, density, nu0, nu_inf,
                                      time_constant, lambda_exponent,
                                      power_law_index)
    shear_stress <= 0 && return 0.0

    residual(shear_rate) = density *
                           carreau_yasuda_kinematic_viscosity(shear_rate, nu0,
                                                              nu_inf,
                                                              time_constant,
                                                              lambda_exponent,
                                                              power_law_index) *
                           shear_rate - shear_stress

    lower = 0.0
    upper = 1.0
    while residual(upper) < 0.0
        upper *= 2.0
        upper > 1.0e12 &&
            error("failed to bracket shear-rate root for shear stress $shear_stress")
    end

    for _ in 1:120
        middle = 0.5 * (lower + upper)
        residual_middle = residual(middle)

        if abs(residual_middle) <= 1.0e-12 * max(shear_stress, 1.0)
            return middle
        elseif residual_middle > 0
            upper = middle
        else
            lower = middle
        end
    end

    return 0.5 * (lower + upper)
end

function analytical_ux_profile(y_positions, power_law_index, channel_height,
                               density, nu0, nu_inf, time_constant,
                               lambda_exponent, pressure_gradient)
    distances_to_centerline = sort(unique(abs.(y_positions .- 0.5 * channel_height)))
    shear_rates = similar(distances_to_centerline)

    for i in eachindex(distances_to_centerline)
        shear_stress = pressure_gradient * distances_to_centerline[i]
        shear_rates[i] = solve_shear_rate_from_stress(shear_stress, density, nu0,
                                                      nu_inf, time_constant,
                                                      lambda_exponent,
                                                      power_law_index)
    end

    velocity_at_distance = zeros(length(distances_to_centerline))
    for i in (lastindex(distances_to_centerline) - 1):-1:firstindex(distances_to_centerline)
        ds = distances_to_centerline[i + 1] - distances_to_centerline[i]
        velocity_at_distance[i] = velocity_at_distance[i + 1] +
                                  0.5 * (shear_rates[i + 1] + shear_rates[i]) * ds
    end

    velocity = Vector{Float64}(undef, length(y_positions))
    for (i, y) in pairs(y_positions)
        distance_to_centerline = abs(y - 0.5 * channel_height)
        velocity[i] = linear_interpolation_clamped(distances_to_centerline,
                                                   velocity_at_distance,
                                                   distance_to_centerline)
    end

    return velocity
end

function newtonian_ux(y, channel_height, density, nu0, pressure_gradient)
    return pressure_gradient / (2.0 * density * nu0) * y * (channel_height - y)
end

function l2_velocity_error(system::TrixiParticles.AbstractFluidSystem,
                           dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    y_positions = [TrixiParticles.current_coords(u, system, particle)[2]
                   for particle in TrixiParticles.eachparticle(system)]
    analytical_velocity = analytical_ux_profile(y_positions, power_law_index,
                                                channel_height, fluid_density, nu0,
                                                nu_inf, carreau_time_constant,
                                                lambda_exponent, pressure_gradient)

    squared_error = 0.0
    squared_reference = 0.0
    for (i, particle) in enumerate(TrixiParticles.eachparticle(system))
        ux = TrixiParticles.current_velocity(v, system, particle)[1]
        squared_error += (ux - analytical_velocity[i])^2
        squared_reference += analytical_velocity[i]^2
    end

    return sqrt(squared_error / nparticles(system)) /
           (sqrt(squared_reference / nparticles(system)) + eps())
end
l2_velocity_error(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

function max_velocity_error(system::TrixiParticles.AbstractFluidSystem,
                            dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    y_positions = [TrixiParticles.current_coords(u, system, particle)[2]
                   for particle in TrixiParticles.eachparticle(system)]
    analytical_velocity = analytical_ux_profile(y_positions, power_law_index,
                                                channel_height, fluid_density, nu0,
                                                nu_inf, carreau_time_constant,
                                                lambda_exponent, pressure_gradient)

    error = 0.0
    for (i, particle) in enumerate(TrixiParticles.eachparticle(system))
        ux = TrixiParticles.current_velocity(v, system, particle)[1]
        error = max(error, abs(ux - analytical_velocity[i]))
    end

    return error
end
max_velocity_error(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

# ==========================================================================================
# ==== Initial Condition
initial_velocity = if initial_condition_mode == "zero"
    (0.0, 0.0)
elseif initial_condition_mode == "analytical"
    y_reference = collect(range(0.0, channel_height; length=4 * ny + 1))
    ux_reference = analytical_ux_profile(y_reference, power_law_index,
                                         channel_height, fluid_density, nu0,
                                         nu_inf, carreau_time_constant,
                                         lambda_exponent, pressure_gradient)
    x -> (linear_interpolation_clamped(y_reference, ux_reference, x[2]), 0.0)
else
    x -> (newtonian_ux(x[2], channel_height, fluid_density, nu0,
                       pressure_gradient), 0.0)
end

# ==========================================================================================
# ==== Fluid
tank = RectangularTank(particle_spacing, (channel_length, channel_height),
                       (channel_length, channel_height), fluid_density;
                       n_layers=boundary_layers,
                       faces=(false, false, true, true),
                       velocity=initial_velocity,
                       coordinates_eltype=Float64)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

sound_speed = sound_speed_factor * reference_velocity
state_equation = StateEquationCole(; sound_speed,
                                    reference_density=fluid_density,
                                    exponent=7)

viscosity = ViscosityCarreauYasuda(; nu0, nu_inf,
                                   lambda=carreau_time_constant,
                                   a=lambda_exponent,
                                   n=power_law_index,
                                   epsilon=max(0.5, eps_factor) * particle_spacing)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid;
                                           density_calculator=ContinuityDensity(),
                                           state_equation,
                                           smoothing_kernel,
                                           smoothing_length,
                                           acceleration=(acceleration_x, 0.0),
                                           viscosity,
                                           shifting_technique=nothing)

# ==========================================================================================
# ==== Boundary
boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                             tank.boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel,
                                             smoothing_length;
                                             state_equation,
                                             viscosity)

boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[0.0, -10.0 * channel_height],
                           max_corner=[channel_length, 10.0 * channel_height])
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box)

semi = Semidiscretization(fluid_system, boundary_system;
                          neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

n_label = replace(string(power_law_index), "." => "p")
output_directory = joinpath("out_poiseuille_carreau", "n_$power_law_index")
result_filename = "validation_run_poiseuille_carreau_2d_n_$(n_label)_ny_$ny"

info_callback = InfoCallback(interval=200)
saving_callback = SolutionSavingCallback(; dt=t_end / 20,
                                         prefix="",
                                         output_directory)
pp_callback = PostprocessCallback(; dt=t_end / 20,
                                  output_directory,
                                  filename=result_filename,
                                  l2_velocity_error,
                                  max_velocity_error,
                                  write_csv=true)
cfl_callback = StepsizeCallback(cfl=0.2)
callbacks = CallbackSet(info_callback, saving_callback, pp_callback,
                        cfl_callback, UpdateCallback())

sol = solve(ode, RDPK3SpFSAL35();
            abstol=1.0e-7,
            reltol=1.0e-4,
            save_everystep=false,
            callback=callbacks,
            maxiters=2_000_000);
