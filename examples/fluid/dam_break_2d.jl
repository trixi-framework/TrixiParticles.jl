# 2D dam break simulation based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

using TrixiParticles
using OrdinaryDiffEq
using JSON
using PyCall
pygui(:qt5)
using PyPlot

# Constants
gravity = 9.81
athmospheric_pressure = 100000.0
fluid_density = 1000.0

# Simulation settings
particle_spacing = 0.02
smoothing_length = 1.2 * particle_spacing
boundary_layers = 3
output_dt = 0.02
relaxation_step_file_prefix = "relaxation"
simulation_step_file_prefix = ""
relaxation_tspan = (0.0, 5.0)
simulation_tspan = (0.0, 1.0)

# Model settings
fluid_density_calculator = ContinuityDensity()
boundary_density_calculator = AdamiPressureExtrapolation()

# Boundary geometry and initial fluid particle positions
initial_fluid_height = 2.0
initial_fluid_size = (initial_fluid_height, 1.0)
tank_size = (floor(5.366 / particle_spacing) * particle_spacing, 4.0)
tank = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers)

# move the right wall of the tank to a new position
function move_wall(tank, new_wall_position)
    reset_faces = (false, true, false, false)
    positions = (0, new_wall_position, 0, 0)
    reset_wall!(tank, reset_faces, positions)
end

move_wall(tank, tank.fluid_size[1])

# ==========================================================================================
# ==== Fluid
sound_speed = 20 * sqrt(gravity * initial_fluid_height)

state_equation = StateEquationCole(sound_speed, 7, fluid_density, athmospheric_pressure,
                                   background_pressure=athmospheric_pressure)

smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity), correction=nothing, pp_values=Dict("dp"=>0.0, "ekin"=>0.0))

# ==========================================================================================
# ==== Boundary models
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation, boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

# K = 9.81 * initial_fluid_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              tank.boundary.mass)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-4)

ode = semidiscretize(semi, relaxation_tspan)

info_callback = InfoCallback(interval=100)
saving_callback_relaxation = SolutionSavingCallback(dt=output_dt,
                                                    prefix=relaxation_step_file_prefix)
density_reinit_cb = nothing #DensityReinitializationCallback(semi.systems[1], dt=0.01)
pp_cb = PostprocessCallback()

callbacks_relaxation = CallbackSet(info_callback, saving_callback_relaxation, pp_cb, density_reinit_cb)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks_relaxation);

# move_wall(tank, tank.tank_size[1])

# # Use solution of the relaxing step as initial coordinates
# restart_with!(semi, sol)

# semi = Semidiscretization(fluid_system, boundary_system,
#                           neighborhood_search=SpatialHashingSearch)
# ode = semidiscretize(semi, simulation_tspan)

# saving_callback = SolutionSavingCallback(dt=output_dt, prefix=simulation_step_file_prefix)
# density_reinit_cb = nothing#DensityReinitializationCallback(semi.systems[1], dt=0.01)
# callbacks = CallbackSet(info_callback, saving_callback, density_reinit_cb, PostprocessCallback())

# # See above for an explanation of the parameter choice
# sol = solve(ode, RDPK3SpFSAL49(),
#             abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
#             reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
#             dtmax=1e-2, # Limit stepsize to prevent crashing
#             save_everystep=false, callback=callbacks);

function running_average(data::Vector{Float64}, window_size::Int)
    @assert window_size >= 1 "Window size for running average must be >= 1"

    cum_sum = cumsum(data)
    cum_sum = vcat(zeros(window_size - 1), cum_sum)  # prepend zeros

    averaged_data = (cum_sum[window_size:end] - cum_sum[1:end-window_size + 1]) / window_size
    return averaged_data
end

using GLM
using DataFrames

function calculate_regression(data::Vector{Float64}, t::Vector{Float64})
    @assert length(data) == length(t) "Data and time vectors must have the same length"

    df = DataFrame(Y=data, T=t)
    model = lm(@formula(Y ~ T), df)  # Perform linear regression

    # Get the regression line values
    trend = predict(model, df)

    # Extract the gradient of the trend line
    gradient = coef(model)[2]

    return trend, gradient
end

function plot_json_data(dir_path::AbstractString="")
    if isempty(dir_path)
        dir_path = pwd()
    end

    files = readdir(dir_path)
    json_files = filter(x -> occursin(r"^values(\d*).json$", x), files)

    if isempty(json_files)
        println("No matching JSON files found.")
        return
    end

    recent_file = sort(json_files, by = x -> stat(joinpath(dir_path, x)).ctime, rev=true)[1]
    file_path = joinpath(dir_path, recent_file)

    json_string = read(file_path, String)
    json_data = JSON.parse(json_string)

    dt_series = json_data["dt"]
    dt_values = Vector{Float64}(dt_series["values"])
    accumulated_dt = cumsum(dt_values)

    function plot_dt()
        xlabel("T")
        ylabel("dt")
        title("dt")
        yscale("log")
        scatter(accumulated_dt, dt_values, s=4, color="blue", label="dt")
        legend()
    end

    subplot_number = 1
    for key in ["dp", "ekin"]
        if haskey(json_data, key)
            subplot_number += 1
        end
    end

    figure(figsize=(10, 5 * subplot_number))

    subplot(subplot_number, 1, 1)
    plot_dt()

    subplot_index = 2
    for key in ["dp", "ekin"]
        if haskey(json_data, key)
            series = json_data[key]
            values = Vector{Float64}(series["values"])

            subplot(subplot_number, 1, subplot_index)
            xlabel("T")
            ylabel(key)
            title(key)
            scatter(accumulated_dt, values, s=4, label=key)

            if key == "ekin"  # calculate and plot running average for ekin
                window_size = 50
                averaged_ekin = running_average(values, window_size)
                # println("Length of accumulated_dt: ", length(accumulated_dt))
                # println("Length of averaged_ekin: ", length(averaged_ekin))
                # println("Window size: ", window_size)
                scatter(accumulated_dt, averaged_ekin, s=4, color="orange", label="ekin (run. avg. windowsize = $window_size)")

                trend_line, gradient = calculate_regression(values, accumulated_dt)
                plot(accumulated_dt, trend_line, label="trend (gradient=$gradient)", color="red", linewidth=2)
            end

            legend()
            subplot_index += 1
        end
    end

    show()
end

plot_json_data()
