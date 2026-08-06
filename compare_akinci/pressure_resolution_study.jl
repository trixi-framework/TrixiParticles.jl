using LinearAlgebra
using Printf
using Serialization
using SparseArrays
using Statistics
using TrixiParticles

const TP = TrixiParticles

include(joinpath(@__DIR__, "pressure_equilibrium.jl"))

function corrected_boundary_mass(particle_spacing)
    plate_size = (0.06, 0.06)
    n_plate = round.(Int, plate_size ./ particle_spacing)
    plate = RectangularShape(particle_spacing, (n_plate..., 3),
                             (-plate_size[1] / 2, -plate_size[2] / 2,
                              -3particle_spacing);
                             density=1000.0)
    return akinci_boundary_hydrodynamic_mass(plate,
                                             SchoenbergCubicSplineKernel{3}(),
                                             particle_spacing - eps(), 1000.0)
end

function assemble_sparse_operator(target_particle_count)
    particle_spacing = cbrt(1.0e-6 / target_particle_count)
    boundary_mass = corrected_boundary_mass(particle_spacing)
    config = case_config("wetting_no")
    semi,
    ode_state = diagnostic_setup(config;
                                 setup_overrides=(;
                                                  particle_spacing,
                                                  fluid_density_calculator=ContinuityDensity(),
                                                  boundary_hydrodynamic_mass=boundary_mass,
                                                  parallelization_backend=SerialBackend()))
    v_ode, u_ode = ode_state
    TP.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    fluid_system = semi.systems[1]
    dimensions = TP.ndims(fluid_system)
    n_fluid = TP.nparticles(fluid_system)
    v_fluid = TP.wrap_v(v_ode, fluid_system, semi)
    u_fluid = TP.wrap_u(u_ode, fluid_system, semi)
    fluid_coordinates = TP.current_coordinates(u_fluid, fluid_system)
    fluid_density = [TP.current_density(v_fluid, fluid_system, particle)
                     for particle in TP.eachparticle(fluid_system)]
    quantities = force_components(v_ode, u_ode, semi, 0.0)

    estimated_entries = 180n_fluid
    rows = sizehint!(Int[], estimated_entries)
    columns = sizehint!(Int[], estimated_entries)
    values = sizehint!(Float64[], estimated_entries)

    TP.foreach_point_neighbor(fluid_system, fluid_system, fluid_coordinates,
                              fluid_coordinates, semi;
                              points=TP.each_integrated_particle(fluid_system)) do particle,
                                                                                   neighbor,
                                                                                   pos_diff,
                                                                                   distance
        distance < sqrt(eps(TP.compact_support(fluid_system, fluid_system)^2)) && return
        gradient = TP.smoothing_kernel_grad_unsafe(fluid_system, pos_diff, distance,
                                                   particle)
        neighbor_mass = TP.hydrodynamic_mass(fluid_system, neighbor)
        coefficient = -neighbor_mass /
                      (fluid_density[particle] * fluid_density[neighbor])
        for dimension in 1:dimensions
            row = dimension + dimensions * (particle - 1)
            push!(rows, row)
            push!(columns, particle)
            push!(values, coefficient * gradient[dimension])
            push!(rows, row)
            push!(columns, neighbor)
            push!(values, coefficient * gradient[dimension])
        end
    end

    baseline_wall_acceleration = zeros(Float64, dimensions, n_fluid)
    wall_coefficient = zeros(Float64, n_fluid)
    smoothing_length = TP.initial_smoothing_length(fluid_system)
    smoothing_kernel = fluid_system.smoothing_kernel
    gravity = -TP.acceleration_source(fluid_system)[end]
    support_count = 0
    integral_cache = Dict{Float64, Float64}()
    for particle in TP.eachparticle(fluid_system)
        wall_distance = fluid_coordinates[end, particle]
        integral = get!(integral_cache, wall_distance) do
            planar_kernel_integral(smoothing_kernel, smoothing_length, wall_distance)
        end
        iszero(integral) && continue
        support_count += 1
        density = fluid_density[particle]
        wall_pressure = max(density * gravity * wall_distance, 0)
        baseline_wall_acceleration[end, particle] = integral * wall_pressure / density
        wall_coefficient[particle] = 2integral / density
        push!(rows, dimensions * particle)
        push!(columns, particle)
        push!(values, wall_coefficient[particle])
    end

    operator = sparse(rows, columns, values, dimensions * n_fluid, n_fluid)
    nonpressure_acceleration = quantities.total - quantities.pressure_boundary
    baseline_acceleration = vec(nonpressure_acceleration + baseline_wall_acceleration)
    return (; operator, baseline_acceleration, wall_coefficient,
            baseline_wall_acceleration=vec(baseline_wall_acceleration[end, :]),
            particle_spacing, n_fluid, support_count,
            boundary_particle_count=TP.nparticles(semi.systems[2]))
end

function cgls(operator, target; initial=nothing, max_iterations=1500,
              tolerance=1.0e-9)
    solution = isnothing(initial) ? zeros(size(operator, 2)) : copy(initial)
    residual = target - operator * solution
    gradient = transpose(operator) * residual
    direction = copy(gradient)
    gradient_norm_squared = dot(gradient, gradient)
    initial_gradient_norm = sqrt(gradient_norm_squared)

    for iteration in 1:max_iterations
        projected_direction = operator * direction
        denominator = dot(projected_direction, projected_direction)
        denominator > eps() || return solution, iteration
        step = gradient_norm_squared / denominator
        solution .+= step .* direction
        residual .-= step .* projected_direction
        gradient_new = transpose(operator) * residual
        gradient_norm_squared_new = dot(gradient_new, gradient_new)
        if sqrt(gradient_norm_squared_new) < tolerance * max(initial_gradient_norm, 1)
            return solution, iteration
        end
        direction .= gradient_new .+
                     (gradient_norm_squared_new / gradient_norm_squared) .* direction
        gradient .= gradient_new
        gradient_norm_squared = gradient_norm_squared_new
    end

    return solution, max_iterations
end

function solve_resolution(target_particle_count)
    assembled = assemble_sparse_operator(target_particle_count)
    operator_kpa = 1000assembled.operator
    target = -assembled.baseline_acceleration
    radius = cbrt(3.0e-6 / (4pi))
    initial_pressure = fill(2 / radius / 1000, assembled.n_fluid)
    pressure_kpa, iterations = cgls(operator_kpa, target; initial=initial_pressure)

    if any(<(0), pressure_kpa)
        pressure_kpa, iterations,
        _ = nonnegative_least_squares(operator_kpa, target;
                                      initial_pressure=max.(pressure_kpa,
                                                            0),
                                      max_iterations=2000)
    end

    residual = assembled.baseline_acceleration + operator_kpa * pressure_kpa
    residual_vectors = reshape(residual, 3, :)
    residual_magnitude = vec(sqrt.(sum(abs2, residual_vectors; dims=1)))
    wall_acceleration = assembled.baseline_wall_acceleration +
                        assembled.wall_coefficient .* (1000pressure_kpa)
    active_wall_acceleration = wall_acceleration[assembled.wall_coefficient .> 0]

    result = (; target_particle_count, particle_count=assembled.n_fluid,
              boundary_particle_count=assembled.boundary_particle_count,
              particle_spacing=assembled.particle_spacing,
              support_count=assembled.support_count, pressure=1000pressure_kpa,
              iterations, residual_rms=sqrt(mean(abs2, residual_magnitude)),
              residual_median=median(residual_magnitude),
              residual_p90=quantile(residual_magnitude, 0.9),
              residual_maximum=maximum(residual_magnitude),
              mean_vertical_residual=mean(residual_vectors[end, :]),
              wall_acceleration_sum=sum(wall_acceleration),
              wall_acceleration_median=median(active_wall_acceleration),
              wall_acceleration_maximum=maximum(active_wall_acceleration))

    @printf("target=%d actual=%d dx=%.6g boundary=%d support=%d\n",
            target_particle_count, result.particle_count, result.particle_spacing,
            result.boundary_particle_count, result.support_count)
    @printf("  residual RMS=%.6f median=%.6f p90=%.6f max=%.6f mean_z=%.6f\n",
            result.residual_rms, result.residual_median, result.residual_p90,
            result.residual_maximum, result.mean_vertical_residual)
    @printf("  pressure range=[%.3f, %.3f] Pa, wall sum=%.3f median=%.3f max=%.3f\n",
            minimum(result.pressure), maximum(result.pressure),
            result.wall_acceleration_sum, result.wall_acceleration_median,
            result.wall_acceleration_maximum)
    return result
end

function run_resolution_study(output_path, targets)
    results = if isfile(output_path)
        open(deserialize, output_path).results
    else
        NamedTuple[]
    end
    for target in targets
        any(result -> result.target_particle_count == target, results) && continue
        push!(results, solve_resolution(target))
        open(output_path, "w") do io
            serialize(io, (; results))
        end
    end
    println("Wrote pressure resolution study to ", output_path)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 ||
        error("usage: pressure_resolution_study.jl OUTPUT.jls TARGET_COUNT [TARGET_COUNT ...]")
    run_resolution_study(ARGS[1], parse.(Int, ARGS[2:end]))
end
