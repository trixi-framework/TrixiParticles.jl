using LinearAlgebra
using Printf
using Serialization
using Statistics
using TrixiParticles

const TP = TrixiParticles

include(joinpath(@__DIR__, "boundary_volume.jl"))
include(joinpath(@__DIR__, "force_analysis.jl"))

function corrected_boundary_mass()
    particle_spacing = cbrt(1.0e-6 / 750)
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

function adami_pressure_weights(fluid_system, boundary_system, v_ode, u_ode, semi)
    n_fluid = TP.nparticles(fluid_system)
    n_boundary = TP.nparticles(boundary_system)
    weights = zeros(eltype(fluid_system), n_boundary, n_fluid)
    normalization = zeros(eltype(fluid_system), n_boundary)

    u_fluid = TP.wrap_u(u_ode, fluid_system, semi)
    u_boundary = TP.wrap_u(u_ode, boundary_system, semi)
    fluid_coordinates = TP.current_coordinates(u_fluid, fluid_system)
    boundary_coordinates = TP.current_coordinates(u_boundary, boundary_system)
    boundary_model = boundary_system.boundary_model

    TP.foreach_point_neighbor(boundary_system, fluid_system, boundary_coordinates,
                              fluid_coordinates, semi;
                              points=TP.eachparticle(boundary_system)) do boundary_particle,
                                                                          fluid_particle,
                                                                          pos_diff,
                                                                          distance
        weight = TP.smoothing_kernel(boundary_model, distance, boundary_particle)
        weights[boundary_particle, fluid_particle] += weight
        normalization[boundary_particle] += weight
    end

    for boundary_particle in axes(weights, 1)
        normalization[boundary_particle] > eps() || continue
        weights[boundary_particle, :] ./= normalization[boundary_particle]
    end

    return weights, copy(boundary_model.pressure)
end

function reflected_adami_pressure_weights(fluid_system, boundary_system, v_ode, u_ode,
                                          semi, ghost_shift)
    n_fluid = TP.nparticles(fluid_system)
    n_boundary = TP.nparticles(boundary_system)
    weights = zeros(eltype(fluid_system), n_boundary, n_fluid)
    pressure_numerator = zeros(eltype(fluid_system), n_boundary)
    normalization = zeros(eltype(fluid_system), n_boundary)

    v_fluid = TP.wrap_v(v_ode, fluid_system, semi)
    u_fluid = TP.wrap_u(u_ode, fluid_system, semi)
    u_boundary = TP.wrap_u(u_ode, boundary_system, semi)
    fluid_coordinates = TP.current_coordinates(u_fluid, fluid_system)
    boundary_coordinates = TP.current_coordinates(u_boundary, boundary_system)
    boundary_model = boundary_system.boundary_model
    support = TP.compact_support(boundary_model.smoothing_kernel,
                                 boundary_model.smoothing_length)
    dimensions = TP.ndims(fluid_system)
    top = maximum(boundary_coordinates[end, :])
    fluid_acceleration = TP.acceleration_source(fluid_system)

    for boundary_particle in TP.eachparticle(boundary_system)
        isapprox(boundary_coordinates[end, boundary_particle], top) || continue
        for fluid_particle in TP.eachparticle(fluid_system)
            distance_squared = zero(eltype(fluid_system))
            for dimension in 1:dimensions
                evaluation_coordinate = boundary_coordinates[dimension, boundary_particle] +
                                        (dimension == dimensions ? ghost_shift : 0)
                difference = evaluation_coordinate -
                             fluid_coordinates[dimension, fluid_particle]
                distance_squared += difference^2
            end
            distance_squared < support^2 || continue
            distance = sqrt(distance_squared)
            weight = TP.smoothing_kernel(boundary_model, distance, boundary_particle)
            density = TP.current_density(v_fluid, fluid_system, fluid_particle)
            hydrostatic_pressure = zero(eltype(fluid_system))
            for dimension in 1:dimensions
                original_difference = boundary_coordinates[dimension, boundary_particle] -
                                      fluid_coordinates[dimension, fluid_particle]
                hydrostatic_pressure += fluid_acceleration[dimension] * density *
                                        original_difference
            end
            weights[boundary_particle, fluid_particle] += weight
            pressure_numerator[boundary_particle] += weight * hydrostatic_pressure
            normalization[boundary_particle] += weight
        end
    end

    boundary_pressure = zeros(eltype(fluid_system), n_boundary)
    for boundary_particle in TP.eachparticle(boundary_system)
        normalization[boundary_particle] > eps() || continue
        weights[boundary_particle, :] ./= normalization[boundary_particle]
        boundary_pressure[boundary_particle] = max(pressure_numerator[boundary_particle] /
                                                   normalization[boundary_particle], 0)
    end

    return weights, boundary_pressure
end

function boundary_pressure_derivative(state_equation, pressure)
    pressure_step = max(1.0e-3, 1.0e-6 * max(abs(pressure), 1.0))
    density = TP.inverse_state_equation(state_equation, pressure)
    density_plus = TP.inverse_state_equation(state_equation, pressure + pressure_step)
    return ((pressure + pressure_step) / density_plus^2 - pressure / density^2) /
           pressure_step
end

function boundary_pressure_derivative_continuity(state_equation, pressure, fluid_density)
    pressure_step = max(1.0e-3, 1.0e-6 * max(abs(pressure), 1.0))
    density = TP.inverse_state_equation(state_equation, pressure)
    density_plus = TP.inverse_state_equation(state_equation, pressure + pressure_step)
    return ((pressure + pressure_step) / (fluid_density * density_plus) -
            pressure / (fluid_density * density)) / pressure_step
end

function pressure_over_density_derivative(state_equation, pressure)
    pressure_step = max(1.0e-3, 1.0e-6 * max(abs(pressure), 1.0))
    density = TP.inverse_state_equation(state_equation, pressure)
    density_plus = TP.inverse_state_equation(state_equation, pressure + pressure_step)
    return ((pressure + pressure_step) / density_plus - pressure / density) /
           pressure_step
end

function planar_kernel_integral(smoothing_kernel, smoothing_length, wall_distance;
                                intervals=200)
    support = TP.compact_support(smoothing_kernel, smoothing_length)
    distance = max(wall_distance, zero(wall_distance))
    distance < support || return zero(wall_distance)
    iseven(intervals) || error("Simpson quadrature requires an even interval count")

    step = (support - distance) / intervals
    integral = zero(wall_distance)
    for index in 0:intervals
        radius = distance + index * step
        value = TP.kernel(smoothing_kernel, radius, smoothing_length) * radius
        coefficient = index == 0 || index == intervals ? 1 : (isodd(index) ? 4 : 2)
        integral += coefficient * value
    end
    return 2pi * step * integral / 3
end

function add_semi_analytical_wall!(operator, baseline_wall_acceleration, fluid_system,
                                   fluid_coordinates, fluid_density, density_calculator)
    dimensions = TP.ndims(fluid_system)
    dimensions == 3 || error("planar semi-analytical prototype currently requires 3D")
    state_equation = fluid_system.state_equation
    smoothing_kernel = fluid_system.smoothing_kernel
    smoothing_length = TP.initial_smoothing_length(fluid_system)
    gravity = -TP.acceleration_source(fluid_system)[end]
    integral_cache = Dict{eltype(fluid_system), eltype(fluid_system)}()

    for particle in TP.eachparticle(fluid_system)
        wall_distance = fluid_coordinates[end, particle]
        integral = get!(integral_cache, wall_distance) do
            planar_kernel_integral(smoothing_kernel, smoothing_length, wall_distance)
        end
        iszero(integral) && continue

        density = fluid_density[particle]
        wall_pressure = max(density * gravity * wall_distance, 0)
        wall_density = TP.inverse_state_equation(state_equation, wall_pressure)
        row = dimensions * particle

        if density_calculator isa SummationDensity
            baseline_wall_acceleration[end,
                                       particle] = integral * wall_pressure /
                                                   wall_density
            operator[row,
                     particle] += integral *
                                  (wall_density / density^2 +
                                   pressure_over_density_derivative(state_equation,
                                                                    wall_pressure))
        else
            baseline_wall_acceleration[end, particle] = integral * wall_pressure / density
            operator[row, particle] += 2integral / density
        end
    end

    return baseline_wall_acceleration
end

function add_mirrored_ghost_wall!(operator, baseline_wall_acceleration, fluid_system,
                                  fluid_coordinates, fluid_density, density_calculator)
    dimensions = TP.ndims(fluid_system)
    dimensions == 3 || error("mirrored ghost prototype currently requires 3D")
    support = TP.compact_support(fluid_system.smoothing_kernel,
                                 TP.initial_smoothing_length(fluid_system))
    gravity = -TP.acceleration_source(fluid_system)[end]

    for source_particle in TP.eachparticle(fluid_system)
        source_height = fluid_coordinates[end, source_particle]
        source_height < support || continue
        ghost_pressure_offset = 2fluid_density[source_particle] * gravity * source_height
        source_mass = TP.hydrodynamic_mass(fluid_system, source_particle)

        for particle in TP.eachparticle(fluid_system)
            pos_diff = SVector(ntuple(dimensions) do dimension
                                   ghost_coordinate = dimension == dimensions ?
                                                      -source_height :
                                                      fluid_coordinates[dimension,
                                                                        source_particle]
                                   fluid_coordinates[dimension, particle] -
                                   ghost_coordinate
                               end)
            distance = norm(pos_diff)
            distance < support || continue
            gradient = TP.smoothing_kernel_grad_unsafe(fluid_system, pos_diff, distance,
                                                       particle)

            if density_calculator isa SummationDensity
                coefficient_particle = -source_mass / fluid_density[particle]^2
                coefficient_source = -source_mass / fluid_density[source_particle]^2
                baseline_coefficient = coefficient_source * ghost_pressure_offset
            else
                coefficient_particle = -source_mass /
                                       (fluid_density[particle] *
                                        fluid_density[source_particle])
                coefficient_source = coefficient_particle
                baseline_coefficient = coefficient_source * ghost_pressure_offset
            end

            for dimension in 1:dimensions
                row = dimension + dimensions * (particle - 1)
                operator[row, particle] += coefficient_particle * gradient[dimension]
                operator[row, source_particle] += coefficient_source * gradient[dimension]
                baseline_wall_acceleration[dimension,
                                           particle] += baseline_coefficient *
                                                        gradient[dimension]
            end
        end
    end

    return baseline_wall_acceleration
end

function curvature_acceleration(assembled, scaled_normal)
    (; semi, fluid_system, v_ode, u_ode) = assembled.setup
    v_fluid = TP.wrap_v(v_ode, fluid_system, semi)
    u_fluid = TP.wrap_u(u_ode, fluid_system, semi)
    coordinates = TP.current_coordinates(u_fluid, fluid_system)
    acceleration = zeros(eltype(fluid_system), TP.ndims(fluid_system),
                         TP.nparticles(fluid_system))
    coefficient = TP.surface_tension_model(fluid_system).surface_tension_coefficient
    correction = fluid_system.correction

    TP.foreach_point_neighbor(fluid_system, fluid_system, coordinates, coordinates, semi;
                              points=TP.each_integrated_particle(fluid_system)) do particle,
                                                                                   neighbor,
                                                                                   pos_diff,
                                                                                   distance
        distance^2 < eps(TP.initial_smoothing_length(fluid_system)^2) && return
        density_particle = TP.current_density(v_fluid, fluid_system, particle)
        density_neighbor = TP.current_density(v_fluid, fluid_system, neighbor)
        correction_density_particle = TP.correction_density(correction, fluid_system,
                                                            particle, density_particle)
        correction_density_neighbor = TP.correction_density(correction, fluid_system,
                                                            neighbor, density_neighbor)
        _, _,
        surface_correction = TP.free_surface_correction(correction, fluid_system,
                                                        correction_density_particle,
                                                        correction_density_neighbor)
        acceleration[:,
                     particle] .-= surface_correction * coefficient .*
                                   (scaled_normal[:, particle] -
                                    scaled_normal[:, neighbor])
    end
    return acceleration
end

function impose_contact_angle(assembled, angle_degrees)
    normal = copy(assembled.quantities.scaled_normal)
    coordinates = assembled.quantities.coordinates
    center = vec(mean(coordinates; dims=2))
    wall_normal = SVector(0.0, 0.0, 1.0)
    angle = deg2rad(angle_degrees)
    contact = assembled.quantities.boundary_neighbor_count .> 0

    for particle in findall(contact)
        magnitude = norm(normal[:, particle])
        iszero(magnitude) && continue
        tangential = normal[:, particle] -
                     dot(normal[:, particle], wall_normal) * wall_normal
        if norm(tangential) <= eps()
            tangential = center - coordinates[:, particle]
            tangential[end] = 0
        end
        tangential_direction = norm(tangential) > eps() ? tangential / norm(tangential) :
                               zero(wall_normal)
        direction = -cos(angle) * wall_normal + sin(angle) * tangential_direction
        normal[:, particle] .= magnitude .* direction
    end

    original_curvature = curvature_acceleration(assembled,
                                                assembled.quantities.scaled_normal)
    reconstruction_error = maximum(abs, original_curvature - assembled.quantities.curvature)
    reconstruction_error < 1.0e-10 ||
        error("curvature reconstruction failed with error $reconstruction_error")
    modified_curvature = curvature_acceleration(assembled, normal)
    baseline_acceleration = assembled.baseline_acceleration +
                            vec(modified_curvature - original_curvature)
    return merge(assembled,
                 (; baseline_acceleration, contact_angle=angle_degrees,
                  contact_count=count(contact),
                  curvature_reconstruction_error=reconstruction_error))
end

function assemble_pressure_operator(density_calculator;
                                    snapshot_path="/tmp/opencode/wetting_no_paper_final.jls",
                                    ghost_shift=0.0, wall_quadrature=:dummy)
    boundary_mass = corrected_boundary_mass()
    config = case_config("wetting_no")
    semi,
    ode_state = diagnostic_setup(config;
                                 setup_overrides=(;
                                                  fluid_density_calculator=density_calculator,
                                                  boundary_hydrodynamic_mass=boundary_mass))
    v_ode, u_ode = ode_state
    snapshot = open(deserialize, snapshot_path)
    frame = first(snapshot.frames)
    fluid_system = semi.systems[1]
    v_fluid = TP.wrap_v(v_ode, fluid_system, semi)
    u_fluid = TP.wrap_u(u_ode, fluid_system, semi)
    dimensions = TP.ndims(fluid_system)
    v_fluid[1:dimensions, :] .= frame.systems[1].velocity
    u_fluid .= frame.systems[1].coordinates
    TP.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    setup = (; semi, fluid_system, v_ode, u_ode)
    boundary_system = semi.systems[2]
    quantities = force_components(v_ode, u_ode, semi, 0.0)
    n_fluid = TP.nparticles(fluid_system)
    operator = zeros(eltype(fluid_system), dimensions * n_fluid, n_fluid)

    fluid_coordinates = TP.current_coordinates(u_fluid, fluid_system)
    fluid_density = [TP.current_density(v_fluid, fluid_system, particle)
                     for particle in TP.eachparticle(fluid_system)]

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
        if density_calculator isa SummationDensity
            coefficient_particle = -neighbor_mass / fluid_density[particle]^2
            coefficient_neighbor = -neighbor_mass / fluid_density[neighbor]^2
        else
            coefficient_particle = -neighbor_mass /
                                   (fluid_density[particle] * fluid_density[neighbor])
            coefficient_neighbor = coefficient_particle
        end
        for dimension in 1:dimensions
            row = dimension + dimensions * (particle - 1)
            operator[row, particle] += coefficient_particle * gradient[dimension]
            operator[row, neighbor] += coefficient_neighbor * gradient[dimension]
        end
    end

    if wall_quadrature == :semi_analytical || wall_quadrature == :mirrored_ghosts
        boundary_acceleration = zeros(eltype(fluid_system), dimensions, n_fluid)
        if wall_quadrature == :semi_analytical
            add_semi_analytical_wall!(operator, boundary_acceleration, fluid_system,
                                      fluid_coordinates, fluid_density, density_calculator)
        else
            add_mirrored_ghost_wall!(operator, boundary_acceleration, fluid_system,
                                     fluid_coordinates, fluid_density, density_calculator)
        end
        nonpressure_acceleration = quantities.total - quantities.pressure_boundary
        baseline_acceleration = vec(nonpressure_acceleration + boundary_acceleration)
        return (; operator, baseline_acceleration, setup, quantities, fluid_density,
                boundary_reconstruction_error=NaN)
    elseif wall_quadrature != :dummy
        throw(ArgumentError("unknown wall quadrature: $wall_quadrature"))
    end

    pressure_weights,
    boundary_pressure = if iszero(ghost_shift)
        adami_pressure_weights(fluid_system, boundary_system, v_ode, u_ode, semi)
    else
        reflected_adami_pressure_weights(fluid_system, boundary_system, v_ode, u_ode, semi,
                                         ghost_shift)
    end
    boundary_model = boundary_system.boundary_model
    boundary_state_equation = boundary_model.state_equation
    boundary_density = TP.inverse_state_equation.(Ref(boundary_state_equation),
                                                  boundary_pressure)
    boundary_mass = boundary_model.hydrodynamic_mass
    u_boundary = TP.wrap_u(u_ode, boundary_system, semi)
    boundary_coordinates = TP.current_coordinates(u_boundary, boundary_system)

    TP.foreach_point_neighbor(fluid_system, boundary_system, fluid_coordinates,
                              boundary_coordinates, semi;
                              points=TP.each_integrated_particle(fluid_system)) do particle,
                                                                                   boundary_particle,
                                                                                   pos_diff,
                                                                                   distance
        gradient = TP.smoothing_kernel_grad_unsafe(fluid_system, pos_diff, distance,
                                                   particle)
        if density_calculator isa SummationDensity
            direct_coefficient = -boundary_mass[boundary_particle] /
                                 fluid_density[particle]^2
            pressure_derivative = boundary_pressure_derivative(boundary_state_equation,
                                                               boundary_pressure[boundary_particle])
        else
            direct_coefficient = -boundary_mass[boundary_particle] /
                                 (fluid_density[particle] *
                                  boundary_density[boundary_particle])
            pressure_derivative = boundary_pressure_derivative_continuity(boundary_state_equation,
                                                                          boundary_pressure[boundary_particle],
                                                                          fluid_density[particle])
        end
        boundary_coefficient = -boundary_mass[boundary_particle] * pressure_derivative
        for dimension in 1:dimensions
            row = dimension + dimensions * (particle - 1)
            operator[row, particle] += direct_coefficient * gradient[dimension]
            factor = boundary_coefficient * gradient[dimension]
            for fluid_pressure_particle in axes(pressure_weights, 2)
                weight = pressure_weights[boundary_particle, fluid_pressure_particle]
                iszero(weight) && continue
                operator[row, fluid_pressure_particle] += factor * weight
            end
        end
    end

    boundary_acceleration = zeros(eltype(fluid_system), dimensions, n_fluid)
    TP.foreach_point_neighbor(fluid_system, boundary_system, fluid_coordinates,
                              boundary_coordinates, semi;
                              points=TP.each_integrated_particle(fluid_system)) do particle,
                                                                                   boundary_particle,
                                                                                   pos_diff,
                                                                                   distance
        gradient = TP.smoothing_kernel_grad_unsafe(fluid_system, pos_diff, distance,
                                                   particle)
        pressure_factor = if density_calculator isa SummationDensity
            boundary_pressure[boundary_particle] / boundary_density[boundary_particle]^2
        else
            boundary_pressure[boundary_particle] /
            (fluid_density[particle] * boundary_density[boundary_particle])
        end
        boundary_acceleration[:,
                              particle] .-= boundary_mass[boundary_particle] *
                                            pressure_factor .* gradient
    end
    nonpressure_acceleration = quantities.total - quantities.pressure_boundary
    baseline_acceleration = vec(nonpressure_acceleration + boundary_acceleration)
    boundary_reconstruction_error = iszero(ghost_shift) ?
                                    maximum(abs,
                                            boundary_acceleration -
                                            quantities.pressure_boundary) : NaN
    return (; operator, baseline_acceleration, setup, quantities, fluid_density,
            boundary_reconstruction_error)
end

function estimate_lipschitz(operator; iterations=40)
    vector = fill(inv(sqrt(size(operator, 2))), size(operator, 2))
    eigenvalue = zero(eltype(operator))
    for _ in 1:iterations
        product = transpose(operator) * (operator * vector)
        eigenvalue = norm(product)
        vector .= product ./ eigenvalue
    end
    return eigenvalue
end

function nonnegative_least_squares(operator, target; initial_pressure=nothing,
                                   max_iterations=5000, tolerance=1.0e-9)
    pressure = isnothing(initial_pressure) ? zeros(size(operator, 2)) :
               max.(copy(initial_pressure), 0)
    extrapolated = copy(pressure)
    momentum = one(eltype(operator))
    lipschitz = estimate_lipschitz(operator)

    for iteration in 1:max_iterations
        gradient = transpose(operator) * (operator * extrapolated - target)
        pressure_new = max.(extrapolated .- gradient ./ lipschitz, 0)
        relative_change = norm(pressure_new - pressure) / max(norm(pressure), 1)
        momentum_new = (1 + sqrt(1 + 4momentum^2)) / 2
        extrapolated .= pressure_new .+
                        ((momentum - 1) / momentum_new) .* (pressure_new - pressure)
        pressure .= pressure_new
        momentum = momentum_new
        relative_change < tolerance && return pressure, iteration, lipschitz
    end

    return pressure, max_iterations, lipschitz
end

function print_equilibrium_summary(label, assembled, pressure_kpa)
    operator_kpa = 1000assembled.operator
    residual = assembled.baseline_acceleration + operator_kpa * pressure_kpa
    residual_vectors = reshape(residual, TP.ndims(assembled.setup.fluid_system), :)
    residual_magnitude = vec(sqrt.(sum(abs2, residual_vectors; dims=1)))
    baseline_vectors = reshape(assembled.baseline_acceleration,
                               TP.ndims(assembled.setup.fluid_system), :)
    baseline_magnitude = vec(sqrt.(sum(abs2, baseline_vectors; dims=1)))

    @printf("%s\n", label)
    @printf("  pressure [Pa]: median %.3f, p90 %.3f, max %.3f, active %d/%d\n",
            1000median(pressure_kpa), 1000quantile(pressure_kpa, 0.9),
            1000maximum(pressure_kpa), count(>(0), pressure_kpa), length(pressure_kpa))
    @printf("  acceleration magnitude baseline/residual RMS: %.4f / %.4f m/s^2\n",
            sqrt(mean(abs2, baseline_magnitude)), sqrt(mean(abs2, residual_magnitude)))
    @printf("  residual magnitude median %.4f, p90 %.4f, max %.4f m/s^2\n",
            median(residual_magnitude), quantile(residual_magnitude, 0.9),
            maximum(residual_magnitude))
    @printf("  residual mean z: %.6f m/s^2\n", mean(residual_vectors[end, :]))
    isfinite(assembled.boundary_reconstruction_error) &&
        @printf("  boundary-pressure reconstruction error: %.3e m/s^2\n",
                assembled.boundary_reconstruction_error)

    return residual_vectors
end

function solve_equilibrium(snapshot_path, output_path)
    particle_spacing = cbrt(1.0e-6 / 750)
    configurations = (("SummationDensity", SummationDensity(), 0.0, :dummy),
                      ("ContinuityDensity", ContinuityDensity(), 0.0, :dummy),
                      ("ContinuityDensity reflected mDBC", ContinuityDensity(),
                       particle_spacing, :dummy),
                      ("SummationDensity semi-analytical wall", SummationDensity(), 0.0,
                       :semi_analytical),
                      ("ContinuityDensity semi-analytical wall", ContinuityDensity(), 0.0,
                       :semi_analytical),
                      ("ContinuityDensity mirrored ghost forces", ContinuityDensity(), 0.0,
                       :mirrored_ghosts))
    results = map(configurations) do configuration
        label, density_calculator, ghost_shift, wall_quadrature = configuration
        assembled = assemble_pressure_operator(density_calculator; snapshot_path,
                                               ghost_shift, wall_quadrature)
        operator_kpa = 1000assembled.operator
        target = -assembled.baseline_acceleration
        radius = cbrt(3.0e-6 / (4pi))
        initial_pressure = fill(2 / radius / 1000, size(operator_kpa, 2))
        pressure_kpa, iterations,
        lipschitz = nonnegative_least_squares(operator_kpa,
                                              target;
                                              initial_pressure)
        residual = print_equilibrium_summary(label, assembled, pressure_kpa)
        normal_matrix = transpose(operator_kpa) * operator_kpa
        normal_rhs = transpose(operator_kpa) * target
        unconstrained_pressure = (normal_matrix + 1.0e-12I) \ normal_rhs
        unconstrained_residual = assembled.baseline_acceleration +
                                 operator_kpa * unconstrained_pressure
        unconstrained_vectors = reshape(unconstrained_residual,
                                        TP.ndims(assembled.setup.fluid_system), :)
        @printf("  unconstrained residual RMS %.4f, mean z %.6f, negative pressure %d/%d\n",
                sqrt(mean(abs2, unconstrained_residual)),
                mean(unconstrained_vectors[end, :]), count(<(0), unconstrained_pressure),
                length(unconstrained_pressure))
        density = TP.inverse_state_equation.(Ref(assembled.setup.fluid_system.state_equation),
                                             1000pressure_kpa)
        (; label, pressure=1000pressure_kpa, density, residual, iterations, lipschitz,
         baseline_acceleration=assembled.baseline_acceleration,
         unconstrained_pressure=1000unconstrained_pressure,
         unconstrained_residual)
    end

    open(output_path, "w") do io
        serialize(io, (; results))
    end
    println("Wrote pressure equilibrium analysis to ", output_path)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: pressure_equilibrium.jl SNAPSHOT.jls OUTPUT.jls")
    solve_equilibrium(ARGS[1], ARGS[2])
end
