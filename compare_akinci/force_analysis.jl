using LinearAlgebra
using OrdinaryDiffEqLowStorageRK
using Printf
using Serialization
using Statistics
using TrixiParticles

include("cases.jl")

const TP = TrixiParticles

function diagnostic_setup(config; setup_overrides=(;))
    simulation_module = Module(Symbol("AkinciForceDiagnostic_", config.name))
    Core.eval(simulation_module, :(using TrixiParticles))
    example = joinpath(examples_dir(), "fluid", config.example)
    kwargs = merge(config.kwargs, setup_overrides,
                   (; tspan=(0.0, 0.0), solution_saveat=(), saving_callback=nothing))
    trixi_include(simulation_module, example; kwargs...)
    solution = Base.invokelatest(Core.eval, simulation_module, :sol)
    return solution.prob.p.semi, deepcopy(solution.prob.u0).x
end

function inject_frame!(v_ode, u_ode, semi, frame, time; system_index=1)
    system = semi.systems[system_index]
    snapshot_system = frame.systems[system_index]
    TP.wrap_v(v_ode, system, semi) .= snapshot_system.velocity
    TP.wrap_u(u_ode, system, semi) .= snapshot_system.coordinates
    TP.update_systems_and_nhs(v_ode, u_ode, semi, time)

    v = TP.wrap_v(v_ode, system, semi)
    density = [TP.current_density(v, system, particle)
               for particle in TP.eachparticle(system)]
    pressure = [TP.current_pressure(v, system, particle)
                for particle in TP.eachparticle(system)]
    density_error = maximum(abs, density - snapshot_system.density)
    pressure_error = maximum(abs, pressure - snapshot_system.pressure)
    return (; density, pressure, density_error, pressure_error)
end

function force_components(v_ode, u_ode, semi, time; system_index=1)
    system = semi.systems[system_index]
    v_system = TP.wrap_v(v_ode, system, semi)
    u_system = TP.wrap_u(u_ode, system, semi)
    coordinates = Array(TP.current_coordinates(u_system, system))
    velocity = Array(TP.current_velocity(v_system, system))
    n_particles = length(TP.eachparticle(system))
    particle_spacing = TP.particle_spacing(system, first(TP.eachparticle(system)))
    dimensions = TP.ndims(system)
    scalar_type = eltype(coordinates)

    pressure_fluid = zeros(scalar_type, dimensions, n_particles)
    pressure_boundary = zero(pressure_fluid)
    viscosity_fluid = zero(pressure_fluid)
    viscosity_boundary = zero(pressure_fluid)
    cohesion = zero(pressure_fluid)
    curvature = zero(pressure_fluid)
    adhesion = zero(pressure_fluid)
    fluid_normal_sum_before_filter = zero(pressure_fluid)
    reference_density_normal_sum_before_filter = zero(pressure_fluid)
    boundary_normal_sum_before_filter = zero(pressure_fluid)
    normal_moment_matrix = zeros(scalar_type, dimensions, dimensions, n_particles)
    smoothed_normal_sum = zero(pressure_fluid)
    smoothed_normal_weight = zeros(scalar_type, n_particles)

    pressure_pair_magnitudes = zeros(scalar_type, n_particles)
    viscosity_pair_magnitudes = zeros(scalar_type, n_particles)
    cohesion_pair_magnitudes = zeros(scalar_type, n_particles)
    curvature_pair_magnitudes = zeros(scalar_type, n_particles)
    adhesion_pair_magnitudes = zeros(scalar_type, n_particles)
    normal_pair_magnitudes = zeros(scalar_type, n_particles)
    fluid_neighbor_count = zeros(Int, n_particles)
    boundary_neighbor_count = zeros(Int, n_particles)
    normal_boundary_neighbor_count = zeros(Int, n_particles)
    surface_correction_sum = zeros(scalar_type, n_particles)
    surface_correction_min = fill(typemax(scalar_type), n_particles)
    surface_correction_max = fill(typemin(scalar_type), n_particles)

    correction = system.correction
    sound_speed = TP.system_sound_speed(system)
    surface_tension = TP.surface_tension_model(system)
    surface_tension isa SurfaceTensionAkinci ||
        error("force decomposition currently requires `SurfaceTensionAkinci`")
    raw_normal = copy(system.cache.surface_normal)

    for particle in TP.each_integrated_particle(system)
        volume = TP.hydrodynamic_mass(system, particle) /
                 TP.current_density(v_system, system, particle)
        weight = volume * TP.smoothing_kernel(system, zero(scalar_type), particle)
        smoothed_normal_sum[:, particle] .= weight .* raw_normal[:, particle]
        smoothed_normal_weight[particle] = weight
    end

    system_coordinates = TP.current_coordinates(u_system, system)
    for neighbor_system in semi.systems
        v_neighbor = TP.wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = TP.wrap_u(u_ode, neighbor_system, semi)
        neighbor_coordinates = TP.current_coordinates(u_neighbor, neighbor_system)
        neighborhood_search = TP.get_neighborhood_search(system, neighbor_system, semi)
        compact_support = TP.compact_support(system, neighbor_system)
        almost_zero = sqrt(eps(compact_support^2))
        neighbor_is_fluid = neighbor_system isa TP.AbstractFluidSystem
        normal_threshold = system.surface_normal_method.boundary_contact_threshold
        maximum_boundary_colorfield = if neighbor_is_fluid || normal_threshold == Inf
            zero(scalar_type)
        else
            maximum(neighbor_system.boundary_model.cache.colorfield)
        end

        for particle in TP.each_integrated_particle(system)
            m_a = TP.hydrodynamic_mass(system, particle)
            p_a = TP.current_pressure(v_system, system, particle)
            velocity_a = TP.current_velocity(v_system, system, particle)
            rho_a = TP.current_density(v_system, system, particle)

            TP.foreach_neighbor(system_coordinates, neighbor_coordinates,
                                neighborhood_search, semi.parallelization_backend,
                                particle) do _, neighbor, pos_diff, distance
                TP.skip_zero_distance(system) && distance < almost_zero && return

                grad_kernel = TP.smoothing_kernel_grad_unsafe(system, pos_diff, distance,
                                                              particle)
                m_b = TP.hydrodynamic_mass(neighbor_system, neighbor)
                velocity_b = TP.current_velocity(v_neighbor, neighbor_system, neighbor)
                rho_b = TP.current_density(v_neighbor, neighbor_system, neighbor)
                p_b = TP.neighbor_pressure(v_neighbor, neighbor_system, neighbor, p_a)
                correction_rho_a = TP.correction_density(correction, system, particle,
                                                         rho_a)
                correction_rho_b = TP.correction_density(correction, neighbor_system,
                                                         neighbor, rho_b)
                (viscosity_correction, pressure_correction,
                 surface_correction) = TP.free_surface_correction(correction, system,
                                                                  correction_rho_a,
                                                                  correction_rho_b)

                pressure_acceleration = pressure_correction *
                                        TP.pressure_acceleration(system, neighbor_system,
                                                                 particle, neighbor,
                                                                 m_a, m_b, p_a, p_b,
                                                                 rho_a, rho_b, pos_diff,
                                                                 distance, grad_kernel,
                                                                 correction)
                pressure_component = neighbor_is_fluid ? pressure_fluid : pressure_boundary
                pressure_component[:, particle] .+= pressure_acceleration
                pressure_pair_magnitudes[particle] += norm(pressure_acceleration)

                viscosity_acceleration = Ref(zero(velocity_a))
                TP.dv_viscosity!(viscosity_acceleration, system, neighbor_system,
                                 v_system, v_neighbor, particle, neighbor, pos_diff,
                                 distance, sound_speed, m_a, m_b, rho_a, rho_b,
                                 velocity_a, velocity_b, grad_kernel,
                                 viscosity_correction)
                viscosity_component = neighbor_is_fluid ? viscosity_fluid :
                                      viscosity_boundary
                viscosity_component[:, particle] .+= viscosity_acceleration[]
                viscosity_pair_magnitudes[particle] += norm(viscosity_acceleration[])

                adhesion_acceleration = Ref(zero(velocity_a))
                TP.adhesion_force!(adhesion_acceleration, surface_tension, system,
                                   neighbor_system, particle, neighbor, pos_diff, distance)
                adhesion[:, particle] .+= adhesion_acceleration[]
                adhesion_pair_magnitudes[particle] += norm(adhesion_acceleration[])

                if neighbor_is_fluid
                    fluid_neighbor_count[particle] += 1
                    normal_contribution = m_b / rho_b * grad_kernel
                    fluid_normal_sum_before_filter[:, particle] .+= normal_contribution
                    reference_density = neighbor_system.state_equation.reference_density
                    reference_density_normal_sum_before_filter[:,
                                                               particle] .+= m_b /
                                                                             reference_density *
                                                                             grad_kernel
                    normal_pair_magnitudes[particle] += norm(normal_contribution)

                    volume = m_b / rho_b
                    kernel_weight = TP.smoothing_kernel(system, distance, particle)
                    smoothed_normal_sum[:,
                                        particle] .+= volume * kernel_weight .*
                                                      raw_normal[:, neighbor]
                    smoothed_normal_weight[particle] += volume * kernel_weight
                    for column in 1:dimensions, row in 1:dimensions
                        normal_moment_matrix[row, column,
                                             particle] -= volume * grad_kernel[row] *
                                                          pos_diff[column]
                    end

                    surface_correction_sum[particle] += surface_correction
                    surface_correction_min[particle] = min(surface_correction_min[particle],
                                                           surface_correction)
                    surface_correction_max[particle] = max(surface_correction_max[particle],
                                                           surface_correction)

                    support_radius = TP.compact_support(system.smoothing_kernel,
                                                        TP.smoothing_length(system,
                                                                            particle))
                    cohesion_acceleration = surface_correction *
                                            TP.cohesion_force_akinci(surface_tension,
                                                                     support_radius, m_b,
                                                                     pos_diff, distance,
                                                                     Val(dimensions))
                    cohesion[:, particle] .+= cohesion_acceleration
                    cohesion_pair_magnitudes[particle] += norm(cohesion_acceleration)

                    normal_a = TP.akinci_surface_normal(system, particle)
                    normal_b = TP.akinci_surface_normal(neighbor_system, neighbor)
                    curvature_acceleration = -surface_correction *
                                             surface_tension.surface_tension_coefficient *
                                             (normal_a - normal_b)
                    curvature[:, particle] .+= curvature_acceleration
                    curvature_pair_magnitudes[particle] += norm(curvature_acceleration)
                else
                    boundary_neighbor_count[particle] += 1
                    if normal_threshold < Inf
                        boundary_colorfield = neighbor_system.boundary_model.cache.colorfield
                        include_boundary = boundary_colorfield[neighbor] /
                                           maximum_boundary_colorfield > normal_threshold
                    else
                        include_boundary = false
                    end
                    if include_boundary
                        normal_contribution = m_a / rho_a * grad_kernel
                        boundary_normal_sum_before_filter[:,
                                                          particle] .+= normal_contribution
                        normal_pair_magnitudes[particle] += norm(normal_contribution)
                        normal_boundary_neighbor_count[particle] += 1
                    end
                end
            end
        end
    end

    surface_correction_mean = surface_correction_sum ./ max.(fluid_neighbor_count, 1)
    surface_correction_min[iszero.(fluid_neighbor_count)] .= 0
    surface_correction_max[iszero.(fluid_neighbor_count)] .= 0

    smoothed_normal = smoothed_normal_sum ./ reshape(smoothed_normal_weight, 1, :)
    gradient_corrected_normal = similar(raw_normal)
    for particle in TP.eachparticle(system)
        moment = normal_moment_matrix[:, :, particle]
        correction_matrix = abs(det(moment)) < 1.0f-9 ?
                            Matrix{scalar_type}(I, dimensions, dimensions) : inv(moment)
        gradient_corrected_normal[:,
                                  particle] .= correction_matrix *
                                               fluid_normal_sum_before_filter[:, particle]
    end

    reconstructed_neighbor_count = fluid_neighbor_count .+ 1 .+
                                   normal_boundary_neighbor_count
    invalid_normal = reconstructed_neighbor_count .< 2^dimensions + 1
    gradient_corrected_normal[:, invalid_normal] .= 0
    smoothed_normal[:, invalid_normal] .= 0

    scaled_normal = similar(raw_normal)
    scaled_fluid_normal = similar(raw_normal)
    scaled_reference_density_normal = similar(raw_normal)
    scaled_gradient_corrected_normal = similar(raw_normal)
    scaled_smoothed_normal = similar(raw_normal)
    scaled_boundary_normal = similar(raw_normal)
    for particle in TP.eachparticle(system)
        support_radius = TP.compact_support(system.smoothing_kernel,
                                            TP.smoothing_length(system, particle))
        scaled_normal[:, particle] .= support_radius .* raw_normal[:, particle]
        scaled_fluid_normal[:,
                            particle] .= support_radius .*
                                         fluid_normal_sum_before_filter[:, particle]
        scaled_reference_density_normal[:,
                                        particle] .= support_radius .*
                                                     reference_density_normal_sum_before_filter[:,
                                                                                                particle]
        scaled_gradient_corrected_normal[:,
                                         particle] .= support_radius .*
                                                      gradient_corrected_normal[:, particle]
        scaled_smoothed_normal[:,
                               particle] .= support_radius .* smoothed_normal[:, particle]
        scaled_boundary_normal[:,
                               particle] .= support_radius .*
                                            boundary_normal_sum_before_filter[:,
                                                                              particle]
    end

    gravity = repeat(collect(system.acceleration), 1, n_particles)
    total = pressure_fluid + pressure_boundary + viscosity_fluid +
            viscosity_boundary + cohesion + curvature + adhesion + gravity

    dv_ode = zero(v_ode)
    TP.system_interaction!(dv_ode, v_ode, u_ode, semi)
    TP.add_source_terms!(dv_ode, v_ode, u_ode, semi, time)
    rhs = Array(TP.wrap_v(dv_ode, system, semi))[1:dimensions, :]

    reconstructed_normal = fluid_normal_sum_before_filter +
                           boundary_normal_sum_before_filter
    reconstructed_normal[:, reconstructed_neighbor_count .< 2^dimensions + 1] .= 0
    normal_residual = maximum(abs, raw_normal - reconstructed_normal)
    rhs_residual = maximum(abs, rhs - total)

    return (; coordinates, velocity, particle_spacing, raw_normal, scaled_normal,
            scaled_fluid_normal, scaled_reference_density_normal, scaled_boundary_normal,
            scaled_gradient_corrected_normal, scaled_smoothed_normal,
            fluid_normal_sum_before_filter, reference_density_normal_sum_before_filter,
            boundary_normal_sum_before_filter, normal_moment_matrix,
            smoothed_normal_weight,
            normal_neighbor_count=copy(system.cache.neighbor_count), fluid_neighbor_count,
            boundary_neighbor_count, normal_boundary_neighbor_count,
            surface_correction_mean,
            surface_correction_min, surface_correction_max,
            pressure_fluid, pressure_boundary, viscosity_fluid, viscosity_boundary,
            cohesion, curvature, adhesion, gravity, total, rhs,
            pressure_pair_magnitudes, viscosity_pair_magnitudes,
            cohesion_pair_magnitudes, curvature_pair_magnitudes,
            adhesion_pair_magnitudes, normal_pair_magnitudes,
            normal_residual, rhs_residual)
end

vector_magnitudes(values) = vec(sqrt.(sum(abs2, values; dims=1)))

function print_distribution(label, values)
    magnitudes = vector_magnitudes(values)
    @printf("  %-20s median=%10.4g  p90=%10.4g  p99=%10.4g  max=%10.4g\n",
            label, median(magnitudes), quantile(magnitudes, 0.9),
            quantile(magnitudes, 0.99), maximum(magnitudes))
end

function radial_components(values, coordinates)
    center_x = median(coordinates[1, :])
    center_y = median(coordinates[2, :])
    offset_x = coordinates[1, :] .- center_x
    offset_y = coordinates[2, :] .- center_y
    radii = hypot.(offset_x, offset_y)
    mask = radii .> sqrt(eps(eltype(radii)))
    radial = (values[1, mask] .* offset_x[mask] +
              values[2, mask] .* offset_y[mask]) ./ radii[mask]
    return radial
end

function print_directions(label, values, coordinates)
    radial = radial_components(values, coordinates)
    @printf("    %-18s radial median=%9.4g, mean=%9.4g; z median=%9.4g, mean=%9.4g\n",
            label, median(radial), mean(radial), median(values[3, :]),
            mean(values[3, :]))
end

function print_summary(time, state, quantities)
    @printf("\nt = %.3f s\n", time)
    @printf("  reconstruction: density error %.3e, pressure error %.3e, normal error %.3e, RHS error %.3e\n",
            state.density_error, state.pressure_error, quantities.normal_residual,
            quantities.rhs_residual)
    print_distribution("scaled normal", quantities.scaled_normal)
    print_distribution("normal from fluid", quantities.scaled_fluid_normal)
    print_distribution("normal at rho0", quantities.scaled_reference_density_normal)
    print_distribution("gradient-corrected", quantities.scaled_gradient_corrected_normal)
    print_distribution("Shepard-smoothed", quantities.scaled_smoothed_normal)
    if any(x -> !iszero(x), quantities.scaled_boundary_normal)
        print_distribution("normal from wall", quantities.scaled_boundary_normal)
    end
    print_distribution("pressure fluid", quantities.pressure_fluid)
    print_distribution("pressure boundary", quantities.pressure_boundary)
    print_distribution("viscosity fluid", quantities.viscosity_fluid)
    print_distribution("viscosity boundary", quantities.viscosity_boundary)
    print_distribution("cohesion", quantities.cohesion)
    print_distribution("curvature", quantities.curvature)
    print_distribution("adhesion", quantities.adhesion)
    print_distribution("total", quantities.total)
    println("  signed components (positive radial is outward, positive z is upward):")
    print_directions("pressure fluid", quantities.pressure_fluid, quantities.coordinates)
    print_directions("pressure boundary", quantities.pressure_boundary,
                     quantities.coordinates)
    print_directions("cohesion", quantities.cohesion, quantities.coordinates)
    print_directions("curvature", quantities.curvature, quantities.coordinates)
    print_directions("adhesion", quantities.adhesion, quantities.coordinates)
    print_directions("total", quantities.total, quantities.coordinates)

    contact = quantities.boundary_neighbor_count .> 0
    if any(contact)
        @printf("  wall-neighbor particles: %d; median z acceleration pressure %.4g, viscosity %.4g, adhesion %.4g, total %.4g\n",
                count(contact), median(quantities.pressure_boundary[3, contact]),
                median(quantities.viscosity_boundary[3, contact]),
                median(quantities.adhesion[3, contact]),
                median(quantities.total[3, contact]))
    else
        println("  wall-neighbor particles: 0")
    end
    normal_magnitudes = vector_magnitudes(quantities.raw_normal)
    normal_active = normal_magnitudes .> 0
    if any(normal_active)
        cancellation = normal_magnitudes[normal_active] ./
                       quantities.normal_pair_magnitudes[normal_active]
        @printf("  normals: %d active, %d filtered; median net/pair-magnitude ratio %.4f\n",
                count(normal_active), length(normal_active) - count(normal_active),
                median(cancellation))
    else
        println("  normals: 0 active")
    end
    @printf("  neighbors: fluid %s, boundary %s, wall-normal %s, normal-cache %s\n",
            extrema(quantities.fluid_neighbor_count),
            extrema(quantities.boundary_neighbor_count),
            extrema(quantities.normal_boundary_neighbor_count),
            extrema(quantities.normal_neighbor_count))
    @printf("  surface correction: [%.4f, %.4f], median mean %.4f\n",
            minimum(quantities.surface_correction_min),
            maximum(quantities.surface_correction_max),
            median(quantities.surface_correction_mean))
end

function analyze_forces(case_name, snapshot_path, output_path; requested_times=nothing,
                        setup_overrides=(;))
    config = case_config(case_name)
    startswith(config.name, "wetting_") || error("force analysis is limited to Figure 8")
    snapshot = open(deserialize, snapshot_path)
    semi, (v_ode, u_ode) = diagnostic_setup(config; setup_overrides)

    frame_indices = if isnothing(requested_times)
        eachindex(snapshot.times)
    else
        map(requested_times) do requested_time
            index = argmin(abs.(snapshot.times .- requested_time))
            isapprox(snapshot.times[index], requested_time; atol=1.0e-10) ||
                error("snapshot has no frame at t=$requested_time")
            index
        end
    end

    frames = GC.@preserve v_ode u_ode begin
        map(frame_indices) do index
            time = snapshot.times[index]
            state = inject_frame!(v_ode, u_ode, semi, snapshot.frames[index], time)
            quantities = force_components(v_ode, u_ode, semi, time)
            print_summary(time, state, quantities)
            return (; time, density=state.density, pressure=state.pressure, quantities...)
        end
    end

    result = (; case_name, source_snapshot=abspath(snapshot_path), frames)
    open(output_path, "w") do io
        serialize(io, result)
    end
    println("\nWrote per-particle force analysis to $output_path")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    3 <= length(ARGS) <= 4 ||
        error("pass a Figure 8 case, input snapshot, output path, and optional comma-separated times")
    requested_times = length(ARGS) == 4 ? parse.(Float64, split(ARGS[4], ',')) : nothing
    analyze_forces(ARGS[1], ARGS[2], ARGS[3]; requested_times)
end
