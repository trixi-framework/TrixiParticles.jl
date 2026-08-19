module CorrectionConvergence

using TrixiParticles
using LinearAlgebra: norm
using Printf: @printf, @sprintf

export run_convergence, print_report, write_csv

function field(position)
    2.0 + position[1] + 0.5 * position[2] + position[1]^2 -
    0.25 * position[1] * position[2] + 0.75 * position[2]^2 +
    0.2 * position[1]^3 - 0.1 * position[1]^2 * position[2] +
    0.15 * position[1] * position[2]^2 - 0.05 * position[2]^3
end

function field_gradient(position)
    return SVector(1.0 + 2.0 * position[1] - 0.25 * position[2] +
                   0.6 * position[1]^2 - 0.2 * position[1] * position[2] +
                   0.15 * position[2]^2,
                   0.5 - 0.25 * position[1] + 1.5 * position[2] -
                   0.1 * position[1]^2 + 0.3 * position[1] * position[2] -
                   0.15 * position[2]^2)
end

function density_field(position)
    return 1000.0 * (1.0 + 0.1 * position[1] + 0.05 * position[2] +
            0.02 * position[1]^2 - 0.01 * position[1] * position[2] +
            0.015 * position[2]^2)
end

function free_surface_pressure(position)
    return position[1] + 0.5 * position[1] * position[2] + 0.2 * position[1]^3 +
           0.1 * position[1] * position[2]^2
end

function free_surface_pressure_gradient(position)
    return SVector(1.0 + 0.5 * position[2] + 0.6 * position[1]^2 +
                   0.1 * position[2]^2,
                   0.5 * position[1] + 0.2 * position[1] * position[2])
end

function setup_operator(n, correction; density_calculator=ContinuityDensity())
    particle_spacing = 1.0 / n
    smoothing_length = 2.0 * particle_spacing
    smoothing_kernel = WendlandC6Kernel{2}()
    fluid = RectangularShape(particle_spacing, (n, n), (0.0, 0.0); density=1000.0)
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator,
                                         state_equation, correction)
    semi = Semidiscretization(system; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v_ode = Array(ode.u0.x[1])
    u_ode = Array(ode.u0.x[2])
    semi = ode.p.semi
    system = first(semi.systems)
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    return (; system, semi, v_ode, u_ode, particle_spacing)
end

function pressure_operator_errors(n, correction, density_calculator,
                                  pressure_formulation)
    setup = setup_operator(n, correction; density_calculator)
    (; system, semi, v_ode, u_ode, particle_spacing) = setup
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    n_particles = TrixiParticles.nparticles(system)
    pressure = [free_surface_pressure(SVector{2}(view(coordinates, :, particle)))
                for particle in TrixiParticles.eachparticle(system)]
    constant_pressure = fill(2.0, n_particles)
    exact_acceleration = zeros(2, n_particles)
    for particle in TrixiParticles.eachparticle(system)
        exact_acceleration[:,
                           particle] = -free_surface_pressure_gradient(SVector{2}(view(coordinates,
                                                                                       :,
                                                                                       particle))) /
                                       TrixiParticles.current_density(v, system, particle)
    end

    acceleration = zeros(2, n_particles)
    constant_acceleration = zeros(2, n_particles)
    gradient_correction = TrixiParticles.correction_gradient(system.correction)
    asymmetric = gradient_correction isa Union{KernelCorrection, GradientCorrection,
                       BlendedGradientCorrection,
                       MixedKernelGradientCorrection}
    GC.@preserve v_ode u_ode begin
        TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                              semi) do particle, neighbor, pos_diff,
                                                       distance
            m_a = TrixiParticles.hydrodynamic_mass(system, particle)
            m_b = TrixiParticles.hydrodynamic_mass(system, neighbor)
            rho_a = TrixiParticles.current_density(v, system, particle)
            rho_b = TrixiParticles.current_density(v, system, neighbor)
            W_a = TrixiParticles.smoothing_kernel_grad(system, SVector(pos_diff), distance,
                                                       particle)

            pressure_acceleration = if asymmetric
                W_b = TrixiParticles.smoothing_kernel_grad(system, SVector(-pos_diff),
                                                           distance, neighbor)
                pressure_formulation(m_a, m_b, rho_a, rho_b, pressure[particle],
                                     pressure[neighbor], W_a, W_b)
            else
                pressure_formulation(m_a, m_b, rho_a, rho_b, pressure[particle],
                                     pressure[neighbor], W_a)
            end
            constant_pressure_acceleration = if asymmetric
                W_b = TrixiParticles.smoothing_kernel_grad(system, SVector(-pos_diff),
                                                           distance, neighbor)
                pressure_formulation(m_a, m_b, rho_a, rho_b,
                                     constant_pressure[particle],
                                     constant_pressure[neighbor], W_a, W_b)
            else
                pressure_formulation(m_a, m_b, rho_a, rho_b,
                                     constant_pressure[particle],
                                     constant_pressure[neighbor], W_a)
            end

            for dimension in 1:2
                acceleration[dimension, particle] += pressure_acceleration[dimension]
                constant_acceleration[dimension,
                                      particle] += constant_pressure_acceleration[dimension]
            end
        end
    end

    support = TrixiParticles.compact_support(system, system)
    boundary = [particle
                for particle in axes(coordinates, 2)
                if isapprox(coordinates[1, particle], minimum(view(coordinates, 1, :));
                            atol=eps()) &&
                   2 * support < coordinates[2, particle] < 1.0 - 2 * support]
    # The conservative asymmetric formulation uses the correction matrix of both particles.
    # Keep both neighborhoods away from unrelated boundaries in both samples.
    interior = [particle
                for particle in axes(coordinates, 2)
                if 2 * support < coordinates[1, particle] < 1.0 - 2 * support &&
                   2 * support < coordinates[2, particle] < 1.0 - 2 * support]
    isempty(boundary) && error("resolution $n has no particles in the boundary sample")
    isempty(interior) &&
        error("resolution $n has no particles in the pressure interior sample")

    function sample_errors(particles)
        manufactured = normalized_l2(acceleration[:, particles],
                                     exact_acceleration[:, particles])
        constant = norm(constant_acceleration[:, particles]) / sqrt(length(particles))
        return (; manufactured, constant)
    end

    return (; particle_spacing, boundary=sample_errors(boundary),
            interior=sample_errors(interior))
end

function setup_summation_density(n, correction)
    particle_spacing = 1.0 / n
    smoothing_length = 2.0 * particle_spacing
    smoothing_kernel = WendlandC6Kernel{2}()
    shape = RectangularShape(particle_spacing, (n, n), (0.0, 0.0); density=1000.0)
    fluid = InitialCondition(; coordinates=shape.coordinates, density=density_field,
                             particle_spacing)
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=SummationDensity(),
                                         state_equation, correction)
    semi = Semidiscretization(system; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v_ode = Array(ode.u0.x[1])
    u_ode = Array(ode.u0.x[2])
    semi = ode.p.semi
    system = first(semi.systems)
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    return (; system, semi, v_ode, u_ode, particle_spacing)
end

function setup_continuity_density(n)
    particle_spacing = 1.0 / n
    smoothing_length = 2.0 * particle_spacing
    smoothing_kernel = WendlandC6Kernel{2}()
    shape = RectangularShape(particle_spacing, (n, n), (0.0, 0.0); density=1000.0)
    fluid = InitialCondition(; coordinates=shape.coordinates, density=density_field,
                             particle_spacing)
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation)
    semi = Semidiscretization(system; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v_ode = Array(ode.u0.x[1])
    u_ode = Array(ode.u0.x[2])
    semi = ode.p.semi
    system = first(semi.systems)

    return (; system, semi, v_ode, u_ode, particle_spacing)
end

function sample_regions(coordinates, support, n)
    min_x = minimum(view(coordinates, 1, :))
    boundary = [particle
                for particle in axes(coordinates, 2)
                if isapprox(coordinates[1, particle], min_x; atol=eps()) &&
                   support < coordinates[2, particle] < 1.0 - support]
    isempty(boundary) && error("resolution $n has no particles in the boundary sample")
    interior = [particle
                for particle in axes(coordinates, 2)
                if support < coordinates[1, particle] < 1.0 - support &&
                   support < coordinates[2, particle] < 1.0 - support]
    isempty(interior) && error("resolution $n has no particles in the interior sample")

    return (; boundary, interior)
end

function operator_errors(n, correction)
    setup = setup_operator(n, correction)
    (; system, semi, v_ode, u_ode, particle_spacing) = setup
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    n_particles = TrixiParticles.nparticles(system)
    values = [field(SVector{2}(view(coordinates, :, particle)))
              for particle in TrixiParticles.eachparticle(system)]
    exact_gradients = zeros(2, n_particles)
    for particle in TrixiParticles.eachparticle(system)
        exact_gradients[:,
                        particle] = field_gradient(SVector{2}(view(coordinates, :,
                                                                   particle)))
    end

    interpolation = zeros(n_particles)
    kernel_coefficient = zeros(n_particles)
    direct_gradient = zeros(2, n_particles)
    difference_gradient = zeros(2, n_particles)
    GC.@preserve v_ode u_ode begin
        TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                              semi) do particle, neighbor, pos_diff,
                                                       distance
            pos_diff_ = SVector(pos_diff)
            volume = TrixiParticles.hydrodynamic_mass(system, neighbor) /
                     TrixiParticles.current_density(v, system, neighbor)
            kernel = TrixiParticles.smoothing_kernel(system, distance, particle)
            gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff_, distance,
                                                            particle)

            interpolation[particle] += volume * values[neighbor] * kernel
            kernel_coefficient[particle] += volume * kernel
            for dimension in 1:2
                direct_gradient[dimension,
                                particle] += volume * values[neighbor] * gradient[dimension]
                difference_gradient[dimension,
                                    particle] += volume *
                                                 (values[neighbor] - values[particle]) *
                                                 gradient[dimension]
            end
        end
    end

    support = TrixiParticles.compact_support(system, system)
    (; boundary, interior) = sample_regions(coordinates, support, n)
    normalized_interpolation = interpolation ./ kernel_coefficient

    function sample_errors(particles)
        exact_values = values[particles]

        interpolation_error = normalized_l2(interpolation[particles], exact_values)
        shepard_error = normalized_l2(normalized_interpolation[particles], exact_values)
        direct_error = normalized_l2(direct_gradient[:, particles],
                                     exact_gradients[:, particles])
        difference_error = normalized_l2(difference_gradient[:, particles],
                                         exact_gradients[:, particles])

        return (; interpolation_error, shepard_error, direct_error, difference_error)
    end

    return (; particle_spacing, boundary=sample_errors(boundary),
            interior=sample_errors(interior))
end

function summation_density_errors(n, correction)
    setup = setup_summation_density(n, correction)
    (; system, semi, v_ode, u_ode, particle_spacing) = setup
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    density = GC.@preserve v_ode u_ode begin
        collect(TrixiParticles.current_density(v, system))
    end
    exact_density = [density_field(SVector{2}(view(coordinates, :, particle)))
                     for particle in TrixiParticles.eachparticle(system)]
    support = TrixiParticles.compact_support(system, system)
    regions = sample_regions(coordinates, support, n)

    function sample_error(particles)
        return normalized_l2(density[particles], exact_density[particles])
    end

    return (; particle_spacing, boundary=sample_error(regions.boundary),
            interior=sample_error(regions.interior))
end

function reinitialized_density_errors(n)
    setup = setup_continuity_density(n)
    (; system, semi, v_ode, u_ode, particle_spacing) = setup
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    TrixiParticles.reinit_density!(system, v, u, v_ode, u_ode, semi)
    density = GC.@preserve v_ode u_ode begin
        collect(TrixiParticles.current_density(v, system))
    end
    exact_density = [density_field(SVector{2}(view(coordinates, :, particle)))
                     for particle in TrixiParticles.eachparticle(system)]
    support = TrixiParticles.compact_support(system, system)
    regions = sample_regions(coordinates, support, n)

    function sample_error(particles)
        return normalized_l2(density[particles], exact_density[particles])
    end

    return (; particle_spacing, boundary=sample_error(regions.boundary),
            interior=sample_error(regions.interior))
end

function normalized_l2(approximation, exact)
    return norm(approximation - exact) / norm(exact)
end

function correction_name(correction)
    isnothing(correction) && return :none
    correction isa ShepardKernelCorrection && return :shepard
    correction isa KernelCorrection && return :kernel
    correction isa GradientCorrection && return :gradient
    correction isa BlendedGradientCorrection && return :blended
    correction isa MixedKernelGradientCorrection && return :mixed
    correction isa CorrectionConfiguration && return :shepard_mixed
    error("unsupported correction $(typeof(correction))")
end

function append_result!(results, previous, method, operator, region, resolution, spacing,
                        error)
    key = (method, operator, region)
    order = if haskey(previous, key)
        previous_spacing, previous_error = previous[key]
        log(previous_error / error) / log(previous_spacing / spacing)
    else
        NaN
    end
    push!(results, (; method, operator, region, resolution, spacing, error, order))
    previous[key] = (spacing, error)
    return results
end

"""
    run_convergence(; resolutions=(12, 24, 48, 96))

Measure local correction and pressure-acceleration scaling on self-similar regular patches with
fixed `h / Δx`. Boundary and symmetric-interior samples are reported separately. These
measurements are not convergence rates of the complete SPH discretization.
"""
function run_convergence(; resolutions=(12, 24, 48, 96))
    corrections = (nothing, KernelCorrection(), GradientCorrection(),
                   BlendedGradientCorrection(0.5), MixedKernelGradientCorrection())
    results = NamedTuple[]
    previous = Dict{Tuple{Symbol, Symbol, Symbol}, Tuple{Float64, Float64}}()

    for resolution in resolutions
        for correction in corrections
            errors = operator_errors(resolution, correction)
            method = correction_name(correction)
            for region in (:boundary, :interior)
                region_errors = getproperty(errors, region)
                append_result!(results, previous, method, :difference_gradient, region,
                               resolution, errors.particle_spacing,
                               region_errors.difference_error)

                if method in (:none, :kernel, :mixed)
                    append_result!(results, previous, method, :direct_gradient, region,
                                   resolution, errors.particle_spacing,
                                   region_errors.direct_error)
                end

                if method == :none
                    append_result!(results, previous, :none, :interpolation, region,
                                   resolution, errors.particle_spacing,
                                   region_errors.interpolation_error)
                    append_result!(results, previous, :shepard, :interpolation, region,
                                   resolution, errors.particle_spacing,
                                   region_errors.shepard_error)
                end
            end
        end

        for (method, correction) in ((:none, nothing),
             (:shepard, ShepardKernelCorrection()))
            errors = summation_density_errors(resolution, correction)
            for region in (:boundary, :interior)
                append_result!(results, previous, method, :summation_density, region,
                               resolution, errors.particle_spacing,
                               getproperty(errors, region))
            end
        end

        reinitialization_errors = reinitialized_density_errors(resolution)
        for region in (:boundary, :interior)
            append_result!(results, previous, :shepard, :density_reinitialization, region,
                           resolution, reinitialization_errors.particle_spacing,
                           getproperty(reinitialization_errors, region))
        end

        resolution < 24 && continue

        summation_corrections = (nothing, ShepardKernelCorrection(), KernelCorrection(),
                                 GradientCorrection(), BlendedGradientCorrection(0.5),
                                 MixedKernelGradientCorrection(),
                                 CorrectionConfiguration(;
                                                         density=ShepardKernelCorrection(),
                                                         gradient=MixedKernelGradientCorrection()))
        continuity_corrections = (nothing, KernelCorrection(), GradientCorrection(),
                                  BlendedGradientCorrection(0.5),
                                  MixedKernelGradientCorrection())
        pressure_cases = ((:pressure_summation, :constant_pressure_summation,
                           SummationDensity(),
                           TrixiParticles.pressure_acceleration_summation_density,
                           summation_corrections),
                          (:pressure_interparticle_summation,
                           :constant_pressure_interparticle_summation, SummationDensity(),
                           TrixiParticles.inter_particle_averaged_pressure,
                           summation_corrections),
                          (:pressure_continuity, :constant_pressure_continuity,
                           ContinuityDensity(),
                           TrixiParticles.pressure_acceleration_continuity_density,
                           continuity_corrections),
                          (:pressure_interparticle_continuity,
                           :constant_pressure_interparticle_continuity, ContinuityDensity(),
                           TrixiParticles.inter_particle_averaged_pressure,
                           continuity_corrections),
                          (:pressure_tensile_positive, :constant_pressure_tensile,
                           ContinuityDensity(), TrixiParticles.tensile_instability_control,
                           (nothing,)))

        for (operator, constant_operator, density_calculator,
             pressure_formulation, corrections) in pressure_cases

            for correction in corrections
                errors = pressure_operator_errors(resolution, correction,
                                                  density_calculator,
                                                  pressure_formulation)
                method = correction_name(correction)
                for region in (:boundary, :interior)
                    region_errors = getproperty(errors, region)
                    append_result!(results, previous, method, operator, region, resolution,
                                   errors.particle_spacing, region_errors.manufactured)
                    append_result!(results, previous, method, constant_operator, region,
                                   resolution, errors.particle_spacing,
                                   region_errors.constant)
                end
            end
        end
    end

    return results
end

function print_report(results; io=stdout)
    println(io, "| Method | Operator | Region | N | L2 error | Observed scaling |")
    println(io, "|:--|:--|:--|--:|--:|--:|")
    for result in results
        order = isnan(result.order) ? "-" : @sprintf("%.3f", result.order)
        @printf(io, "| %s | %s | %s | %d | %.6e | %s |\n", result.method,
                result.operator, result.region, result.resolution, result.error, order)
    end
    return results
end

function write_csv(filename, results)
    directory = dirname(filename)
    isempty(directory) || mkpath(directory)
    open(filename, "w") do io
        println(io,
                "method,operator,region,resolution,spacing,l2_error,observed_scaling")
        for result in results
            println(io,
                    join((result.method, result.operator, result.region, result.resolution,
                          result.spacing, result.error, result.order), ','))
        end
    end
    return filename
end

end # module CorrectionConvergence

if abspath(PROGRAM_FILE) == @__FILE__
    results = CorrectionConvergence.run_convergence()
    CorrectionConvergence.print_report(results)
    output_file = joinpath("out", "correction_convergence.csv")
    CorrectionConvergence.write_csv(output_file, results)
    println("\nWrote $output_file")
end
