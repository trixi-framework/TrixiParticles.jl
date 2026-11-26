module CorrectionConvergence

using TrixiParticles
using LinearAlgebra: norm
using Printf: @printf, @sprintf

export run_convergence, print_report, write_csv

# Smooth across every seam. The positive offset also lets us compare the positive
# branch of tensile instability control with the continuity-density pressure law.
function field(x)
    f = 2 + sinpi(2 * x[1])
    length(x) >= 2 && (f += 0.5 * cospi(2 * x[2]))
    length(x) == 3 && (f += 0.25 * sinpi(2 * x[3]))
    return f
end
field_gradient(x, ::Val{1}) = SVector(2pi * cospi(2 * x[1]))
field_gradient(x, ::Val{2}) = SVector(2pi * cospi(2 * x[1]), -pi * sinpi(2 * x[2]))
field_gradient(x, ::Val{3}) = SVector(field_gradient(x, Val(2))..., 0.5pi * cospi(2 * x[3]))
density_field(x) = 1000 * (1 + 0.1 * (field(x) - 2))
relative_error(approximation, exact) = norm(approximation - exact) / norm(exact)

function setup_operator(n, correction, ::Val{D}; density=1000.0,
                        density_calculator=ContinuityDensity(),
                        pressure_acceleration=nothing) where {D}
    spacing = 1.0 / n
    origin = ntuple(_ -> 0.0, Val(D))
    fluid = RectangularShape(spacing, ntuple(_ -> n, Val(D)), origin; density)
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel=WendlandC6Kernel{D}(),
                                         smoothing_length=2 * spacing, state_equation,
                                         density_calculator, pressure_acceleration,
                                         density_correction=TrixiParticles.correction_density(correction),
                                         gradient_correction=TrixiParticles.correction_gradient(correction))
    periodic_box = PeriodicBox(min_corner=origin, max_corner=ntuple(_ -> 1.0, Val(D)))
    semi = Semidiscretization(system; parallelization_backend=SerialBackend(),
                              neighborhood_search=GridNeighborhoodSearch{D}(; periodic_box))
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v_ode, u_ode = ode.u0.x
    semi = ode.p.semi
    system = only(semi.systems)
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    return (; system, semi, v_ode, u_ode)
end

function operator_errors(n, correction, dim)
    (; system, semi, v_ode, u_ode) = setup_operator(n, correction, dim)
    GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        coords = TrixiParticles.current_coordinates(u, system)
        values = field.(eachcol(coords))
        gradients = map(x -> field_gradient(x, dim), eachcol(coords))
        interpolation = zeros(length(values))
        kernel_sum = zero(interpolation)
        direct = zero(gradients)
        difference = zero(gradients)

        TrixiParticles.foreach_point_neighbor(system, system, coords, coords,
                                              semi) do a, b, pos_diff, distance
            volume = TrixiParticles.hydrodynamic_mass(system, b) /
                     TrixiParticles.current_density(v, system, b)
            weight = volume * TrixiParticles.smoothing_kernel(system, distance, a)
            gradient = volume * TrixiParticles.smoothing_kernel_grad(system, pos_diff,
                                                            distance, a)
            interpolation[a] += weight * values[b]
            kernel_sum[a] += weight
            direct[a] += values[b] * gradient
            difference[a] += (values[b] - values[a]) * gradient
        end

        return (; interpolation=relative_error(interpolation, values),
                shepard=relative_error(interpolation ./ kernel_sum, values),
                direct=relative_error(direct, gradients),
                difference=relative_error(difference, gradients))
    end
end

function density_error(n, correction, density_calculator, dim)
    (; system, semi, v_ode,
     u_ode) = setup_operator(n, correction, dim; density=density_field,
                             density_calculator)
    GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        if density_calculator isa ContinuityDensity
            TrixiParticles.reinit_density!(system, v, u, v_ode, u_ode, semi)
        end
        exact = density_field.(eachcol(TrixiParticles.current_coordinates(u, system)))
        return relative_error(TrixiParticles.current_density(v, system), exact)
    end
end

function pressure_errors(n, correction, density_calculator, pressure_acceleration, dim)
    (; system, semi, v_ode,
     u_ode) = setup_operator(n, correction, dim; density_calculator,
                             pressure_acceleration)
    GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        coords = TrixiParticles.current_coordinates(u, system)
        pressure = field.(eachcol(coords))
        exact = map(x -> -field_gradient(x, dim), eachcol(coords)) ./
                TrixiParticles.current_density(v, system)
        acceleration = zero(exact)
        constant_acceleration = zero(exact)

        TrixiParticles.foreach_point_neighbor(system, system, coords, coords,
                                              semi) do a, b, pos_diff, distance
            m_a = TrixiParticles.hydrodynamic_mass(system, a)
            m_b = TrixiParticles.hydrodynamic_mass(system, b)
            rho_a = TrixiParticles.current_density(v, system, a)
            rho_b = TrixiParticles.current_density(v, system, b)
            W_a = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance, a)
            pair(p_a,
                 p_b) = TrixiParticles.pressure_acceleration(system, system, a, b,
                                                             m_a, m_b, p_a, p_b,
                                                             rho_a, rho_b, pos_diff,
                                                             distance, W_a,
                                                             system.correction)
            acceleration[a] += pair(pressure[a], pressure[b])
            constant_acceleration[a] += pair(2.0, 2.0)
        end

        return (; manufactured=relative_error(acceleration, exact),
                constant=norm(constant_acceleration) / sqrt(length(exact)))
    end
end

default_resolutions(dimension) = dimension == 3 ? (12, 24, 48) : (12, 24, 48, 96)

"""Measure local operator scaling on a periodic unit lattice at fixed `h / Δx = 2`."""
function run_convergence(; dimension=2, resolutions=default_resolutions(dimension))
    dimension isa Int && dimension in 1:3 ||
        throw(ArgumentError("dimension must be 1, 2 or 3"))
    dim = Val(dimension)
    # The periodic grid search needs at least three cells of width 4Δx per axis.
    all(n -> n isa Integer && n >= 12, resolutions) && !isempty(resolutions) &&
    all(>(0), diff(collect(resolutions))) ||
        throw(ArgumentError("resolutions must be strictly increasing integers ≥ 12"))

    gradients = ((:none, nothing), (:kernel, KernelCorrection()),
                 (:gradient, GradientCorrection()),
                 (:blended, BlendedGradientCorrection(0.5)),
                 (:mixed, MixedKernelGradientCorrection()))
    shepard = ShepardKernelCorrection()
    combined = CorrectionConfiguration(; density=shepard,
                                       gradient=MixedKernelGradientCorrection())
    summation = (gradients..., (:shepard, shepard), (:shepard_mixed, combined))
    pressure_cases = ((:summation, SummationDensity(),
                       TrixiParticles.pressure_acceleration_summation_density, summation),
                      (:interparticle_summation, SummationDensity(),
                       TrixiParticles.inter_particle_averaged_pressure, summation),
                      (:continuity, ContinuityDensity(),
                       TrixiParticles.pressure_acceleration_continuity_density, gradients),
                      (:interparticle_continuity, ContinuityDensity(),
                       TrixiParticles.inter_particle_averaged_pressure, gradients),
                      (:tensile_positive, ContinuityDensity(),
                       TrixiParticles.tensile_instability_control, ((:none, nothing),)))

    results = NamedTuple[]
    previous = Dict{Tuple{Symbol, Symbol}, Tuple{Float64, Float64}}()
    for resolution in resolutions
        spacing = 1.0 / resolution
        function record!(method, operator, error)
            key = (method, operator)
            order = NaN
            if haskey(previous, key)
                prev_spacing, prev_error = previous[key]
                if min(error, prev_error) > 0
                    order = log(prev_error / error) / log(prev_spacing / spacing)
                end
            end
            push!(results,
                  (; method, operator, region=:periodic, dimension, resolution, spacing,
                   error, order))
            previous[key] = (spacing, error)
        end

        for (method, correction) in gradients
            errors = operator_errors(resolution, correction, dim)
            record!(method, :difference_gradient, errors.difference)
            if method in (:none, :kernel, :mixed)
                record!(method, :direct_gradient, errors.direct)
            end
            if method == :none
                record!(:none, :interpolation, errors.interpolation)
                record!(:shepard, :interpolation, errors.shepard)
            end
        end
        for (method, correction) in ((:none, nothing), (:shepard, shepard))
            record!(method, :summation_density,
                    density_error(resolution, correction, SummationDensity(), dim))
        end
        record!(:shepard, :density_reinitialization,
                density_error(resolution, nothing, ContinuityDensity(), dim))

        for (name, calculator, formulation, corrections) in pressure_cases
            for (method, correction) in corrections
                errors = pressure_errors(resolution, correction, calculator, formulation,
                                         dim)
                record!(method, Symbol(:pressure_, name), errors.manufactured)
                record!(method, Symbol(:constant_pressure_, name), errors.constant)
            end
        end
    end
    return results
end

function print_report(results; io=stdout)
    println(io,
            "| Method | Operator | Region | Dimension | N | L2 error | Observed scaling |")
    println(io, "|:--|:--|:--|--:|--:|--:|--:|")
    for result in results
        order = isnan(result.order) ? "-" : @sprintf("%.3f", result.order)
        @printf(io, "| %s | %s | %s | %d | %d | %.6e | %s |\n", result.method,
                result.operator, result.region, result.dimension, result.resolution,
                result.error, order)
    end
    return results
end

function write_csv(filename, results)
    mkpath(dirname(abspath(filename)))
    open(filename, "w") do io
        println(io,
                "method,operator,region,dimension,resolution,spacing,l2_error,observed_scaling")
        for result in results
            println(io,
                    join((result.method, result.operator, result.region, result.dimension,
                          result.resolution,
                          result.spacing, result.error, result.order), ','))
        end
    end
    return filename
end

end # module CorrectionConvergence

if abspath(PROGRAM_FILE) == @__FILE__
    results = CorrectionConvergence.run_convergence()
    CorrectionConvergence.print_report(results)
    output_file = joinpath("out", "corrections_periodic", "correction_convergence.csv")
    CorrectionConvergence.write_csv(output_file, results)
    println("\nWrote $output_file")
end
