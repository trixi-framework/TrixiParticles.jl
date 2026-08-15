module SurfacePressureDifferenceValidation

using TrixiParticles
using LinearAlgebra: norm
using Printf: @printf

function pressure_field(position)
    x, y = position
    return x * (1 - x) * y * (1 - y) * (1 + 0.2x + 0.1y)
end

function pressure_gradient(position)
    x, y = position
    factor = 1 + 0.2x + 0.1y
    return SVector((1 - 2x) * y * (1 - y) * factor +
                   0.2x * (1 - x) * y * (1 - y),
                   x * (1 - x) * (1 - 2y) * factor +
                   0.1x * (1 - x) * y * (1 - y))
end

function errors(n, correction)
    particle_spacing = 1.0 / n
    smoothing_length = 2.0 * particle_spacing
    smoothing_kernel = WendlandC6Kernel{2}()
    fluid = RectangularShape(particle_spacing, (n, n), (0.0, 0.0); density=1000.0)
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)
    system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation,
                                         gradient_correction=correction,
                                         surface_method=ColorfieldSurfaceDetection(ideal_density_threshold=0.9),
                                         surface_pressure=SurfacePressureDifference(),
                                         reference_particle_spacing=particle_spacing)
    semi = Semidiscretization(system; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v_ode = Array(ode.u0.x[1])
    u_ode = Array(ode.u0.x[2])
    semi = ode.p.semi
    system = first(semi.systems)
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    n_particles = TrixiParticles.nparticles(system)
    pressure = [pressure_field(SVector{2}(view(coordinates, :, particle)))
                for particle in TrixiParticles.eachparticle(system)]
    acceleration = zeros(2, n_particles)
    constant_acceleration = zeros(2, n_particles)

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
            gradient_correction = TrixiParticles.correction_gradient(system.correction)
            acceleration[:,
                         particle] .+= TrixiParticles.pressure_acceleration(system,
                                                                            system,
                                                                            particle,
                                                                            neighbor,
                                                                            m_a, m_b,
                                                                            pressure[particle],
                                                                            pressure[neighbor],
                                                                            rho_a, rho_b,
                                                                            pos_diff,
                                                                            distance, W_a,
                                                                            gradient_correction)
            constant_acceleration[:,
                                  particle] .+= TrixiParticles.pressure_acceleration(system,
                                                                                     system,
                                                                                     particle,
                                                                                     neighbor,
                                                                                     m_a,
                                                                                     m_b,
                                                                                     2.0,
                                                                                     2.0,
                                                                                     rho_a,
                                                                                     rho_b,
                                                                                     pos_diff,
                                                                                     distance,
                                                                                     W_a,
                                                                                     gradient_correction)
        end
    end

    exact = zeros(2, n_particles)
    for particle in TrixiParticles.eachparticle(system)
        exact[:,
              particle] = -pressure_gradient(SVector{2}(view(coordinates, :, particle))) /
                          TrixiParticles.current_density(v, system, particle)
    end

    support = TrixiParticles.compact_support(system, system)
    min_x = minimum(view(coordinates, 1, :))
    boundary = [particle
                for particle in axes(coordinates, 2)
                if isapprox(coordinates[1, particle], min_x; atol=eps()) &&
                   support < coordinates[2, particle] < 1.0 - support]
    interior = [particle
                for particle in axes(coordinates, 2)
                if 2support < coordinates[1, particle] < 1.0 - 2support &&
                   2support < coordinates[2, particle] < 1.0 - 2support]

    relative_error(particles) = norm(acceleration[:, particles] - exact[:, particles]) /
                                norm(exact[:, particles])
    constant_error(particles) = norm(constant_acceleration[:, particles]) /
                                sqrt(length(particles))
    mass = TrixiParticles.hydrodynamic_mass(system, 1)
    momentum_residual = norm(mass * vec(sum(acceleration; dims=2)))

    return (; spacing=particle_spacing, boundary=relative_error(boundary),
            interior=relative_error(interior),
            constant_boundary=constant_error(boundary), momentum_residual)
end

function run(; resolutions=(24, 48, 96))
    results = NamedTuple[]
    for (method, correction) in ((:gradient, GradientCorrection()),
         (:mixed, MixedKernelGradientCorrection()))
        previous = nothing
        println("\n$method")
        println("| N | Boundary error | Boundary order | Interior error | Interior order | " *
                "Constant boundary | Momentum residual |")
        println("|--:|---------------:|---------------:|---------------:|---------------:|" *
                "------------------:|------------------:|")
        for n in resolutions
            result = errors(n, correction)
            boundary_order = isnothing(previous) ? NaN :
                             log(previous.boundary / result.boundary) /
                             log(previous.spacing / result.spacing)
            interior_order = isnothing(previous) ? NaN :
                             log(previous.interior / result.interior) /
                             log(previous.spacing / result.spacing)
            @printf("| %d | %.6e | %s | %.6e | %s | %.6e | %.6e |\n", n,
                    result.boundary,
                    isnan(boundary_order) ? "-" : string(boundary_order),
                    result.interior,
                    isnan(interior_order) ? "-" : string(interior_order),
                    result.constant_boundary, result.momentum_residual)
            push!(results, (; method, n, result..., boundary_order, interior_order))
            previous = result
        end
    end
    return results
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    SurfacePressureDifferenceValidation.run()
end
