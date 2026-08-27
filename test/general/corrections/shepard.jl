@testset "Shepard correction" begin
    setup = update_correction!(correction_setup(ShepardKernelCorrection();
                                                density_calculator=SummationDensity()))
    v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
    density = TrixiParticles.current_density(v, setup.system)

    @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
    @test setup.system.pressure ≈ setup.system.state_equation.(density)

    setup_edac = update_correction!(correction_setup(ShepardKernelCorrection();
                                                     density_calculator=SummationDensity(),
                                                     edac=true))
    @test all(isfinite, setup_edac.system.cache.kernel_correction_coefficient)
    @test all(isfinite, setup_edac.system.cache.density)

    coefficients = ones(TrixiParticles.nparticles(setup.system))
    coefficients[1] = 0.0
    coefficients[2] = NaN
    TrixiParticles.sanitize_kernel_correction_coefficient!(coefficients, setup.system,
                                                           setup.semi)
    @test coefficients[1:2] == ones(2)
end

@testset "Shepard partition of unity" begin
    setup = correction_setup(nothing)
    (; system, semi, v_ode, u_ode) = setup
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coefficient = zeros(TrixiParticles.nparticles(system))
    numerator = zero(coefficient)

    TrixiParticles.compute_shepard_coeff!(system,
                                          TrixiParticles.current_coordinates(u, system),
                                          v_ode, u_ode, semi, coefficient)
    coordinates = TrixiParticles.current_coordinates(u, system)
    TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                          semi) do particle, neighbor, pos_diff, distance
        numerator[particle] += TrixiParticles.hydrodynamic_mass(system, neighbor) *
                               TrixiParticles.smoothing_kernel(system, distance, particle)
    end

    @test numerator ./ coefficient ≈ fill(1000.0, length(numerator)) atol = 2e-12
    @test TrixiParticles.current_density(v, system) == fill(1000.0, length(numerator))
end

@testset "Continuity density reinitialization" begin
    setup = correction_setup()
    v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
    u = TrixiParticles.wrap_u(setup.u_ode, setup.system, setup.semi)
    TrixiParticles.reinit_density!(setup.system, v, u, setup.v_ode, setup.u_ode,
                                   setup.semi)

    @test TrixiParticles.current_density(v, setup.system) ≈ fill(1000.0, 81) atol = 2e-12
    @test maximum(abs, setup.system.pressure) < 2e-10
end
