@testset "Shepard correction" begin
    # Recompute the correction after a full update pass.
    setup = update_correction!(correction_setup(ShepardKernelCorrection();
                                                density_calculator=SummationDensity()))
    v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
    density = TrixiParticles.current_density(v, setup.system)

    # The corrected coefficients must be finite, and the pressure must be computed
    # from the Shepard-corrected density.
    @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
    @test setup.system.pressure ≈ setup.system.state_equation.(density)

    setup_edac = update_correction!(correction_setup(ShepardKernelCorrection();
                                                     density_calculator=SummationDensity(),
                                                     edac=true))
    @test all(isfinite, setup_edac.system.cache.kernel_correction_coefficient)
    @test all(isfinite, setup_edac.system.cache.density)

    # The sanitizer must replace zero and NaN coefficients by one, leaving valid
    # coefficients untouched.
    coefficients = ones(TrixiParticles.nparticles(setup.system))
    coefficients[1] = 0.0
    coefficients[2] = NaN
    TrixiParticles.sanitize_kernel_correction_coefficient!(coefficients, setup.system,
                                                           setup.semi)
    @test coefficients[1:2] == ones(2)
end

@testset "Shepard partition of unity" begin
    # Verify that the Shepard coefficient reproduces a constant density field exactly:
    # the uncorrected kernel sum (numerator) divided by the Shepard coefficient
    # (denominator) must be the reference density everywhere.
    setup = correction_setup(nothing)
    (; system, semi, v_ode, u_ode) = setup
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coefficient = zeros(TrixiParticles.nparticles(system))
    numerator = zero(coefficient)

    TrixiParticles.compute_shepard_coeff!(system,
                                          TrixiParticles.current_coordinates(u, system),
                                          v_ode, u_ode, semi, coefficient)
    # Recompute the uncorrected kernel sum by hand for comparison.
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
    # Reinitializing a continuity density with the Shepard operator must reproduce
    # a constant density field exactly.
    setup = correction_setup()
    v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
    u = TrixiParticles.wrap_u(setup.u_ode, setup.system, setup.semi)
    TrixiParticles.reinit_density!(setup.system, v, u, setup.v_ode, setup.u_ode,
                                   setup.semi)

    @test TrixiParticles.current_density(v, setup.system) ≈ fill(1000.0, 81) atol = 2e-12
    # The pressure must be recomputed from the reinitialized density.
    @test maximum(abs, setup.system.pressure) < 2e-10
end
