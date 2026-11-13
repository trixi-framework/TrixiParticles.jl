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

    # Only nonfinite and nonpositive coefficients are invalid. In particular,
    # Float32 values below `sqrt(eps(Float32))` remain valid Shepard normalizers.
    coefficients = Float32[0, -1, NaN, Inf, 1.0f-4]
    TrixiParticles.sanitize_kernel_correction_coefficient!(coefficients, setup.system,
                                                           setup.semi)
    @test coefficients == Float32[1, 1, 1, 1, 1.0f-4]

    # Correction caches include inactive open-boundary buffer entries. Sanitizing
    # every coefficient prevents the full-array density division from producing NaNs.
    buffered = update_correction!(correction_setup(ShepardKernelCorrection();
                                                   density_calculator=SummationDensity(),
                                                   buffer_size=2,
                                                   neighborhood_search=nothing))
    @test all(isfinite, buffered.system.cache.kernel_correction_coefficient)
    @test all(isfinite, buffered.system.cache.density)
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

@testset "Evolved-density impulse reinitialization" begin
    for edac in (false, true)
        setup = correction_setup(nothing; edac, density_calculator=ContinuityDensity())
        (; system, semi, v_ode, u_ode) = setup
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        particle = cld(nparticles(system), 2)
        v[end, :] .= 1200.0
        v[end, particle] = 1600.0
        if edac
            v[3, :] .= range(2.0, 5.0; length=size(v, 2))
        end
        pressure_before = copy(TrixiParticles.current_pressure(v, system))

        # With one density impulse, Shepard's evolved-volume result has a closed
        # harmonic-mean expression. A no-op or initial-density volumes cannot satisfy it.
        numerator = Ref(0.0)
        coordinates = TrixiParticles.current_coordinates(u, system)
        TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                              semi;
                                              points=particle:particle) do a, b, r, distance
            numerator[] += TrixiParticles.hydrodynamic_mass(system, b) *
                           TrixiParticles.smoothing_kernel(system, distance, a)
        end
        self_weight = TrixiParticles.hydrodynamic_mass(system, particle) *
                      TrixiParticles.smoothing_kernel(system, 0.0, particle)
        expected = numerator[] / (numerator[] / 1200 + self_weight * (1/1600 - 1/1200))
        TrixiParticles.reinit_density!(system, v, u, v_ode, u_ode, semi)
        @test v[end, particle] ≈ expected rtol=1e-13
        @test 1200 < v[end, particle] < 1600
        if edac
            @test TrixiParticles.current_pressure(v, system) == pressure_before
        else
            @test system.pressure ≈
                  system.state_equation.(TrixiParticles.current_density(v, system))
        end
    end
end
