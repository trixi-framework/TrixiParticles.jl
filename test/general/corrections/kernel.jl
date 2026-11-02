@testset "Kernel correction" begin
    # Kernel correction initializes both cache components for WCSPH and EDAC.
    for edac in (false, true)
        setup = correction_setup(KernelCorrection(); edac,
                                 pressure_acceleration=nothing)
        update_correction!(setup)

        @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
        @test all(isfinite, setup.system.cache.dw_gamma)
    end

    # The corrected gradient of a constant field vanishes on regular and perturbed grids.
    for perturbation in (false, true)
        setup = update_correction!(correction_setup(KernelCorrection(); perturbation))
        moments = correction_moments(setup)
        @test maximum(abs, moments.zeroth_gradient_moment) < 2e-12
    end

    # Boundary cache arrays retain the boundary scalar type.
    density32 = fill(1000.0f0, 4)
    mass32 = fill(10.0f0, 4)
    state_equation = StateEquationCole(; sound_speed=10.0f0,
                                       reference_density=1000.0f0, exponent=1)
    boundary = BoundaryModelDummyParticles(density32, mass32, SummationDensity(),
                                           WendlandC6Kernel{2}(), 0.2f0;
                                           state_equation,
                                           gradient_correction=KernelCorrection())
    @test eltype(boundary.cache.dw_gamma) == Float32

    # Restarting preserves the correction state and its resulting RHS.
    for edac in (false, true),
        density_calculator in (SummationDensity(),
                               ContinuityDensity())
        result = correction_restart_result(KernelCorrection(); edac, density_calculator)
        @test result.state_equal
        @test result.rhs_equal
        @test result.cache_finite
    end

    @testset "fallback to uncorrected gradient for degenerate coefficients" begin
        for correction in (KernelCorrection(), MixedKernelGradientCorrection()),
            edac in (false, true),
            density_calculator in (SummationDensity(), ContinuityDensity())

            # Use a pressure formulation compatible with asymmetric kernel corrections
            # for EDAC systems.
            setup = correction_setup(correction; edac, density_calculator, n=4,
                                     pressure_acceleration=nothing)
            setup.system.mass .= 0
            update_correction!(setup)
            @test all(==(1), setup.system.cache.kernel_correction_coefficient)
            @test all(iszero, setup.system.cache.dw_gamma)
            @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
            @test all(isfinite, setup.system.cache.dw_gamma)
            # Only WCSPH with ContinuityDensity keeps a finite density with zero mass;
            # SummationDensity yields zero density and EDAC couples pressure in `v`.
            if density_calculator isa ContinuityDensity && !edac
                dv = zero(setup.v_ode)
                TrixiParticles.kick!(dv, setup.v_ode, setup.u_ode,
                                     (; semi=setup.semi, split_integration_data=nothing),
                                     0.0)
                @test all(isfinite, dv)
            end

            # Tiny mass below `sqrt(eps(T))` => fallback (only well-defined for
            # ContinuityDensity, where density is independent of mass).
            if density_calculator isa ContinuityDensity
                setup = correction_setup(correction; edac, density_calculator, n=4,
                                         pressure_acceleration=nothing)
                setup.system.mass .= 1.0e-12
                update_correction!(setup)
                @test all(==(1), setup.system.cache.kernel_correction_coefficient)
                @test all(iszero, setup.system.cache.dw_gamma)
            end

            # Non-finite or negative coefficients => fallback caches remain finite
            # For SummationDensity, a negative mass yields a positive volume
            # (density is also negative), so it does not reliably produce a
            # degenerate coefficient.
            bad_masses = density_calculator isa ContinuityDensity ?
                         (NaN, Inf, -1.0) : (NaN, Inf)
            for bad_mass in bad_masses
                setup = correction_setup(correction; edac, density_calculator, n=4,
                                         pressure_acceleration=nothing)
                setup.system.mass .= bad_mass
                update_correction!(setup)
                @test all(==(1), setup.system.cache.kernel_correction_coefficient)
                @test all(iszero, setup.system.cache.dw_gamma)
                @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
                @test all(isfinite, setup.system.cache.dw_gamma)
            end
            # ContinuityDensity with non-finite density also yields fallback
            if density_calculator isa ContinuityDensity
                setup = correction_setup(correction; edac, density_calculator, n=4,
                                         pressure_acceleration=nothing)
                v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
                v[end, :] .= NaN
                update_correction!(setup)
                @test all(==(1), setup.system.cache.kernel_correction_coefficient)
                @test all(iszero, setup.system.cache.dw_gamma)
                @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
            end
        end

        # The fallback disables the correction: dw_gamma=0 and coefficient=1
        # imply the corrected kernel gradient reduces to the uncorrected one.
        for correction in (KernelCorrection(), MixedKernelGradientCorrection())
            setup = correction_setup(correction; n=4, pressure_acceleration=nothing)
            setup.system.mass .= 0
            # ContinuityDensity ensures finite density for a well-defined RHS
            # check, but the fallback for zero mass is independent of density.
            update_correction!(setup)
            system = setup.system
            pos_diff = SVector(0.1, 0.2)
            distance = sqrt(sum(abs2, pos_diff))
            h = TrixiParticles.initial_smoothing_length(system)
            kernel = TrixiParticles.system_smoothing_kernel(system)
            for particle in TrixiParticles.eachparticle(system)
                # Only test particles that actually fell back (all in this degenerate setup)
                @test setup.system.cache.kernel_correction_coefficient[particle] == 1
                corr_grad = TrixiParticles.corrected_kernel_grad_unsafe(kernel, pos_diff,
                                                                        distance, h,
                                                                        correction, system,
                                                                        particle)
                uncorr = TrixiParticles.kernel_grad(kernel, pos_diff, distance, h)
                @test corr_grad ≈ uncorr
            end
        end
    end
end
