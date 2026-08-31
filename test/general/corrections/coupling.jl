# A sentinel correction verifies that force corrections are routed independently.
struct CustomForceCorrection end

function TrixiParticles.free_surface_correction(::CustomForceCorrection,
                                                particle_system, rho_a, rho_b)
    return 2, 3, 4
end

@testset "Correction role routing" begin
    setup = correction_setup()
    correction = CustomForceCorrection()
    selected = TrixiParticles.correction_force(correction)
    @test selected === correction
    @test TrixiParticles.free_surface_correction(selected, setup.system,
                                                 1000.0, 1000.0) == (2, 3, 4)
end

@testset "Supported pressure variation matrix" begin
    # Store a nonuniform pressure in the state location used by each formulation.
    function set_pressure_field!(setup, edac)
        pressure = range(1.0, 2.0; length=TrixiParticles.nparticles(setup.system))
        if edac
            v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
            v[3, :] .= pressure
        elseif setup.system.density_calculator isa ContinuityDensity
            v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
            v[end, :] .= 1000.0 .+ pressure
        else
            setup.system.pressure .= pressure
        end
        return setup
    end

    # Enumerate only correction and pressure-formulation combinations supported by each density model.
    summation_corrections = (nothing, ShepardKernelCorrection(), KernelCorrection(),
                             GradientCorrection(), BlendedGradientCorrection(0.5),
                             MixedKernelGradientCorrection())
    continuity_corrections = (nothing, KernelCorrection(), GradientCorrection(),
                              BlendedGradientCorrection(0.5),
                              MixedKernelGradientCorrection())
    summation_pressure = (nothing,
                          TrixiParticles.pressure_acceleration_summation_density,
                          TrixiParticles.inter_particle_averaged_pressure)
    continuity_pressure = (nothing,
                           TrixiParticles.pressure_acceleration_continuity_density,
                           TrixiParticles.inter_particle_averaged_pressure)

    # All supported summation-density combinations produce a finite, nonzero pressure force.
    for edac in (false, true), correction in summation_corrections,
        pressure_acceleration in summation_pressure
        setup = correction_setup(correction; n=4, edac,
                                 density_calculator=SummationDensity(),
                                 pressure_acceleration)
        set_pressure_field!(setup, edac)
        dv_ode = zero(setup.v_ode)
        TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                             (; semi=setup.semi, split_integration_data=nothing), 0.0)
        @test all(isfinite, dv_ode)
        @test any(!iszero, view(dv_ode, 1:2, :))
    end

    # Exercise the corresponding continuity-density combinations.
    for edac in (false, true), correction in continuity_corrections,
        pressure_acceleration in continuity_pressure
        setup = correction_setup(correction; n=4, edac,
                                 density_calculator=ContinuityDensity(),
                                 pressure_acceleration)
        set_pressure_field!(setup, edac)
        dv_ode = zero(setup.v_ode)
        TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                             (; semi=setup.semi, split_integration_data=nothing), 0.0)
        @test all(isfinite, dv_ode)
        @test any(!iszero, view(dv_ode, 1:2, :))
    end

    # Tensile instability control permits the uncorrected continuity formulation only.
    for edac in (false, true)
        setup = correction_setup(; n=4, edac,
                                 density_calculator=ContinuityDensity(),
                                 pressure_acceleration=tensile_instability_control)
        set_pressure_field!(setup, edac)
        dv_ode = zero(setup.v_ode)
        TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                             (; semi=setup.semi, split_integration_data=nothing), 0.0)
        @test all(isfinite, dv_ode)
        @test any(!iszero, view(dv_ode, 1:2, :))

        for correction in continuity_corrections[2:end]
            @test_throws ArgumentError correction_setup(correction; n=4, edac,
                                                        density_calculator=ContinuityDensity(),
                                                        pressure_acceleration=tensile_instability_control)
        end
    end
end

@testset "Corrected structure coupling" begin
    # Check every correction cache associated with either a fluid or structure boundary model.
    function cache_is_finite(system)
        cache = system isa Union{RigidBodySystem, TotalLagrangianSPHSystem} ?
                system.boundary_model.cache : system.cache
        return all((:kernel_correction_coefficient, :dw_gamma,
                    :correction_matrix)) do name
            !hasproperty(cache, name) || all(isfinite, getproperty(cache, name))
        end
    end

    # Build a perturbed fluid-structure pair and return force-balance diagnostics.
    function coupled_result(kind, structure_kind, correction;
                            boundary_correction=correction, reverse_order=false,
                            average_pressure_reduction=false)
        spacing = 0.1
        density = 1000.0
        kernel = WendlandC6Kernel{2}()
        smoothing_length = 2spacing
        state_equation = kind == :wcsph ?
                         StateEquationCole(; sound_speed=10.0,
                                           reference_density=density,
                                           exponent=1) : nothing
        # Break pair symmetry so corrected gradients and the reaction-force path are exercised.
        fluid_initial = RectangularShape(spacing, (4, 3), (0.0, 0.0); density)
        fluid_initial.coordinates[1, 2] += 0.013
        fluid_initial.coordinates[2, 5] -= 0.009
        if kind == :wcsph
            fluid = WeaklyCompressibleSPHSystem(fluid_initial; smoothing_kernel=kernel,
                                                smoothing_length,
                                                density_calculator=SummationDensity(),
                                                state_equation, correction)
        else
            fluid = EntropicallyDampedSPHSystem(fluid_initial;
                                                smoothing_kernel=kernel,
                                                smoothing_length,
                                                sound_speed=10.0,
                                                density_calculator=SummationDensity(),
                                                correction,
                                                average_pressure_reduction)
        end

        structure_initial = RectangularShape(spacing, (4, 2), (0.0, -0.2);
                                             density=1200.0)
        hydrodynamic_density = fill(density, TrixiParticles.nparticles(structure_initial))
        hydrodynamic_mass = hydrodynamic_density .* spacing^2
        boundary_model = BoundaryModelDummyParticles(hydrodynamic_density,
                                                     hydrodynamic_mass,
                                                     AdamiPressureExtrapolation(),
                                                     kernel, smoothing_length;
                                                     state_equation,
                                                     correction=boundary_correction)
        if structure_kind == :rigid
            structure = RigidBodySystem(structure_initial; boundary_model,
                                        particle_spacing=spacing)
        else
            structure = TotalLagrangianSPHSystem(structure_initial;
                                                 smoothing_kernel=kernel,
                                                 smoothing_length,
                                                 young_modulus=0.0,
                                                 poisson_ratio=0.0,
                                                 boundary_model)
        end

        # Reversing this order must not affect corrections or the coupled RHS.
        systems = reverse_order ? (structure, fluid) : (fluid, structure)
        semi = Semidiscretization(systems...; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        fluid = only(system
                     for system in ode.p.semi.systems
                     if system isa Union{WeaklyCompressibleSPHSystem,
                              EntropicallyDampedSPHSystem})
        structure = only(system
                         for system in ode.p.semi.systems
                         if system isa Union{RigidBodySystem,
                                             TotalLagrangianSPHSystem})
        v_fluid = TrixiParticles.wrap_v(v_ode, fluid, ode.p.semi)
        if kind == :wcsph
            v_fluid .= 0.0
        else
            v_fluid[3, :] .= range(1.0, 2.0; length=size(v_fluid, 2))
        end

        # Compare total fluid and structure forces rather than their incompatible accelerations.
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode,
                             (; semi=ode.p.semi, split_integration_data=nothing), 0.0)
        dv_fluid = TrixiParticles.wrap_v(dv_ode, fluid, ode.p.semi)
        fluid_force = vec(sum(fluid.mass' .* view(dv_fluid, 1:2, :); dims=2))
        if structure_kind == :rigid
            structure_force = vec(sum(structure.force_per_particle; dims=2))
        else
            dv_structure = TrixiParticles.wrap_v(dv_ode, structure, ode.p.semi)
            structure_force = vec(sum(structure.mass' .* view(dv_structure, 1:2, :);
                                      dims=2))
        end
        force_scale = norm(fluid_force) + norm(structure_force)
        relative_residual = norm(fluid_force + structure_force) /
                            max(force_scale, eps())

        fluid_correction = hasproperty(fluid.cache, :correction_matrix) ?
                           copy(fluid.cache.correction_matrix) : nothing
        structure_cache = structure.boundary_model.cache
        structure_correction = hasproperty(structure_cache, :correction_matrix) ?
                               copy(structure_cache.correction_matrix) : nothing

        return (; relative_residual, force_scale, fluid_force, structure_force,
                fluid_rhs=copy(dv_fluid), fluid_correction, structure_correction,
                finite=all(isfinite, dv_ode) && cache_is_finite(fluid) &&
                       cache_is_finite(structure))
    end

    # Every supported correction conserves momentum for rigid and deformable structures.
    corrections = (KernelCorrection(), GradientCorrection(),
                   BlendedGradientCorrection(0.4), MixedKernelGradientCorrection())
    for kind in (:wcsph, :edac), structure_kind in (:rigid, :tlsph),
        correction in corrections
        result = coupled_result(kind, structure_kind, correction)
        @test result.finite
        @test result.force_scale > eps()
        @test result.relative_residual < 2e-13
    end

    # Global system ordering must not alter EDAC fluid-structure results.
    for structure_kind in (:rigid, :tlsph)
        forward = coupled_result(:edac, structure_kind, GradientCorrection())
        reverse = coupled_result(:edac, structure_kind, GradientCorrection();
                                 reverse_order=true)
        @test forward.fluid_correction≈reverse.fluid_correction rtol=5e-13 atol=5e-13
        @test forward.structure_correction≈reverse.structure_correction rtol=5e-13 atol=5e-13
        @test forward.fluid_rhs≈reverse.fluid_rhs rtol=1e-11 atol=1e-10
        @test forward.fluid_force≈reverse.fluid_force rtol=1e-11 atol=1e-10
        @test forward.structure_force≈reverse.structure_force rtol=1e-11 atol=1e-10
    end

    # Pair-pressure offsets must preserve the same reaction-force balance.
    result = coupled_result(:edac, :rigid, GradientCorrection();
                            average_pressure_reduction=true)
    @test result.force_scale > eps()
    @test result.relative_residual < 2e-13

    # Kernel corrections remain conservative when only the fluid owns the correction cache.
    for structure_kind in (:rigid, :tlsph),
        correction in (KernelCorrection(), MixedKernelGradientCorrection())
        result = coupled_result(:wcsph, structure_kind, correction;
                                boundary_correction=nothing)
        @test result.finite
        @test result.force_scale > eps()
        @test result.relative_residual < 2e-13
    end
end
