using .FSIPairFixtures: structure_fluid_pair_state

@testset verbose=true "Hydrodynamic boundary kernels" begin
    @testset "TLSPH elastic kernel is separate" begin
        for correction in (KernelCorrection(), MixedKernelGradientCorrection())
            state = structure_fluid_pair_state(;
                                               fluid_options=(; correction,
                                                              pressure_acceleration=nothing),
                                               structure_smoothing_length=0.4,
                                               structure_smoothing_kernel=WendlandC2Kernel{2}())
            (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
            dv_fluid, dv_structure = zero(v_fluid), zero(v_structure)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid,
                                     u_fluid,
                                     structure, fluid, semi)
            grad_s = [-15 / (56pi), 0.0]
            grad_f = (-grad_s - 5 / (112pi) * [0.1, -0.2]) / 1.3
            correction isa MixedKernelGradientCorrection &&
                (grad_f = [1.0 0.3; 0.0 1.0] * grad_f)
            expected = 1100.0 * 700.0 / (1005.0 * 950.0) * (500.0 * grad_f - 230.0 * grad_s)
            @test structure.mass[1] * dv_structure[1:2, 1] ≈ expected
            @test -1100.0 * dv_fluid[1:2, 1] ≈ expected
            @test iszero(TrixiParticles.smoothing_kernel_grad(structure, SVector(1.5, 0.0),
                                                              1.5, 1))
            @test TrixiParticles.hydrodynamic_kernel_grad(structure, SVector(1.5, 0.0), 1.5,
                                                          1) ≈ grad_s
        end
    end

    @testset "Boundary correction caches are prepared" begin
        for kind in (:tlsph, :rigid),
            correction in
            (KernelCorrection(), GradientCorrection(), MixedKernelGradientCorrection())

            state = structure_fluid_pair_state(; structure_kind=kind,
                                               structure_smoothing_length=0.4,
                                               boundary_density=PressureMirroring(),
                                               boundary_correction=correction,
                                               fluid_options=(;
                                                              correction=KernelCorrection(),
                                                              pressure_acceleration=nothing))
            (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure,
             u_structure) = state
            model = structure.boundary_model
            for field in (:dw_gamma, :kernel_correction_coefficient, :correction_matrix)
                haskey(model.cache, field) && fill!(getproperty(model.cache, field), NaN)
            end
            TrixiParticles.update_boundary_interpolation!(structure, v_structure,
                                                          u_structure,
                                                          ode.u0.x[1], ode.u0.x[2], semi,
                                                          0.0)
            for field in (:dw_gamma, :kernel_correction_coefficient, :correction_matrix)
                haskey(model.cache, field) &&
                    @test all(isfinite, getproperty(model.cache, field))
            end
            grad_s = [-15 / (56pi), 0.0]
            if correction isa KernelCorrection ||
               correction isa MixedKernelGradientCorrection
                gamma = 1100.0 / 1005.0 * 5 / (112pi)
                # TLSPH includes the particle itself. Contact-only rigid self-search is empty.
                kind == :tlsph && (gamma += 700.0 / 950.0 * 10 / (7pi))
                dw_gamma = 1100.0 / 1005.0 * grad_s / gamma
                @test model.cache.kernel_correction_coefficient[1] ≈ gamma
                @test model.cache.dw_gamma[:, 1] ≈ dw_gamma
                grad_s = (grad_s - 5 / (112pi) * dw_gamma) / gamma
            end
            grad_f = ([15 / (56pi), 0.0] - 5 / (112pi) * [0.1, -0.2]) / 1.3
            expected = 1100.0 * 700.0 * 500.0 / (1005.0 * 950.0) * (grad_f - grad_s)
            dv_structure = zero(v_structure)
            TrixiParticles.reset_interaction_caches!(structure)
            TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid,
                                     u_fluid,
                                     structure, fluid, semi)
            force = kind == :rigid ? structure.force_per_particle[:, 1] :
                    structure.mass[1] * dv_structure[1:2, 1]
            @test force ≈ expected
        end
    end

    @testset "Rigid boundary density uses its own gradient" begin
        state = structure_fluid_pair_state(; structure_kind=:rigid,
                                           boundary_density=ContinuityDensity(),
                                           fluid_options=(; correction=KernelCorrection(),
                                                          pressure_acceleration=nothing))
        (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
        dv_structure = zero(v_structure)
        TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid, u_fluid,
                                 structure, fluid, semi)
        @test dv_structure[end, 1] ≈
              950.0 / 1005.0 * 1100.0 * dot([-0.75, -0.9], [-15 / (56pi), 0.0])
    end
end
