using .FSIPairFixtures: structure_fluid_pair_state

@testset verbose=true "TVF and Monaghan-Kajtar repulsion" begin
    for (scheme, kind) in ((:wcsph, :tlsph), (:wcsph, :wall), (:edac, :tlsph))
        tvf = TransportVelocityAdami(background_pressure=1000.0)
        state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                           monaghan_kajtar=true,
                                           fluid_options=(; shifting_technique=tvf))
        (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure, u_structure) = state
        fluid.cache.delta_v .= 0
        dv_fluid = zero(v_fluid)
        TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                 fluid, structure, semi)
        repulsion = -10.0 / 0.5 * (1.77 / 32) * (1 + 2.5 * 1.5 + 2 * 1.5^2) * 0.5^5
        # Independent MK force at q=1.5: TVF previously doubled this at zero shift.
        @test dv_fluid[1:2, 1] ≈ [repulsion, 0.0]

        TrixiParticles.update_shifting!(fluid, tvf, v_fluid, u_fluid,
                                        ode.u0.x[1], ode.u0.x[2], semi)
        grad_f = [15 / (56pi), 0.0]
        volume_term = ((1100.0 / 1005.0)^2 + 1.0) / 1100.0
        expected_shift = -1000.0 / (8 * 10.0) *
                         (scheme == :wcsph ? 2 / 1005.0 : volume_term) * grad_f
        @test TrixiParticles.delta_v(fluid, 1) ≈ expected_shift
        if kind == :tlsph
            dv_structure = zero(v_structure)
            TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid,
                                     u_fluid,
                                     structure, fluid, semi)
            @test structure.mass[1] * dv_structure[1:2, 1] ≈ -1100.0 * [repulsion, 0.0]
        end
    end

    @testset "Corrected transport operator does not use an elastic kernel" begin
        state = structure_fluid_pair_state(; monaghan_kajtar=true,
                                           structure_smoothing_length=0.4,
                                           fluid_options=(; correction=KernelCorrection(),
                                                          pressure_acceleration=nothing,
                                                          shifting_technique=TransportVelocityAdami(background_pressure=1000.0)))
        (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
        grad_f = ([15 / (56pi), 0.0] - 5 / (112pi) * [0.1, -0.2]) / 1.3
        tensor_term = -v_fluid[1:2, 1] * dot([0.4, -0.3], grad_f)
        repulsion = -10.0 / 0.5 * (1.77 / 32) * (1 + 2.5 * 1.5 + 2 * 1.5^2) * 0.5^5
        dv_fluid = zero(v_fluid)
        TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                 fluid, structure, semi)
        @test dv_fluid[1:2, 1] ≈ [repulsion, 0.0] + tensor_term
    end
end
