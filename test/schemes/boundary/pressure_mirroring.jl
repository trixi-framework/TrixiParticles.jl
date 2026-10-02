using .FSIPairFixtures: structure_fluid_pair_state

@testset verbose=true "Pressure mirroring across fluid schemes" begin
    for scheme in (:wcsph, :edac, :iisph), kind in (:wall, :tlsph, :rigid)
        state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                           boundary_density=PressureMirroring())
        (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
        @test TrixiParticles.neighbor_pressure(v_structure, structure, 1, 500.0) == 500.0
        @test structure.boundary_model.pressure[1] == 230.0
        dv_fluid = zero(v_fluid)
        TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                 fluid, structure, semi)
        grad_f = [15 / (56pi), 0.0]
        coefficient = if scheme == :wcsph
            700.0 * 1000.0 / (1005.0 * 950.0)
        elseif scheme == :edac
            ((1100.0 / 1005.0)^2 + (700.0 / 950.0)^2) / 1100.0 * 500.0
        else
            700.0 * (500.0 / 1005.0^2 + 500.0 / 950.0^2)
        end
        @test dv_fluid[1:2, 1] ≈ -coefficient * grad_f
        if kind != :wall
            dv_structure = zero(v_structure)
            TrixiParticles.reset_interaction_caches!(structure)
            TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid,
                                     u_fluid,
                                     structure, fluid, semi)
            force = kind == :rigid ? structure.force_per_particle[:, 1] :
                    structure.mass[1] * dv_structure[1:2, 1]
            @test force ≈ -1100.0 * dv_fluid[1:2, 1]
        end
    end

    @testset "EDAC wall pressure evolution and average" begin
        for average_pressure_reduction in (false, true)
            state = structure_fluid_pair_state(; fluid_scheme=:edac, structure_kind=:wall,
                                               boundary_density=PressureMirroring(),
                                               fluid_options=(; average_pressure_reduction))
            (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure,
             u_structure) = state
            TrixiParticles.update_average_pressure!(fluid, fluid.average_pressure_reduction,
                                                    ode.u0.x[1], ode.u0.x[2], semi)
            p_avg = average_pressure_reduction ? 500.0 : 0.0
            @test TrixiParticles.average_pressure(fluid, 1) == p_avg
            dv_fluid = zero(v_fluid)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            grad_f = [15 / (56pi), 0.0]
            volume_term = ((1100.0 / 1005.0)^2 + (700.0 / 950.0)^2) / 1100.0
            @test dv_fluid[1:2, 1] ≈ -volume_term * (500.0 - p_avg) * grad_f
            @test dv_fluid[3, 1] ≈ 700.0 * 1005.0 / 950.0 * 10.0^2 * dot([1.0, 0.5], grad_f)
        end
    end
end
