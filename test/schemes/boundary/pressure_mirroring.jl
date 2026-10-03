using .FSIPairFixtures: structure_fluid_pair_state

@testset verbose=true "Pressure mirroring across fluid schemes" begin
    # Prescribed pressure also exercises the IISPH pair RHS without a pressure solve.
    for scheme in (:wcsph, :edac, :iisph), kind in (:wall, :tlsph, :rigid)
        state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                           boundary_density=PressureMirroring())
        (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
        # The boundary cache intentionally holds 230 instead of the fluid's 500.
        # Mirroring must be pair-local: reading or overwriting the cache would be wrong
        # when neighboring fluid particles have different pressures.
        @test TrixiParticles.neighbor_pressure(v_structure, structure, 1, 500.0) == 500.0
        @test structure.boundary_model.pressure[1] == 230.0
        dv_fluid = zero(v_fluid)
        TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                 fluid, structure, semi)
        # At r/h = 1.5, W'(r) = -15/(56pi); x_f - x_s points left, so grad_f points right.
        grad_f = [15 / (56pi), 0.0]
        # Evaluate each scheme's pressure law independently with p_s = p_f = 500.
        # WCSPH uses m_s(p_f+p_s)/(rho_f rho_s); EDAC uses (V_f^2+V_s^2)p/m_f;
        # IISPH uses m_s(p_f/rho_f^2+p_s/rho_s^2), with V = m/rho.
        coefficient = if scheme == :wcsph
            700.0 * 1000.0 / (1005.0 * 950.0)
        elseif scheme == :edac
            ((1100.0 / 1005.0)^2 + (700.0 / 950.0)^2) / 1100.0 * 500.0
        else
            700.0 * (500.0 / 1005.0^2 + 500.0 / 950.0^2)
        end
        @test dv_fluid[1:2, 1] ≈ -coefficient * grad_f
        if kind != :wall
            # Structural acceleration uses material mass; both representations must
            # recover the same physical reaction -m_f a_f. Fixed walls have no RHS load.
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
            # Self and mirrored wall neighbors both contribute 500. Reading the stale
            # wall cache (230) would change the mean and leave a spurious pressure force.
            p_avg = average_pressure_reduction ? 500.0 : 0.0
            @test TrixiParticles.average_pressure(fluid, 1) == p_avg
            dv_fluid = zero(v_fluid)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            grad_f = [15 / (56pi), 0.0]
            volume_term = ((1100.0 / 1005.0)^2 + (700.0 / 950.0)^2) / 1100.0
            @test dv_fluid[1:2, 1] ≈ -volume_term * (500.0 - p_avg) * grad_f
            # Mirroring makes p_f-p_s zero, eliminating EDAC diffusion. Only its
            # artificial EOS remains: dp_f/dt = m_s rho_f/rho_s c^2 (v_f-v_s) dot grad_f.
            # The fixed wall has v_s = 0; row 3 stores pressure after the two velocity rows.
            @test dv_fluid[3, 1] ≈ 700.0 * 1005.0 / 950.0 * 10.0^2 * dot([1.0, 0.5], grad_f)
        end
    end
end
