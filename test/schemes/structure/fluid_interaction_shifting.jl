using .FSIPairFixtures: structure_fluid_pair_state

@testset verbose=true "Physical loads and shifting transport" begin
    for scheme in (:wcsph, :edac), kind in (:tlsph, :rigid),
        shifting in
        (ConsistentShiftingSun2019(), TransportVelocityAdami(background_pressure=1000.0))

        state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                           fluid_options=(; shifting_technique=shifting))
        (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
        TrixiParticles.current_pressure(v_fluid, fluid) .= 0
        structure.boundary_model.pressure .= 0
        for scale in (1.0, 7.0)
            fluid.cache.delta_v[:, 1] .= scale .* [0.4, -0.3]
            dv_fluid, dv_structure = zero(v_fluid), zero(v_structure)
            TrixiParticles.reset_interaction_caches!(structure)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid,
                                     u_fluid,
                                     structure, fluid, semi)
            factor = if shifting isa ParticleShiftingTechnique
                2 * 700.0 / 950.0
            elseif scheme == :wcsph
                -700.0 / 950.0
            else
                -((1100.0 / 1005.0)^2 + (700.0 / 950.0)^2) / 1100.0 * 1005.0 * 950.0 /
                (1005.0 + 950.0)
            end
            expected = factor * v_fluid[1:2, 1] *
                       dot(scale .* [0.4, -0.3], [15 / (56pi), 0.0])
            @test dv_fluid[1:2, 1] ≈ expected
            force = kind == :rigid ? structure.force_per_particle[:, 1] :
                    structure.mass[1] * dv_structure[1:2, 1]
            @test iszero(force)
        end
    end

    @testset "3D forces and rigid torque" begin
        for scheme in (:wcsph, :edac), kind in (:tlsph, :rigid),
            backend in (SerialBackend(), PolyesterBackend())
            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                               dimensions=3,
                                               parallelization_backend=backend,
                                               coordinates=[1.2 1.4 1.6; -0.4 0.5 0.1;
                                                            0.2 -0.3 0.6],
                                               fluid_options=(;
                                                              viscosity=ViscosityAdami(nu=0.1),
                                                              shifting_technique=ConsistentShiftingSun2019()))
            (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure,
             u_structure) = state
            # Isolate physical fluid acceleration by zeroing the transport velocity.
            delta_v = copy(fluid.cache.delta_v)
            fluid.cache.delta_v .= 0
            dv_fluid = zero(v_fluid)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            fluid.cache.delta_v .= delta_v
            dv_structure = zero(v_structure)
            TrixiParticles.reset_interaction_caches!(structure)
            TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid,
                                     u_fluid,
                                     structure, fluid, semi)
            forces = kind == :rigid ? copy(structure.force_per_particle) :
                     dv_structure[1:3, :] .* reshape(structure.mass, 1, :)
            @test vec(sum(forces; dims=2)) ≈ -1100.0 * dv_fluid[1:3, 1]
            if kind == :rigid
                TrixiParticles.update_final!(structure, v_structure, u_structure,
                                             ode.u0.x[1], ode.u0.x[2], semi, 0.0)
                expected_torque = sum(cross(u_structure[:, i] - structure.center_of_mass[],
                                            forces[:, i])
                                      for i in eachparticle(structure))
                TrixiParticles.apply_resultant_force_and_torque!(dv_structure, structure,
                                                                 semi)
                @test structure.resultant_force[] ≈ vec(sum(forces; dims=2))
                @test structure.resultant_torque[] ≈ expected_torque
            end
        end
    end

    @testset "Additional viscosity models share the physical operator" begin
        for scheme in (:wcsph, :edac), kind in (:tlsph, :rigid),
            boundary_viscosity in (ViscosityAdamiSGS(nu=0.1), ViscosityMorrisSGS(nu=0.1),
             ViscosityCarreauYasuda(nu0=0.1, nu_inf=0.01, lambda=1.0, a=2.0, n=0.5))

            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                               boundary_viscosity,
                                               fluid_options=(;
                                                              viscosity=ViscosityAdami(nu=0.4)))
            (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
            TrixiParticles.current_pressure(v_fluid, fluid) .= 0
            structure.boundary_model.pressure .= 0
            dv_fluid, dv_structure = zero(v_fluid), zero(v_structure)
            TrixiParticles.reset_interaction_caches!(structure)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid,
                                     u_fluid,
                                     structure, fluid, semi)
            force = kind == :rigid ? structure.force_per_particle[:, 1] :
                    structure.mass[1] * dv_structure[1:2, 1]
            @test force ≈ -1100.0 * dv_fluid[1:2, 1]
            @test force[1] > 0
        end
    end
end
