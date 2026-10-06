using .FSIPairFixtures: structure_fluid_pair_state

@testset verbose=true "Hydrodynamic boundary kernels" begin
    @testset "Equal boundary-fluid support is required" begin
        for kind in (:tlsph, :rigid)
            state = structure_fluid_pair_state(; structure_kind=kind)
            structure = state.structure
            structure = TrixiParticles.@set structure.boundary_model.smoothing_length = 0.5
            @test_throws ArgumentError Semidiscretization(state.fluid, structure)
        end
    end

    @testset "TLSPH elastic kernel is separate" begin
        # r = 1.5 lies outside elastic support 2*0.4 but inside hydrodynamic support 2*1.
        # Reusing the elastic gradient would erase the boundary-pressure contribution.
        for correction in (KernelCorrection(), MixedKernelGradientCorrection())
            state = structure_fluid_pair_state(;
                                               fluid_options=(; correction,
                                                              pressure_acceleration=nothing),
                                               boundary_density=PressureMirroring(),
                                               structure_smoothing_length=0.4,
                                               structure_smoothing_kernel=WendlandC2Kernel{2}())
            (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure,
             u_structure) = state
            # Use the normal update path, not prescribed correction caches.
            TrixiParticles.update_systems_and_nhs(ode.u0.x[1], ode.u0.x[2], semi, 0.0)
            @test TrixiParticles.compact_support(structure, fluid) ==
                  TrixiParticles.compact_support(fluid, structure) == 2.0
            dv_fluid, dv_structure = zero(v_fluid), zero(v_structure)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid,
                                     u_fluid,
                                     structure, fluid, semi)
            # For the 2D cubic spline at h=1, r=1.5: W=5/(112pi), W'=-15/(56pi).
            # The fluid correction includes itself and the boundary particle.
            grad_s = [-15 / (56pi), 0.0]
            gamma_f = 1100.0 / 1005.0 * 10 / (7pi) + 700.0 / 950.0 * 5 / (112pi)
            dw_gamma_f = 700.0 / 950.0 * (-grad_s) / gamma_f
            grad_f = (-grad_s - 5 / (112pi) * dw_gamma_f) / gamma_f
            # The collinear neighborhood makes the mixed gradient matrix the identity.
            # Continuity-density pressure law with distinct gradients:
            # F_s = m_f*m_s/(rho_f*rho_s) * (p_f*grad_f - p_s*grad_s).
            expected = 1100.0 * 700.0 * 500.0 / (1005.0 * 950.0) * (grad_f - grad_s)
            @test structure.mass[1] * dv_structure[1:2, 1] ≈ expected
            @test -1100.0 * dv_fluid[1:2, 1] ≈ expected
            # Confirm the fixture actually separates elastic and hydrodynamic operators.
            @test iszero(TrixiParticles.smoothing_kernel_grad(structure, SVector(1.5, 0.0),
                                                              1.5, 1))
            @test TrixiParticles.hydrodynamic_kernel_grad(structure, SVector(1.5, 0.0), 1.5,
                                                          1) ≈ grad_s
        end
    end

    @testset "Boundary correction caches are prepared" begin
        for kind in (:tlsph, :rigid),
            correction in
            (ShepardKernelCorrection(), KernelCorrection(), GradientCorrection(),
             BlendedGradientCorrection(0.5), MixedKernelGradientCorrection())

            state = structure_fluid_pair_state(; structure_kind=kind,
                                               boundary_density=PressureMirroring(),
                                               boundary_correction=correction,
                                               fluid_options=(;
                                                              correction=KernelCorrection(),
                                                              pressure_acceleration=nothing))
            (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure,
             u_structure) = state
            model = structure.boundary_model
            @test TrixiParticles.compact_support(structure, fluid) ==
                  TrixiParticles.compact_support(fluid, structure) == 2.0
            # Poison every allocated boundary cache: finite results must come from
            # the real interpolation/update path, not allocation contents or the fixture.
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
            if correction isa ShepardKernelCorrection || correction isa KernelCorrection ||
               correction isa MixedKernelGradientCorrection
                # gamma = sum_j V_j W_sj and dw_gamma = sum_j V_j grad_s W_sj / gamma.
                # Here V_f=1100/1005, V_s=700/950, W(1.5)=5/(112pi), W(0)=10/(7pi).
                gamma = 1100.0 / 1005.0 * 5 / (112pi) + 700.0 / 950.0 * 10 / (7pi)
                @test model.cache.kernel_correction_coefficient[1] ≈ gamma
                if !(correction isa ShepardKernelCorrection)
                    dw_gamma = 1100.0 / 1005.0 * grad_s / gamma
                    @test model.cache.dw_gamma[:, 1] ≈ dw_gamma
                    grad_s = (grad_s - 5 / (112pi) * dw_gamma) / gamma
                end
            end
            # One collinear neighbor gives a rank-deficient 2D gradient matrix;
            # gradient and mixed corrections therefore fall back to the identity.
            # Pressure mirroring sets p_s=p_f=500 in this pair, yielding the difference
            # of independently evaluated gradients in the reaction force.
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

    # Independent cubic-spline formulas at h=1, used to sum the complete current
    # hydrodynamic neighborhood without consulting any production neighbor search.
    cubic_value(r) = 10 / (7pi) * (r < 1 ? 1 - 1.5r^2 + 0.75r^3 :
                      r < 2 ? 0.25(2 - r)^3 : 0.0)
    function cubic_gradient(pos_diff)
        r = norm(pos_diff)
        (iszero(r) || r >= 2) && return zero(pos_diff)
        derivative = 10 / (7pi) * (r < 1 ? -3r + 2.25r^2 : -0.75(2 - r)^2)
        return derivative / r * pos_diff
    end

    function check_boundary_caches(state, correction)
        (; structure, semi, u_structure, u_fluid) = state
        model = structure.boundary_model
        coords = TrixiParticles.current_coordinates(u_structure, structure)
        neighbor_coords = hcat(u_fluid, coords)
        volumes = [1100.0 / 1005.0; model.hydrodynamic_mass ./ 950.0]
        for particle in eachparticle(structure)
            gamma = 0.0
            sum_gradient = zeros(2)
            for neighbor in axes(neighbor_coords, 2)
                system_index = neighbor == 1 ? 1 : 2
                semi.interaction_matrix[2, system_index] || continue
                pos_diff = coords[:, particle] - neighbor_coords[:, neighbor]
                gamma += volumes[neighbor] * cubic_value(norm(pos_diff))
                sum_gradient += volumes[neighbor] * cubic_gradient(pos_diff)
            end
            dw = sum_gradient / gamma
            if haskey(model.cache, :kernel_correction_coefficient)
                @test model.cache.kernel_correction_coefficient[particle] ≈ gamma
            end
            if haskey(model.cache, :dw_gamma)
                @test model.cache.dw_gamma[:, particle] ≈ dw
            end
            if haskey(model.cache, :correction_matrix)
                moment = zeros(2, 2)
                for neighbor in axes(neighbor_coords, 2)
                    system_index = neighbor == 1 ? 1 : 2
                    semi.interaction_matrix[2, system_index] || continue
                    pos_diff = coords[:, particle] - neighbor_coords[:, neighbor]
                    grad = cubic_gradient(pos_diff)
                    if correction isa MixedKernelGradientCorrection
                        grad = (grad - cubic_value(norm(pos_diff)) * dw) / gamma
                    end
                    moment -= volumes[neighbor] * grad * pos_diff'
                end
                expected = abs(det(moment)) < 1.0e-9 ? Matrix{Float64}(I, 2, 2) :
                           inv(moment)
                @test model.cache.correction_matrix[:, :, particle] ≈ expected
            end
        end
    end

    @testset "Current hydrodynamic structure neighborhoods" begin
        search_configs = ((; neighborhood_search=GridNeighborhoodSearch{2}(),
                           neighborhood_search_handler=SharedNHSHandler),
                          (; neighborhood_search=GridNeighborhoodSearch{2}(),
                           neighborhood_search_handler=PairsNHSHandler),
                          (; neighborhood_search=PrecomputedNeighborhoodSearch{2}()),
                          (; neighborhood_search=nothing))
        for kind in (:tlsph, :rigid),
            correction in (ShepardKernelCorrection(), KernelCorrection(),
             GradientCorrection(), BlendedGradientCorrection(0.5),
             MixedKernelGradientCorrection()),
            search_options in search_configs

            state = structure_fluid_pair_state(; structure_kind=kind,
                                               structure_smoothing_length=0.4,
                                               boundary_density=PressureMirroring(),
                                               boundary_correction=correction,
                                               coordinates=[1.5 2.5 1.5; 0.0 0.0 1.0],
                                               search_options...)
            (; structure, semi, ode, u_structure) = state
            @test TrixiParticles.compact_support(structure, state.fluid) ==
                  TrixiParticles.compact_support(state.fluid, structure) == 2.0
            elastic_search = kind == :tlsph ? structure.self_interaction_nhs : nothing
            elastic_matrix = kind == :tlsph ? copy(structure.correction_matrix) : nothing
            for x in (2.5, 3.75, 1.8)
                # Move the second neighbor outside and back inside hydrodynamic
                # support. A frozen elastic list cannot follow these changes.
                u_structure[1, 2] = x
                for field in (:dw_gamma, :kernel_correction_coefficient, :correction_matrix)
                    haskey(structure.boundary_model.cache, field) &&
                        fill!(getproperty(structure.boundary_model.cache, field), NaN)
                end
                TrixiParticles.update_systems_and_nhs(ode.u0.x[1], ode.u0.x[2], semi, 0.0)
                check_boundary_caches(state, correction)
                if kind == :tlsph
                    @test structure.self_interaction_nhs === elastic_search
                    @test structure.correction_matrix == elastic_matrix
                end
            end
            # Rebuilding a gradient correction must not use the previous matrix.
            TrixiParticles.update_systems_and_nhs(ode.u0.x[1], ode.u0.x[2], semi, 0.0)
            check_boundary_caches(state, correction)
            # Exercise handler adaptation, including the pair-local correction searches.
            cpu_semi = TrixiParticles.transfer2cpu(semi)
            @test TrixiParticles.get_boundary_correction_neighborhood_search(cpu_semi.systems[2],
                                                                             cpu_semi.systems[2],
                                                                             cpu_semi).search_radius ==
                  2.0
        end
    end

    @testset "Correction assembly respects disabled interactions" begin
        for kind in (:tlsph, :rigid), handler in (PairsNHSHandler, SharedNHSHandler)
            state = structure_fluid_pair_state(; structure_kind=kind,
                                               boundary_density=PressureMirroring(),
                                               boundary_correction=KernelCorrection(),
                                               interaction_matrix=Bool[1 1; 1 0],
                                               neighborhood_search_handler=handler)
            TrixiParticles.update_systems_and_nhs(state.ode.u0.x[1], state.ode.u0.x[2],
                                                  state.semi, 0.0)
            check_boundary_caches(state, KernelCorrection())
        end
    end

    @testset "Periodic hydrodynamic self-neighbors" begin
        box = PeriodicBox(; min_corner=[-4.0, -4.0], max_corner=[4.0, 4.0])
        for search in (GridNeighborhoodSearch{2}(; periodic_box=box),
             PrecomputedNeighborhoodSearch{2}(; periodic_box=box))
            state = structure_fluid_pair_state(; structure_smoothing_length=0.4,
                                               boundary_density=PressureMirroring(),
                                               boundary_correction=KernelCorrection(),
                                               coordinates=[3.5 -3.5; 0.0 0.0],
                                               neighborhood_search=search)
            TrixiParticles.update_systems_and_nhs(state.ode.u0.x[1], state.ode.u0.x[2],
                                                  state.semi, 0.0)
            cache = state.structure.boundary_model.cache
            # The fluid is outside support. The boundary particles are one unit
            # apart across the periodic boundary, despite their coordinate gap.
            @test cache.kernel_correction_coefficient[1] ≈
                  (700.0 * cubic_value(0.0) + 750.0 * cubic_value(1.0)) / 950.0
            @test cache.kernel_correction_coefficient[2] ≈
                  (750.0 * cubic_value(0.0) + 700.0 * cubic_value(1.0)) / 950.0
        end
    end
end
