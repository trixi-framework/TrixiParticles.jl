@trixi_testset "Resize and Copy" begin
    n_particles = 10
    dict_ = Dict(12 => "Filled", 0 => "Empty")

    @testset "$(dict_[n_particles_child]) Child System" for n_particles_child in [0, 12]
        refinement_patterns = [
            TriangularSplitting(),
            CubicSplitting(),
            HexagonalSplitting(),
        ]

        candidates = [[1, 10], [1, 3, 5, 7, 9], [2, 4, 6, 8], collect(1:n_particles)]

        old_ranges_v = ([1:(3n_particles)],
                        [(3n_particles + 1):(3n_particles + 3n_particles_child)])
        old_ranges_u = ([1:(2n_particles)],
                        [(2n_particles + 1):(2n_particles + 2n_particles_child)])

        old_nparticles = (n_particles, n_particles_child)

        ic = InitialCondition(; particle_spacing=0.1, coordinates=ones(2, n_particles),
                              density=ones(n_particles))

        refinement_criterion = RefinementZone(edge_length_x=Inf, edge_length_y=Inf,
                                              zone_origin=(0.0, 0.0))

        refinement_callback = ParticleRefinementCallback(interval=1)
        callback = refinement_callback.affect!

        @testset "Refinement Pattern $refinement_pattern" for refinement_pattern in refinement_patterns
            @testset "Refinement Candidates $j" for j in eachindex(candidates)
                particle_refinement = ParticleRefinement(refinement_criterion;
                                                         refinement_pattern=refinement_pattern)

                system_parent = WeaklyCompressibleSPHSystem(ic, ContinuityDensity(),
                                                            nothing,
                                                            SchoenbergCubicSplineKernel{2}(),
                                                            1.0; particle_refinement)

                semi = Semidiscretization(system_parent)

                # Resize emtpy child system
                resize!(semi.systems[2].mass, n_particles_child)

                # Add candidates
                system_parent.particle_refinement.candidates = candidates[j]

                # Create vectors filled with the corresponding particle index
                v_ode_parent = reshape(stack([i * ones(Int, 3) for i in 1:n_particles]),
                                       (3 * n_particles,))
                u_ode_parent = reshape(stack([i * ones(Int, 2) for i in 1:n_particles]),
                                       (2 * n_particles,))

                if n_particles_child > 0
                    v_ode_child = reshape(stack([i * ones(Int, 3)
                                                 for i in 1:n_particles_child]),
                                          (3 * n_particles_child,))
                    u_ode_child = reshape(stack([i * ones(Int, 2)
                                                 for i in 1:n_particles_child]),
                                          (2 * n_particles_child,))

                    v_ode = vcat(v_ode_parent, v_ode_child)
                    u_ode = vcat(u_ode_parent, u_ode_child)
                else
                    v_ode = v_ode_parent
                    u_ode = u_ode_parent
                end

                _v_cache = copy(v_ode)
                _u_cache = copy(u_ode)

                TrixiParticles.resize_and_copy!(callback, semi, v_ode, u_ode,
                                                _v_cache, _u_cache)

                # Iterator without candidates
                eachparticle_excluding_candidates = (setdiff(1:n_particles, candidates[j]),
                                                     Base.OneTo(n_particles_child))

                n_candidates = length(candidates[j])
                n_children = TrixiParticles.nchilds(system_parent, particle_refinement)
                n_total_particles = n_particles + n_particles_child - n_candidates +
                                    n_candidates * n_children

                # Ranges after resizing
                new_ranges_v = ([1:(3nparticles(system_parent))],
                                [(3nparticles(system_parent) + 1):(3n_total_particles)])
                new_ranges_u = ([1:(2nparticles(system_parent))],
                                [(2nparticles(system_parent) + 1):(2n_total_particles)])

                @testset "Iterators And Ranges" begin
                    @test callback.nparticles_cache == old_nparticles
                    @test callback.ranges_v_cache == old_ranges_v
                    @test callback.ranges_u_cache == old_ranges_u
                    @test callback.eachparticle_cache == eachparticle_excluding_candidates

                    @test nparticles(system_parent) == n_particles - n_candidates ==
                          length(eachparticle_excluding_candidates[1])
                    @test nparticles(semi.systems[2]) ==
                          n_candidates * n_children + n_particles_child

                    @test semi.ranges_v == new_ranges_v
                    @test semi.ranges_u == new_ranges_u
                end

                @testset "Resized Integrator-Array" begin
                    # Parent system
                    v_parent = TrixiParticles.wrap_v(v_ode, system_parent, semi)

                    @test size(v_parent, 2) == length(eachparticle_excluding_candidates[1])

                    # Test `copy_values_v!`
                    for particle in 1:nparticles(system_parent)
                        for dim in 1:3
                            @test v_parent[dim, particle] ==
                                  eachparticle_excluding_candidates[1][particle]
                        end
                    end

                    u_parent = TrixiParticles.wrap_u(u_ode, system_parent, semi)

                    @test size(u_parent, 2) == length(eachparticle_excluding_candidates[1])

                    # Test `copy_values_u!`
                    for particle in 1:nparticles(system_parent)
                        for dim in 1:ndims(system_parent)
                            @test u_parent[dim, particle] ==
                                  eachparticle_excluding_candidates[1][particle]
                        end
                    end

                    # Child system
                    v_child = TrixiParticles.wrap_v(v_ode, semi.systems[2], semi)
                    @test size(v_child, 2) == n_candidates * n_children + n_particles_child

                    # Test `copy_values_v!`
                    for particle in 1:n_particles_child
                        for dim in 1:3
                            @test v_child[dim, particle] == particle
                        end
                    end

                    u_child = TrixiParticles.wrap_u(u_ode, semi.systems[2], semi)
                    @test size(u_child, 2) == n_candidates * n_children + n_particles_child

                    # Test `copy_values_u!`
                    for particle in 1:n_particles_child
                        for dim in 1:2
                            @test u_child[dim, particle] == particle
                        end
                    end
                end
            end
        end
    end
end
