@trixi_testset "Create System" begin
    particle_spacing = 0.1
    smoothing_length = 1.2 * particle_spacing

    refinement_criterion_1 = RefinementZone(edge_length_x=1.0, edge_length_y=0.5,
                                            zone_origin=(0.0, 0.0))
    refinement_criterion_2 = RefinementZone(edge_length_x=0.5, edge_length_y=0.75,
                                            zone_origin=(0.5, 1.0))

    ic = InitialCondition(; particle_spacing, coordinates=ones(2, 2), density=[1.0, 1.0])

    @testset "Single Criteria" begin
        particle_refinement = ParticleRefinement(refinement_criterion_1)

        system_parent = WeaklyCompressibleSPHSystem(ic, ContinuityDensity(),
                                                    nothing,
                                                    SchoenbergCubicSplineKernel{2}(),
                                                    smoothing_length;
                                                    particle_refinement)

        systems = TrixiParticles.create_child_systems((system_parent,))

        system_child = systems[2]

        particle_spacing_child = system_child.initial_condition.particle_spacing
        factor = particle_refinement.refinement_pattern.separation_parameter

        @test length(systems) == 2
        @test nparticles(system_parent) == 2
        @test nparticles(system_child) == 0
        @test system_parent.particle_refinement.system_child == system_child
        @test TrixiParticles.refinement_level(system_parent.particle_refinement) == 0
        @test system_child.particle_refinement isa Nothing
        @test particle_spacing_child == factor * smoothing_length
        # TODO: Test mass distribution
    end

    @testset "Multilevel Refinement" begin
        particle_refinement = ParticleRefinement(refinement_criterion_1,
                                                 criteria_next_levels=[
                                                     refinement_criterion_1,
                                                 ])

        system_parent = WeaklyCompressibleSPHSystem(ic, ContinuityDensity(),
                                                    nothing,
                                                    SchoenbergCubicSplineKernel{2}(),
                                                    smoothing_length;
                                                    particle_refinement)

        systems = TrixiParticles.create_child_systems((system_parent,))

        system_child_1 = systems[2]
        system_child_2 = systems[3]

        particle_spacing_child_1 = system_child_1.initial_condition.particle_spacing
        particle_spacing_child_2 = system_child_2.initial_condition.particle_spacing
        factor1 = particle_refinement.refinement_pattern.separation_parameter
        factor2 = factor1^2

        @test length(systems) == 3
        @test nparticles(system_parent) == 2
        @test nparticles(system_child_1) == 0
        @test nparticles(system_child_2) == 0
        @test system_parent.particle_refinement.system_child == system_child_1
        @test system_child_1.particle_refinement.system_child == system_child_2
        @test TrixiParticles.refinement_level(system_parent.particle_refinement) == 0
        @test TrixiParticles.refinement_level(system_child_1.particle_refinement) == 1
        @test system_child_2.particle_refinement isa Nothing
        @test particle_spacing_child_1 == factor1 * smoothing_length
        @test particle_spacing_child_2 == factor2 * smoothing_length
        # TODO: Test mass distribution
    end

    @testset "Multiple Criteria" begin
        particle_refinement = ParticleRefinement(refinement_criterion_1,
                                                 refinement_criterion_2)

        system_parent = WeaklyCompressibleSPHSystem(ic, ContinuityDensity(),
                                                    nothing,
                                                    SchoenbergCubicSplineKernel{2}(),
                                                    smoothing_length;
                                                    particle_refinement)

        @test length(system_parent.particle_refinement.refinement_criteria) == 2

        (; refinement_criteria) = system_parent.particle_refinement
        @test refinement_criteria[1].zone_origin(0, 0, 0, 0, 0, 0, 0) == [0.0, 0.0]
        @test refinement_criteria[2].zone_origin(0, 0, 0, 0, 0, 0, 0) == [0.5, 1.0]
    end
end

@trixi_testset "Refinement Pattern" begin
    struct MockSystem <: TrixiParticles.System{2}
        smoothing_length::Any
    end

    Base.eltype(::MockSystem) = Float64

    mock_system = MockSystem(1.0)

    refinement_patterns = [
        TriangularSplitting(),
        CubicSplitting(),
        HexagonalSplitting(),
        TriangularSplitting(; center_particle=false),
        CubicSplitting(; center_particle=false),
        HexagonalSplitting(; center_particle=false),
    ]

    expected_positions = [
        [[0.0, 0.5], [-0.4330127018922193, -0.25], [0.4330127018922193, -0.25], [0.0, 0.0]],
        [
            [0.35355339059327373, 0.35355339059327373],
            [0.35355339059327373, -0.35355339059327373],
            [-0.35355339059327373, -0.35355339059327373],
            [-0.35355339059327373, 0.35355339059327373],
            [0.0, 0.0],
        ],
        [
            [0.5, 0.0],
            [-0.5, 0.0],
            [0.25, 0.4330127018922193],
            [0.25, -0.4330127018922193],
            [-0.25, 0.4330127018922193],
            [-0.25, -0.4330127018922193],
            [0.0, 0.0],
        ],
        [[0.0, 0.5], [-0.4330127018922193, -0.25], [0.4330127018922193, -0.25]],
        [
            [0.35355339059327373, 0.35355339059327373],
            [0.35355339059327373, -0.35355339059327373],
            [-0.35355339059327373, -0.35355339059327373],
            [-0.35355339059327373, 0.35355339059327373],
        ],
        [
            [0.5, 0.0],
            [-0.5, 0.0],
            [0.25, 0.4330127018922193],
            [0.25, -0.4330127018922193],
            [-0.25, 0.4330127018922193],
            [-0.25, -0.4330127018922193],
        ],
    ]

    @testset "$(refinement_patterns[i])" for i in eachindex(refinement_patterns)
        positions = TrixiParticles.relative_position_children(mock_system,
                                                              refinement_patterns[i])

        @test expected_positions[i] ≈ positions
    end
end

@trixi_testset "Refine Particle" begin
    nx = 5
    n_particles = nx^2
    particle_parent = 12
    particle_spacing = 0.1
    smoothing_length = 1.2 * particle_spacing

    nhs = TrixiParticles.TrivialNeighborhoodSearch{2}(2smoothing_length, 1:n_particles)

    ic = RectangularShape(particle_spacing, (nx, nx), (0.0, 0.0), velocity=ones(2),
                          mass=1.0, density=2.0)

    system_child = WeaklyCompressibleSPHSystem(ic, ContinuityDensity(),
                                               nothing,
                                               SchoenbergCubicSplineKernel{2}(),
                                               smoothing_length)

    refinement_criterion = RefinementZone(edge_length_x=Inf, edge_length_y=Inf,
                                          zone_origin=(0.0, 0.0))

    refinement_patterns = [
        TriangularSplitting(),
        CubicSplitting(),
        HexagonalSplitting(),
        TriangularSplitting(; center_particle=false),
        CubicSplitting(; center_particle=false),
        HexagonalSplitting(; center_particle=false),
    ]

    @testset "$refinement_pattern" for refinement_pattern in refinement_patterns
        particle_refinement = ParticleRefinement(refinement_criterion; refinement_pattern)

        system_parent = WeaklyCompressibleSPHSystem(ic, ContinuityDensity(),
                                                    nothing,
                                                    SchoenbergCubicSplineKernel{2}(),
                                                    smoothing_length;
                                                    particle_refinement)

        n_children = TrixiParticles.nchilds(system_parent, particle_refinement)

        resize!(system_child.mass, n_children)

        relative_positions = TrixiParticles.relative_position_children(system_parent,
                                                                       refinement_pattern)
        mass_ratios = TrixiParticles.mass_distribution(system_parent,
                                                       refinement_pattern)

        particle_refinement.rel_position_children = relative_positions
        particle_refinement.available_children = n_children
        particle_refinement.mass_ratio = mass_ratios

        v_parent = vcat(ic.velocity, ic.density')
        u_parent = copy(ic.coordinates)

        v_child = Array{Float64, 2}(undef, 3, n_children)
        u_child = Array{Float64, 2}(undef, 2, n_children)

        mass_parent = system_parent.mass[particle_parent]

        TrixiParticles.bear_children!(system_child, system_parent, particle_parent,
                                      mass_parent, nhs, particle_refinement,
                                      v_parent, u_parent, v_child, u_child)

        parent_pos = u_parent[:, particle_parent]
        for child in 1:n_children
            @test u_child[:, child] ≈ parent_pos .+ relative_positions[child]
            @test v_child[:, child] == [1.0, 1.0, 2.0]
            @test system_child.mass[child] == mass_parent * mass_ratios[child]
        end
    end
end
