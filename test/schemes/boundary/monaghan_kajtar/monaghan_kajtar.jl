
@testset verbose=true "Monghan-Kajtar Repulsive Particles" begin
    @testset "`show`" begin
        boundary_model = BoundaryModelMonaghanKajtar(10.0, 3.0, 0.1, [1.0])

        show_compact = "BoundaryModelMonaghanKajtar(10.0, 3.0, Nothing)"
        @test repr(boundary_model) == show_compact
    end

    @testset "RHS" begin
        particle_spacing = 0.1

        # The state equation is only needed to unpack `sound_speed`, so we can mock
        # it by using a `NamedTuple`.
        state_equation = (; sound_speed=0.0)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2particle_spacing
        search_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        # 3x3 fluid particles to the left of a 1x3 vertical wall, with the rightmost
        # fluid particle one `particle_spacing` away from the boundary (which is at x=0)
        fluid = rectangular_patch(particle_spacing, (3, 3), perturbation_factor=0.0,
                                  perturbation_factor_position=0.0,
                                  offset=(-1.5particle_spacing, 0.0))
        fluid_system = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length)

        # Use double spacing for the boundary (exactly the opposite of what we would do
        # in a simulation) to test that forces grow infinitely when a fluid particle
        # comes too close to the boundary, independent of the boundary particle spacing.
        boundary = rectangular_patch(2particle_spacing, (1, 3), perturbation_factor=0.0,
                                     perturbation_factor_position=0.0)
        K = 1.0
        spacing_ratio = 0.5
        boundary_model = BoundaryModelMonaghanKajtar(K, spacing_ratio, 2particle_spacing,
                                                     boundary.mass)
        boundary_system = WallBoundarySystem(boundary, boundary_model)

        # Density is integrated with `ContinuityDensity`
        v = vcat(fluid.velocity, fluid.density')
        u = fluid.coordinates

        v_neighbor = zeros(0, TrixiParticles.nparticles(boundary_system))
        u_neighbor = boundary.coordinates

        semi = DummySemidiscretization()

        # Result
        dv = zero(fluid.velocity)
        TrixiParticles.interact!(dv, v, u, v_neighbor, u_neighbor, fluid_system,
                                 boundary_system, semi)

        # Due to the symmetric setup, all particles will only be accelerated horizontally
        @test isapprox(dv[2, :], zeros(TrixiParticles.nparticles(fluid_system)), atol=1e-14)

        # For the leftmost column of fluid particles, the boundary particles are outside the
        # compact support of the kernel.
        @test iszero(dv[:, [1, 4, 7]])

        # The rightmost column of fluid particles should experience strong accelerations
        # towards the left.
        @test all(dv[1, [3, 6, 9]] .< -300)

        # The middle column of fluid particles should experience weaker accelerations
        @test isapprox(dv[1, [2, 5, 8]], [-26.052449, -95.162888, -26.052449])
    end
end
