
@testset verbose=true "Monghan-Kajtar Repulsive Particles" begin
    @testset "`show`" begin
        boundary_model = BoundaryModelMonaghanKajtar(10.0, 3.0, 0.1, [1.0])

        show_compact = "BoundaryModelMonaghanKajtar(10.0, 3.0, Nothing)"
        @test repr(boundary_model) == show_compact
        @test boundary_model.minimum_distance_ratio == 0.01

        regularized_model = BoundaryModelMonaghanKajtar(10.0, 3.0, 0.1, [1.0];
                                                        minimum_distance_ratio=0.001)
        @test regularized_model.minimum_distance_ratio == 0.001
        @test_throws ArgumentError BoundaryModelMonaghanKajtar(10.0, 3.0, 0.1, [1.0];
                                                               minimum_distance_ratio=0.0)
    end

    @testset "RHS" begin
        particle_spacing = 0.1

        state_equation = StateEquationCole(; sound_speed=1.0, reference_density=1000.0,
                                           exponent=7)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2particle_spacing
        search_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        # 3x3 fluid particles to the left of a 1x3 vertical wall, with the rightmost
        # fluid particle one `particle_spacing` away from the boundary (which is at x=0)
        fluid = rectangular_patch(particle_spacing, (3, 3), perturbation_factor=0.0,
                                  perturbation_factor_position=0.0,
                                  offset=(-1.5particle_spacing, 0.0))
        fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation)

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

        structure_model = BoundaryModelMonaghanKajtar(K, 1.0, particle_spacing,
                                                      fluid.mass)
        structure_system = TotalLagrangianSPHSystem(fluid; smoothing_kernel,
                                                    smoothing_length,
                                                    young_modulus=1.0e6,
                                                    poisson_ratio=0.3,
                                                    boundary_model=structure_model)
        semi_coupled = Semidiscretization(fluid_system, boundary_system, structure_system;
                                          parallelization_backend=SerialBackend())
        ode = semidiscretize(semi_coupled, (0.0, 0.1))
        @test_nowarn TrixiParticles.update_nhs!(semi_coupled, ode.u0.x[2])

        function structure_marker(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            system isa TotalLagrangianSPHSystem || return nothing
            coordinates = TrixiParticles.initial_coordinates(system)
            return fill(Int32(7), size(coordinates, 2))
        end

        mktempdir() do output_directory
            @test_nowarn trixi2vtk(ode.u0, semi_coupled, 0.0; output_directory,
                                   structure_marker)
            data = vtk2trixi(joinpath(output_directory, "structure_1_current.vtu"))
            @test data.structure_marker == fill(Int32(7),
                       TrixiParticles.nparticles(structure_system))

            boundary_data = vtk2trixi(joinpath(output_directory,
                                               "boundary_1_current.vtu");
                                      create_initial_condition=false)
            @test size(boundary_data.coordinates, 2) ==
                  TrixiParticles.nparticles(boundary_system)
            @test TrixiParticles.time_span((0.0, 0.1),
                                           (joinpath(output_directory,
                                                     "boundary_1_current.vtu"),)) ==
                  (0.0, 0.1)
        end
    end
end
