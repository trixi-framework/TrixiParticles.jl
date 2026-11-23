@testset verbose=true "Deactivate Out of Bounds Particles" begin
    struct MockSystemOutOfBounds <: TrixiParticles.AbstractSystem{2}
        buffer::TrixiParticles.SystemBuffer
    end

    TrixiParticles.nparticles(system::MockSystemOutOfBounds) = length(system.buffer.active_particle)
    Base.eltype(system::MockSystemOutOfBounds) = Float64

    @testset "Particles Inside Bounds" begin
        # Setup: 5 particles, all inside bounds
        buffer = TrixiParticles.SystemBuffer(5, 0)
        system = MockSystemOutOfBounds(buffer)

        u = [-0.5 0.0 0.5 -0.8 0.8
             -0.5 -0.5 0.0 0.5 0.5]

        cell_list = TrixiParticles.FullGridCellList(; min_corner=(-1.0, -1.0),
                                                    max_corner=(1.0, 1.0),
                                                    search_radius=0.1)
        dummy_nhs = (; cell_size=0.1, n_cells=0, periodic_box=nothing, cell_list)
        semi = DummySemidiscretization()

        # All particles should remain active
        initial_count = count(buffer.active_particle)
        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, dummy_nhs,
                                                           cell_list, u, u, semi)
        @test count(buffer.active_particle) == initial_count
    end

    @testset "Particles Outside Bounds" begin
        # Setup: 5 particles, some outside bounds
        buffer = TrixiParticles.SystemBuffer(5, 0)
        system = MockSystemOutOfBounds(buffer)

        # Particles 3 and 5 are outside the bounds
        u = [-0.5 0.0 2.0 -0.8 -2.0
             -0.5 -0.5 0.0 0.5 0.5]

        cell_list = TrixiParticles.FullGridCellList(; min_corner=(-1.0, -1.0),
                                                    max_corner=(1.0, 1.0),
                                                    search_radius=0.1)
        dummy_nhs = (; cell_size=0.1, n_cells=0, periodic_box=nothing, cell_list)
        semi = DummySemidiscretization()

        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, dummy_nhs,
                                                           cell_list, u, u, semi)

        # Particles 3 and 5 should be deactivated
        @test buffer.active_particle[3] == false
        @test buffer.active_particle[5] == false
        # Others should still be active
        @test buffer.active_particle[1] == true
        @test buffer.active_particle[2] == true
        @test buffer.active_particle[4] == true

        @test TrixiParticles.each_active_particle(system, buffer) == [1, 2, 4]
    end

    @testset "Edge Cases" begin
        # Test the 1001//1000 padding logic
        buffer = TrixiParticles.SystemBuffer(3, 0)
        system = MockSystemOutOfBounds(buffer)

        # Particles directly at the boundaries (should remain active)
        u = [-1.0 1.0 0.0
             -1.0 1.0 0.0]

        cell_list = TrixiParticles.FullGridCellList(; min_corner=(-1.0, -1.0),
                                                    max_corner=(1.0, 1.0),
                                                    search_radius=0.1)
        dummy_nhs = (; cell_size=0.1, n_cells=0, periodic_box=nothing, cell_list)
        semi = DummySemidiscretization()

        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, dummy_nhs,
                                                           cell_list, u, u, semi)

        # All should still be active
        @test all(buffer.active_particle)
    end
end

@testset "Unequal query and neighbor counts" begin
    kernel = SchoenbergCubicSplineKernel{2}()
    initial_conditions = (InitialCondition(; coordinates=[0.0 0.5 4.0; 0.0 0.0 0.0],
                                           density=1000.0, particle_spacing=1.0),
                          InitialCondition(; coordinates=reshape([0.75, 0.0], 2, 1),
                                           density=1000.0, particle_spacing=1.0))
    systems = map(initial_conditions) do ic
        WeaklyCompressibleSPHSystem(ic; smoothing_kernel=kernel, smoothing_length=1.0,
                                    density_calculator=ContinuityDensity(),
                                    state_equation=nothing)
    end
    trivial = TrixiParticles.TrivialNeighborhoodSearch{2}()
    templates = (PrecomputedNeighborhoodSearch{2}(),
                 PrecomputedNeighborhoodSearch{2}(; update_neighborhood_search=trivial),
                 GridNeighborhoodSearch{2}(), trivial)
    cases = ((systems[1], systems[2], [1, 1, 0], [0, 0, 1]),
             (systems[2], systems[1], [2], [1]))

    function count_neighbors(search, x, y)
        counts = zeros(Int, size(x, 2))
        PointNeighbors.foreach_point_neighbor(x, y, search;
                                              parallelization_backend=SerialBackend()) do particle,
                                                                                          _,
                                                                                          _,
                                                                                          _
            counts[particle] += 1
        end
        return counts
    end

    for template in templates, (system, neighbor, expected, expected_moved) in cases
        x = TrixiParticles.initial_coordinates(system)
        y = copy(TrixiParticles.initial_coordinates(neighbor))
        search = TrixiParticles.create_neighborhood_search(template, system, neighbor)
        PointNeighbors.initialize!(search, x, y; parallelization_backend=SerialBackend())
        @test count_neighbors(search, x, y) == expected

        y[1, 1] += 4.0
        PointNeighbors.update!(search, x, y; points_moving=(false, true),
                               parallelization_backend=SerialBackend())
        @test count_neighbors(search, x, y) == expected_moved
    end
end
