@testset verbose=true "Deactivate Out of Bounds Particles" begin
    struct MockSystemOutOfBounds <: TrixiParticles.AbstractSystem{2}
        buffer::TrixiParticles.SystemBuffer
    end

    function TrixiParticles.get_neighborhood_search(system::MockSystemOutOfBounds,
                                                    semi::DummySemidiscretization)
        return (cell_size=(0.1, 0.1),)
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
        semi = DummySemidiscretization()

        # All particles should remain active
        initial_count = count(buffer.active_particle)
        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, cell_list, u,
                                                           semi)
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
        semi = DummySemidiscretization()

        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, cell_list, u,
                                                           semi)

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
        semi = DummySemidiscretization()

        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, cell_list, u,
                                                           semi)

        # All should still be active
        @test all(buffer.active_particle)
    end
end
