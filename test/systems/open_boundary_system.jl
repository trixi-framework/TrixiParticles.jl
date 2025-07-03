@testset verbose=true "`OpenBoundarySPHSystem`" begin
    @testset verbose=true "Illegal Inputs" begin
        plane = ([0.0, 0.0], [0.0, 1.0])
        flow_direction = (1.0, 0.0)

        # Mock fluid system
        struct FluidSystemMock2 <: TrixiParticles.FluidSystem{2} end
        TrixiParticles.initial_smoothing_length(system::FluidSystemMock2) = 1.0
        TrixiParticles.nparticles(system::FluidSystemMock2) = 1

        inflow = BoundaryZone(; plane, particle_spacing=0.1,
                              plane_normal=flow_direction, density=1.0,
                              open_boundary_layers=2, boundary_type=InFlow())

        error_str = "`reference_velocity` must be either a function mapping " *
                    "each particle's coordinates and time to its velocity, " *
                    "an array where the ``i``-th column holds the velocity of particle ``i`` " *
                    "or, for a constant fluid velocity, a vector of length 2 for a 2D problem holding this velocity"

        reference_velocity = 1.0
        @test_throws ArgumentError(error_str) OpenBoundarySPHSystem(inflow;
                                                                    boundary_model=BoundaryModelLastiwka(),
                                                                    buffer_size=0,
                                                                    fluid_system=FluidSystemMock2(),
                                                                    reference_density=0,
                                                                    reference_pressure=0,
                                                                    reference_velocity)

        error_str = "`reference_pressure` must be either a function mapping " *
                    "each particle's coordinates and time to its pressure, " *
                    "a vector holding the pressure of each particle, or a scalar"

        reference_pressure = [1.0, 1.0]
        @test_throws ArgumentError(error_str) OpenBoundarySPHSystem(inflow;
                                                                    boundary_model=BoundaryModelLastiwka(),
                                                                    buffer_size=0,
                                                                    fluid_system=FluidSystemMock2(),
                                                                    reference_density=0,
                                                                    reference_velocity=[1.0,
                                                                        1.0],
                                                                    reference_pressure)

        error_str = "`reference_density` must be either a function mapping " *
                    "each particle's coordinates and time to its density, " *
                    "a vector holding the density of each particle, or a scalar"

        reference_density = [1.0, 1.0]
        @test_throws ArgumentError(error_str) OpenBoundarySPHSystem(inflow;
                                                                    boundary_model=BoundaryModelLastiwka(),
                                                                    buffer_size=0,
                                                                    fluid_system=FluidSystemMock2(),
                                                                    reference_density,
                                                                    reference_velocity=[1.0,
                                                                        1.0],
                                                                    reference_pressure=0)
    end
    @testset "`show`" begin
        inflow = BoundaryZone(; plane=([0.0, 0.0], [0.0, 1.0]), particle_spacing=0.05,
                              plane_normal=(1.0, 0.0), density=1.0,
                              open_boundary_layers=4, boundary_type=InFlow())
        system = OpenBoundarySPHSystem(inflow; buffer_size=0,
                                       boundary_model=BoundaryModelLastiwka(),
                                       reference_density=0.0,
                                       reference_pressure=0.0,
                                       reference_velocity=[0.0, 0.0],
                                       fluid_system=FluidSystemMock2())

        show_compact = "OpenBoundarySPHSystem{2}(InFlow) with 80 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySPHSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ #particles: ………………………………………………… 80                                                               │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelLastiwka                                            │
        │ boundary type: ………………………………………… InFlow                                                           │
        │ prescribed velocity: ………………………… constant_vector                                                  │
        │ prescribed pressure: ………………………… constant_scalar                                                  │
        │ prescribed density: …………………………… constant_scalar                                                  │
        │ width: ……………………………………………………………… 0.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box

        outflow = BoundaryZone(; plane=([0.0, 0.0], [0.0, 1.0]), particle_spacing=0.05,
                               plane_normal=(1.0, 0.0), density=1.0, open_boundary_layers=4,
                               boundary_type=OutFlow())
        system = OpenBoundarySPHSystem(outflow; buffer_size=0,
                                       boundary_model=BoundaryModelLastiwka(),
                                       reference_density=0.0,
                                       reference_pressure=0.0,
                                       reference_velocity=[0.0, 0.0],
                                       fluid_system=FluidSystemMock2())

        show_compact = "OpenBoundarySPHSystem{2}(OutFlow) with 80 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySPHSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ #particles: ………………………………………………… 80                                                               │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelLastiwka                                            │
        │ boundary type: ………………………………………… OutFlow                                                          │
        │ prescribed velocity: ………………………… constant_vector                                                  │
        │ prescribed pressure: ………………………… constant_scalar                                                  │
        │ prescribed density: …………………………… constant_scalar                                                  │
        │ width: ……………………………………………………………… 0.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box
    end
end
