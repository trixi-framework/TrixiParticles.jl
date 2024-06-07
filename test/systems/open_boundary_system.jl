@testset verbose=true "OpenBoundarySPHSystem" begin
    @testset verbose=true "Illegal Inputs" begin
        plane = ([0.0, 0.0], [0.0, 1.0])
        flow_direction = (1.0, 0.0)

        # Mock fluid system
        struct FluidSystemMock2 <: TrixiParticles.FluidSystem{2, Nothing} end

        inflow = InFlow(; plane, particle_spacing=0.1,
                        flow_direction, density=1.0, open_boundary_layers=2)

        error_str = "`reference_velocity` must be either a function mapping " *
                    "each particle's coordinates and time to its velocity, " *
                    "an array where the ``i``-th column holds the velocity of particle ``i`` " *
                    "or, for a constant fluid velocity, a vector of length 2 for a 2D problem holding this velocity"

        reference_velocity = 1.0
        @test_throws ArgumentError(error_str) OpenBoundarySPHSystem(inflow; sound_speed=1.0,
                                                                    buffer_size=0,
                                                                    fluid_system=FluidSystemMock2(),
                                                                    reference_velocity)

        error_str = "`reference_pressure` must be either a function mapping " *
                    "each particle's coordinates and time to its pressure, " *
                    "a vector holding the pressure of each particle, or a scalar"

        reference_pressure = [1.0, 1.0]
        @test_throws ArgumentError(error_str) OpenBoundarySPHSystem(inflow; sound_speed=1.0,
                                                                    buffer_size=0,
                                                                    fluid_system=FluidSystemMock2(),
                                                                    reference_pressure)

        error_str = "`reference_density` must be either a function mapping " *
                    "each particle's coordinates and time to its density, " *
                    "a vector holding the density of each particle, or a scalar"

        reference_density = [1.0, 1.0]
        @test_throws ArgumentError(error_str) OpenBoundarySPHSystem(inflow; sound_speed=1.0,
                                                                    buffer_size=0,
                                                                    fluid_system=FluidSystemMock2(),
                                                                    reference_density)
    end
    @testset "show" begin
        inflow = InFlow(; plane=([0.0, 0.0], [0.0, 1.0]), particle_spacing=0.05,
                        flow_direction=(1.0, 0.0), density=1.0, open_boundary_layers=4)
        system = OpenBoundarySPHSystem(inflow; sound_speed=1.0, buffer_size=0,
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
        │ boundary: ……………………………………………………… InFlow                                                           │
        │ flow direction: ……………………………………… [1.0, 0.0]                                                       │
        │ prescribed velocity: ………………………… constant_vector                                                  │
        │ prescribed pressure: ………………………… constant_scalar                                                  │
        │ prescribed density: …………………………… constant_scalar                                                  │
        │ width: ……………………………………………………………… 0.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box

        outflow = OutFlow(; plane=([0.0, 0.0], [0.0, 1.0]), particle_spacing=0.05,
                          flow_direction=(1.0, 0.0), density=1.0, open_boundary_layers=4)
        system = OpenBoundarySPHSystem(outflow; sound_speed=1.0, buffer_size=0,
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
        │ boundary: ……………………………………………………… OutFlow                                                          │
        │ flow direction: ……………………………………… [1.0, 0.0]                                                       │
        │ prescribed velocity: ………………………… constant_vector                                                  │
        │ prescribed pressure: ………………………… constant_scalar                                                  │
        │ prescribed density: …………………………… constant_scalar                                                  │
        │ width: ……………………………………………………………… 0.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end
end
