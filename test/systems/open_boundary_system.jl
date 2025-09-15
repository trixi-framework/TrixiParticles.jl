@testset verbose=true "`OpenBoundarySPHSystem`" begin
    @testset "`show`" begin

        # Mock fluid system
        struct FluidSystemMock2 <: TrixiParticles.AbstractFluidSystem{2} end
        TrixiParticles.initial_smoothing_length(system::FluidSystemMock2) = 1.0
        TrixiParticles.nparticles(system::FluidSystemMock2) = 1

        inflow = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.0]),
                              particle_spacing=0.05,
                              plane_normal=(1.0, 0.0), density=1.0,
                              open_boundary_layers=4, boundary_type=InFlow())
        system = OpenBoundarySPHSystem(inflow; buffer_size=0,
                                       boundary_model=BoundaryModelCharacteristicsLastiwka(),
                                       fluid_system=FluidSystemMock2())

        show_compact = "OpenBoundarySPHSystem{2}() with 80 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySPHSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ #particles: ………………………………………………… 80                                                               │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ #boundary_zones: …………………………………… 1                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelCharacteristicsLastiwka                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box

        outflow = BoundaryZone(; boundary_face=([5.0, 0.0], [5.0, 1.0]),
                               particle_spacing=0.05,
                               plane_normal=(1.0, 0.0), density=1.0, open_boundary_layers=4,
                               boundary_type=OutFlow())
        system = OpenBoundarySPHSystem(outflow; buffer_size=0,
                                       boundary_model=BoundaryModelMirroringTafuni(),
                                       fluid_system=FluidSystemMock2())

        show_compact = "OpenBoundarySPHSystem{2}() with 80 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySPHSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ #particles: ………………………………………………… 80                                                               │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ #boundary_zones: …………………………………… 1                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelMirroringTafuni                                     │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box

        system = OpenBoundarySPHSystem(outflow, inflow; buffer_size=0,
                                       boundary_model=BoundaryModelMirroringTafuni(),
                                       fluid_system=FluidSystemMock2())

        show_compact = "OpenBoundarySPHSystem{2}() with 160 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySPHSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ #particles: ………………………………………………… 160                                                              │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ #boundary_zones: …………………………………… 2                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelMirroringTafuni                                     │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box
    end
end
