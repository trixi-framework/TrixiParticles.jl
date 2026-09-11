@testset verbose=true "`OpenBoundarySystem`" begin
    @testset "`show`" begin

        # Mock fluid system
        struct FluidSystemMock2 <: TrixiParticles.AbstractFluidSystem{2}
            pressure_acceleration_formulation::Nothing
            density_diffusion::Nothing
        end
        TrixiParticles.initial_smoothing_length(system::FluidSystemMock2) = 1.0
        TrixiParticles.nparticles(system::FluidSystemMock2) = 1
        TrixiParticles.system_smoothing_kernel(system::FluidSystemMock2) = nothing
        TrixiParticles.density_calculator(system::FluidSystemMock2) = TrixiParticles.ContinuityDensity()

        inflow = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.0]),
                              particle_spacing=0.05,
                              face_normal=(1.0, 0.0), density=1.0,
                              open_boundary_layers=4, boundary_type=InFlow())
        system = OpenBoundarySystem(inflow; buffer_size=0,
                                    boundary_model=BoundaryModelCharacteristicsLastiwka(),
                                    fluid_system=FluidSystemMock2(nothing, nothing))

        show_compact = "OpenBoundarySystem{2}() with 80 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySystem{2}                                                                            │
        │ ═════════════════════                                                                            │
        │ #particles: ………………………………………………… 80                                                               │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ #boundary_zones: …………………………………… 1                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelCharacteristicsLastiwka                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box

        outflow = BoundaryZone(; boundary_face=([5.0, 0.0], [5.0, 1.0]),
                               particle_spacing=0.05,
                               face_normal=(1.0, 0.0), density=1.0, open_boundary_layers=4,
                               boundary_type=OutFlow())
        system = OpenBoundarySystem(outflow; buffer_size=0,
                                    boundary_model=BoundaryModelMirroringTafuni(),
                                    fluid_system=FluidSystemMock2(nothing, nothing))

        show_compact = "OpenBoundarySystem{2}() with 80 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySystem{2}                                                                            │
        │ ═════════════════════                                                                            │
        │ #particles: ………………………………………………… 80                                                               │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ #boundary_zones: …………………………………… 1                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelMirroringTafuni                                     │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box

        system = OpenBoundarySystem(outflow, inflow; buffer_size=0,
                                    boundary_model=BoundaryModelMirroringTafuni(),
                                    fluid_system=FluidSystemMock2(nothing, nothing))

        show_compact = "OpenBoundarySystem{2}() with 160 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySystem{2}                                                                            │
        │ ═════════════════════                                                                            │
        │ #particles: ………………………………………………… 160                                                              │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ #boundary_zones: …………………………………… 2                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelMirroringTafuni                                     │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box

        system = OpenBoundarySystem(outflow, inflow; buffer_size=0,
                                    boundary_model=BoundaryModelDynamicalPressureZhang(),
                                    fluid_system=FluidSystemMock2(nothing, nothing))

        show_compact = "OpenBoundarySystem{2}() with 160 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySystem{2}                                                                            │
        │ ═════════════════════                                                                            │
        │ #particles: ………………………………………………… 160                                                              │
        │ #buffer_particles: ……………………………… 0                                                                │
        │ #boundary_zones: …………………………………… 2                                                                │
        │ fluid system: …………………………………………… FluidSystemMock2                                                 │
        │ boundary model: ……………………………………… BoundaryModelDynamicalPressureZhang                              │
        │ density diffusion: ……………………………… nothing                                                          │
        │ shifting technique: …………………………… nothing                                                          │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", system) == show_box
    end

    @testset "boundary zone width" begin
        particle_spacing = 0.05
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 0.1

        fluid = RectangularShape(particle_spacing, (2, 2), (0.0, 0.0), density=1.0)
        fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length, sound_speed=1.0)

        boundary_zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.0]),
                                     particle_spacing, face_normal=(1.0, 0.0),
                                     density=1.0, open_boundary_layers=7,
                                     boundary_type=InFlow())

        compact_support = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
        error_str = "boundary zone 1 has width $(boundary_zone.zone_width), " *
                    "but must be at least two compact supports ($(2 * compact_support))"

        @test_nowarn OpenBoundarySystem(boundary_zone; buffer_size=0,
                                        boundary_model=BoundaryModelCharacteristicsLastiwka(),
                                        fluid_system)

        @test_nowarn OpenBoundarySystem(boundary_zone; buffer_size=0,
                                        boundary_model=BoundaryModelDynamicalPressureZhang(),
                                        fluid_system)

        fluid_system_with_shifting = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                                 smoothing_length,
                                                                 sound_speed=1.0,
                                                                 shifting_technique=ParticleShiftingTechnique())

        @test_throws ArgumentError(error_str) OpenBoundarySystem(boundary_zone;
                                                                 buffer_size=0,
                                                                 boundary_model=BoundaryModelDynamicalPressureZhang(),
                                                                 fluid_system=fluid_system_with_shifting)
    end

    @testset "particle conversion in periodic boundary zones" begin
        # Create a block of fluid particles and an outflow boundary zone on the right.
        # The domain is periodic in y-direction, so the outflow boundary zone is periodic.
        # This test checks that a particle leaving the domain through the bottom-right corner,
        # i.e., outside of the boundary zone, but inside it after applying periodicity,
        # is correctly converted to a boundary particle and added inside the boundary zone.
        particle_spacing = 0.1
        density = 1000.0

        fluid = RectangularShape(particle_spacing, (20, 10), (0.0, 0.0); density)
        smoothing_kernel = WendlandC2Kernel{2}()
        fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length=1.5 * particle_spacing,
                                                   sound_speed=10.0, buffer_size=1)

        outflow = BoundaryZone(; boundary_face=([2.0, 0.0], [2.0, 1.0]),
                               face_normal=(-1.0, 0.0), particle_spacing,
                               density, open_boundary_layers=4,
                               boundary_type=OutFlow())
        open_boundary = OpenBoundarySystem(outflow; fluid_system,
                                           boundary_model=BoundaryModelMirroringTafuni(),
                                           buffer_size=1)

        min_corner = [0.0, 0.0]
        max_corner = [2.5, 1.0]
        periodic_box = PeriodicBox(; min_corner, max_corner)
        neighborhood_search = GridNeighborhoodSearch{2}(;
                                                        cell_list=FullGridCellList(;
                                                                                   min_corner,
                                                                                   max_corner),
                                                        update_strategy=ParallelUpdate(),
                                                        periodic_box)
        semi = Semidiscretization(fluid_system, open_boundary;
                                  neighborhood_search)
        ode = semidiscretize(semi, (0.0, 1.0))
        v_ode, u_ode = ode.u0.x
        TrixiParticles.initialize!(open_boundary, semi)

        u_fluid = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        v_open_boundary = TrixiParticles.wrap_v(v_ode, open_boundary, semi)
        u_open_boundary = TrixiParticles.wrap_u(u_ode, open_boundary, semi)

        # Use the fluid particle with index 1 and identify the first inactive particle
        # in the open boundary buffer to which the fluid particle will be converted.
        particle = 1
        particle_new = findfirst(==(false), open_boundary.buffer.active_particle)

        # Move the fluid particle to the bottom-right corner of the domain,
        # outside of the boundary zone but inside it after applying periodicity.
        u_fluid[:, particle] .= [2.05, -eps()]

        TrixiParticles.check_domain!(open_boundary, v_open_boundary, u_open_boundary,
                                     v_ode, u_ode, semi)

        @test !fluid_system.buffer.active_particle[particle]
        @test open_boundary.buffer.active_particle[particle_new]

        # Test that the new boundary particle is inside the boundary zone
        # after applying periodicity.
        @test TrixiParticles.is_in_boundary_zone(outflow,
                                                 SVector(Tuple(u_open_boundary[:,
                                                                               particle_new])))
    end
end
