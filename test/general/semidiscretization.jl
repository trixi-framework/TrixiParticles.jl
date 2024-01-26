# Use `@trixi_testset` to isolate the mock functions in a separate namespace
@trixi_testset "Semidiscretization" begin
    # Mock systems
    struct System1 <: TrixiParticles.System{3} end
    struct System2 <: TrixiParticles.System{3} end

    system1 = System1()
    system2 = System2()

    TrixiParticles.timer_name(::System1) = "mock1"
    TrixiParticles.timer_name(::System2) = "mock2"

    Base.ndims(::System1) = 2
    Base.ndims(::System2) = 2

    TrixiParticles.u_nvariables(::System1) = 3
    TrixiParticles.u_nvariables(::System2) = 4
    TrixiParticles.v_nvariables(::System1) = 3
    TrixiParticles.v_nvariables(::System2) = 2
    TrixiParticles.nparticles(::System1) = 2
    TrixiParticles.nparticles(::System2) = 3
    TrixiParticles.n_moving_particles(::System1) = 2
    TrixiParticles.n_moving_particles(::System2) = 3

    TrixiParticles.compact_support(::System1, neighbor) = 0.2
    TrixiParticles.compact_support(::System2, neighbor) = 0.2

    @testset verbose=true "Constructor" begin
        semi = Semidiscretization(system1, system2)

        # Verification
        @test semi.ranges_u == (1:6, 7:18)
        @test semi.ranges_v == (1:6, 7:12)

        nhs = ((TrixiParticles.TrivialNeighborhoodSearch{2}(0.2, Base.OneTo(2)),
                TrixiParticles.TrivialNeighborhoodSearch{2}(0.2, Base.OneTo(3))),
               (TrixiParticles.TrivialNeighborhoodSearch{2}(0.2, Base.OneTo(2)),
                TrixiParticles.TrivialNeighborhoodSearch{2}(0.2, Base.OneTo(3))))
        @test semi.neighborhood_searches == nhs
    end

    @testset verbose=true "Check Configuration" begin
        @testset verbose=true "Solid-Fluid Interaction" begin
            # Mock boundary model
            struct BoundaryModelMock <: TrixiParticles.BoundaryModel end

            model = BoundaryModelMock()
            TrixiParticles.compact_support(system, ::BoundaryModelMock, neighbor) = 0.2

            kernel = SchoenbergCubicSplineKernel{2}()

            ic = InitialCondition(; particle_spacing=1.0, coordinates=ones(2, 2),
                                  density=[1.0, 1.0])

            # FSI without boundary model.
            solid_system1 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0,
                                                     boundary_model=1)

            error_str = "Please specify a boundary model for `TotalLagrangianSPHSystem` " *
                         "when simulating a $(TrixiParticles.timer_name(system1))-structure interaction."
            @test_throws ArgumentError(error_str) Semidiscretization(system1,
                                                                      solid_system1)

            # FSI with boundary model.
            solid_system2 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0,
                                                     boundary_model=model)

            semi1 = Semidiscretization(system1, system2, solid_system2)
            semi2 = Semidiscretization(system1, solid_system2, system2)
            semi3 = Semidiscretization(solid_system2, system1, system2)

            semi_expected = (solid_system2, system1, system2)

            @test semi_expected == semi1.systems
            @test semi_expected == semi2.systems
            @test semi_expected == semi3.systems
        end

        @testset verbose=true "WCSPH-Boundary Interaction" begin
            kernel = SchoenbergCubicSplineKernel{2}()
            ic = InitialCondition(; particle_spacing=1.0, coordinates=ones(2, 2),
                                  density=[1.0, 1.0])

            boundary_model = BoundaryModelDummyParticles(ic.density, ic.mass,
                                                         SummationDensity(), kernel, 1.0)
            boundary_system = BoundarySPHSystem(ic, boundary_model)
            fluid_system = WeaklyCompressibleSPHSystem(ic, SummationDensity(), nothing,
                                                       kernel, 1.0)

            error_str = "`WeaklyCompressibleSPHSystem` cannot be used without setting a " *
                         "`state_equation` for all boundary systems"
            @test_throws ArgumentError(error_str) Semidiscretization(fluid_system,
                                                                      boundary_system)
        end
    end

    @testset verbose=true "show" begin
        semi = Semidiscretization(system1, system2)

        show_compact = "Semidiscretization($System1(), $System2(), neighborhood_search=TrivialNeighborhoodSearch)"
        @test repr(semi) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ Semidiscretization                                                                               │
        │ ══════════════════                                                                               │
        │ #spatial dimensions: ………………………… 2                                                                │
        │ #systems: ……………………………………………………… 2                                                                │
        │ neighborhood search: ………………………… TrivialNeighborhoodSearch                                        │
        │ damping coefficient: ………………………… nothing                                                          │
        │ total #particles: ………………………………… 5                                                                │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", semi) == show_box
    end
end
