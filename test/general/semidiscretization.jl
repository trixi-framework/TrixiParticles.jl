# Use `@trixi_testset` to isolate the mock functions in a separate namespace
@trixi_testset "Semidiscretization" begin
    # Mock systems
    struct System1 <: TrixiParticles.System{3, Nothing} end
    struct System2 <: TrixiParticles.System{3, Nothing} end

    system1 = System1()
    system2 = System2()

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
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        # Verification
        @test semi.ranges_u == (1:6, 7:18)
        @test semi.ranges_v == (1:6, 7:12)

        nhs = ((TrixiParticles.TrivialNeighborhoodSearch{3}(0.2, Base.OneTo(2)),
                TrixiParticles.TrivialNeighborhoodSearch{3}(0.2, Base.OneTo(3))),
               (TrixiParticles.TrivialNeighborhoodSearch{3}(0.2, Base.OneTo(2)),
                TrixiParticles.TrivialNeighborhoodSearch{3}(0.2, Base.OneTo(3))))
        @test semi.neighborhood_searches == nhs
    end

    @testset verbose=true "Check Configuration" begin
        @testset verbose=true "Solid-Fluid Interaction" begin
            # Mock boundary model
            struct BoundaryModelMock end

            # Mock fluid system
            struct FluidSystemMock <: TrixiParticles.FluidSystem{2, Nothing} end

            kernel = Val(:smoothing_kernel)
            Base.ndims(::Val{:smoothing_kernel}) = 2

            ic = InitialCondition(; particle_spacing=1.0, coordinates=ones(2, 2),
                                  density=[1.0, 1.0])

            fluid_system = FluidSystemMock()
            model_a = BoundaryModelMock()
            model_b = BoundaryModelDummyParticles([1.0], [1.0], ContinuityDensity(), kernel,
                                                  1.0)

            # FSI without boundary model.
            solid_system1 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0)

            error_str = "a boundary model for `TotalLagrangianSPHSystem` must be " *
                        "specified when simulating a fluid-structure interaction."
            @test_throws ArgumentError(error_str) Semidiscretization(fluid_system,
                                                                     solid_system1,
                                                                     neighborhood_search=nothing)

            # FSI with boundary model
            solid_system2 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0,
                                                     boundary_model=model_a)

            @test_nowarn TrixiParticles.check_configuration((solid_system2, fluid_system))

            # FSI with wrong boundary model
            solid_system3 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0,
                                                     boundary_model=model_b)

            error_str = "`BoundaryModelDummyParticles` with density calculator " *
                        "`ContinuityDensity` is not yet supported for a `TotalLagrangianSPHSystem`"
            @test_throws ArgumentError(error_str) Semidiscretization(solid_system3,
                                                                     fluid_system,
                                                                     neighborhood_search=nothing)
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
                        "`state_equation` for all boundary models"
            @test_throws ArgumentError(error_str) Semidiscretization(fluid_system,
                                                                     boundary_system)
        end
    end

    @testset verbose=true "`show`" begin
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        show_compact = "Semidiscretization($System1(), $System2(), neighborhood_search=TrivialNeighborhoodSearch)"
        @test repr(semi) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ Semidiscretization                                                                               │
        │ ══════════════════                                                                               │
        │ #spatial dimensions: ………………………… 3                                                                │
        │ #systems: ……………………………………………………… 2                                                                │
        │ neighborhood search: ………………………… TrivialNeighborhoodSearch                                        │
        │ total #particles: ………………………………… 5                                                                │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", semi) == show_box
    end

    @testset verbose=true "Source Terms" begin
        TrixiParticles.source_terms(::System1) = SourceTermDamping(damping_coefficient=0.1)
        TrixiParticles.particle_density(v, system::System1, particle) = 0.0
        TrixiParticles.particle_pressure(v, system::System1, particle) = 0.0

        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        dv_ode = zeros(3 * 2 + 2 * 3)
        du_ode = zeros(3 * 2 + 4 * 3)
        u_ode = zero(du_ode)

        v1 = [1.0 2.0
              3.0 4.0
              5.0 6.0]
        v2 = zeros(4 * 3)
        v_ode = vcat(vec(v1), v2)

        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi)

        dv1 = TrixiParticles.wrap_v(dv_ode, system1, semi)
        @test dv1 == -0.1 * v1

        dv2 = TrixiParticles.wrap_v(dv_ode, system2, semi)
        @test iszero(dv2)
    end
end
