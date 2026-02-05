# Use `@trixi_testset` to isolate the mock functions in a separate namespace
@trixi_testset "Semidiscretization" begin
    # Mock systems
    struct System1 <: TrixiParticles.AbstractSystem{3} end
    struct System2 <: TrixiParticles.AbstractSystem{3} end

    system1 = System1()
    system2 = System2()

    Base.eltype(::System1) = Float64
    TrixiParticles.coordinates_eltype(::System1) = Float32
    TrixiParticles.u_nvariables(::System1) = 3
    TrixiParticles.u_nvariables(::System2) = 4
    TrixiParticles.v_nvariables(::System1) = 3
    TrixiParticles.v_nvariables(::System2) = 2
    TrixiParticles.nparticles(::System1) = 2
    TrixiParticles.nparticles(::System2) = 3
    TrixiParticles.n_integrated_particles(::System1) = 2
    TrixiParticles.n_integrated_particles(::System2) = 3

    TrixiParticles.compact_support(::System1, neighbor) = 0.2
    TrixiParticles.compact_support(::System2, neighbor) = 0.2

    @testset verbose=true "Constructor" begin
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        # Verification
        @test semi.ranges_u == (1:6, 7:18)
        @test semi.ranges_v == (1:6, 7:12)

        nhs = ((TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
                                                            eachpoint=1:2),
                TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
                                                            eachpoint=1:3)),
               (TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
                                                            eachpoint=1:2),
                TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
                                                            eachpoint=1:3)))
        @test semi.neighborhood_searches == nhs
    end

    @testset verbose=true "Check Configuration" begin
        @testset verbose=true "Structure-Fluid Interaction" begin
            # Mock boundary model
            struct BoundaryModelMock
                hydrodynamic_mass::Any
            end

            # Mock fluid system
            struct FluidSystemMock <: TrixiParticles.AbstractFluidSystem{2}
                surface_tension::Nothing
                FluidSystemMock() = new(nothing)
            end

            kernel = Val(:smoothing_kernel)
            Base.ndims(::Val{:smoothing_kernel}) = 2
            TrixiParticles.compact_support(::Val{:smoothing_kernel}, h) = h

            ic = InitialCondition(; particle_spacing=1.0, coordinates=ones(2, 2),
                                  density=[1.0, 1.0])

            fluid_system = FluidSystemMock()
            model_a = BoundaryModelMock(zeros(2))
            model_b = BoundaryModelDummyParticles([1.0, 1.0], [1.0, 1.0],
                                                  ContinuityDensity(), kernel, 1.0)
            model_c = BoundaryModelMock(zeros(3))

            # FSI without boundary model
            structure_system1 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0)

            error_str = "a boundary model for `TotalLagrangianSPHSystem` must be " *
                        "specified when simulating a fluid-structure interaction."
            @test_throws ArgumentError(error_str) Semidiscretization(fluid_system,
                                                                     structure_system1,
                                                                     neighborhood_search=nothing)

            # FSI with boundary model
            structure_system2 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0,
                                                         boundary_model=model_a)
            structure_system2 = TrixiParticles.initialize_self_interaction_nhs(structure_system2,
                                                                               nothing,
                                                                               nothing)

            @test_nowarn TrixiParticles.check_configuration((structure_system2,
                                                             fluid_system),
                                                            nothing)

            # FSI with wrong boundary model
            structure_system3 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0,
                                                         boundary_model=model_b)

            error_str = "`BoundaryModelDummyParticles` with density calculator " *
                        "`ContinuityDensity` is not yet supported for a `TotalLagrangianSPHSystem`"
            @test_throws ArgumentError(error_str) Semidiscretization(structure_system3,
                                                                     fluid_system,
                                                                     neighborhood_search=nothing)

            # FSI with wrong boundary model
            structure_system4 = TotalLagrangianSPHSystem(ic, kernel, 1.0, 1.0, 1.0,
                                                         boundary_model=model_c)

            error_str = "the boundary model was initialized with 3 particles, " *
                        "but the `TotalLagrangianSPHSystem` has 2 particles."
            @test_throws ArgumentError(error_str) Semidiscretization(structure_system4,
                                                                     fluid_system,
                                                                     neighborhood_search=nothing)
        end

        @testset verbose=true "WCSPH-Boundary Interaction" begin
            kernel = SchoenbergCubicSplineKernel{2}()
            ic = InitialCondition(; particle_spacing=1.0, coordinates=ones(2, 2),
                                  density=[1.0, 1.0])

            boundary_model = BoundaryModelDummyParticles(ic.density, ic.mass,
                                                         SummationDensity(), kernel, 1.0)
            boundary_system = WallBoundarySystem(ic, boundary_model)
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
        │ eltype: …………………………………………………………… Float64                                                          │
        │ coordinates eltype: …………………………… Float32                                                          │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", semi) == show_box
    end

    @testset verbose=true "Source Terms" begin
        TrixiParticles.source_terms(::System1) = SourceTermDamping(damping_coefficient=0.1)
        TrixiParticles.current_density(v, system::System1, particle) = 0.0
        TrixiParticles.current_pressure(v, system::System1, particle) = 0.0

        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        dv_ode = zeros(3 * 2 + 2 * 3)
        du_ode = zeros(3 * 2 + 4 * 3)
        u_ode = zero(du_ode)

        v1 = [1.0 2.0
              3.0 4.0
              5.0 6.0]
        v2 = zeros(4 * 3)
        v_ode = vcat(vec(v1), v2)

        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi, 0.0)

        dv1 = TrixiParticles.wrap_v(dv_ode, system1, semi)
        @test dv1 == -0.1 * v1

        dv2 = TrixiParticles.wrap_v(dv_ode, system2, semi)
        @test iszero(dv2)
    end
end
