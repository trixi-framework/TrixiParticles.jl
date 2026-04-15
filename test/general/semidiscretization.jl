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
        @test semi.system_interactions == trues(2, 2)

        nhs = [TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
               eachpoint=1:2)
               TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
               eachpoint=1:2);;
               TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
               eachpoint=1:3)
               TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
               eachpoint=1:3)]
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
                surface_normal_method::Nothing
                FluidSystemMock() = new(nothing, nothing)
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

    @testset verbose=true "Custom System Interactions" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        state_equation = StateEquationCole(sound_speed=10.0,
                                           reference_density=1000.0,
                                           exponent=7)
        smoothing_length = 1.0

        function make_particle(x, density, velocity)
            coordinates = reshape([x, 0.0], 2, 1)
            velocity_matrix = reshape(collect(velocity), 2, 1)

            return InitialCondition(; coordinates, velocity=velocity_matrix,
                                    density=[density], particle_spacing=1.0)
        end

        function make_systems()
            fluid_a_ic = make_particle(0.0, 1005.0, (0.4, -0.1))
            fluid_b_ic = make_particle(0.5, 995.0, (-0.2, 0.3))
            boundary_a_ic = make_particle(1.0, 1002.0, (0.0, 0.0))
            boundary_b_ic = make_particle(1.5, 998.0, (0.0, 0.0))

            fluid_a = WeaklyCompressibleSPHSystem(fluid_a_ic, ContinuityDensity(),
                                                  state_equation, kernel,
                                                  smoothing_length)
            fluid_b = WeaklyCompressibleSPHSystem(fluid_b_ic, ContinuityDensity(),
                                                  state_equation, kernel,
                                                  smoothing_length)

            boundary_model_a = BoundaryModelDummyParticles(boundary_a_ic.density,
                                                           boundary_a_ic.mass,
                                                           ContinuityDensity(), kernel,
                                                           smoothing_length,
                                                           state_equation=state_equation)
            boundary_model_b = BoundaryModelDummyParticles(boundary_b_ic.density,
                                                           boundary_b_ic.mass,
                                                           ContinuityDensity(), kernel,
                                                           smoothing_length,
                                                           state_equation=state_equation)

            boundary_a = WallBoundarySystem(boundary_a_ic, boundary_model_a)
            boundary_b = WallBoundarySystem(boundary_b_ic, boundary_model_b)

            return fluid_a, fluid_b, boundary_a, boundary_b
        end

        function create_ode_state(semi)
            n_u = sum(TrixiParticles.u_nvariables(system) *
                      TrixiParticles.n_integrated_particles(system)
                      for system in semi.systems)
            n_v = sum(TrixiParticles.v_nvariables(system) *
                      TrixiParticles.n_integrated_particles(system)
                      for system in semi.systems)

            v_ode = zeros(n_v)
            u_ode = zeros(n_u)

            TrixiParticles.foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
                TrixiParticles.write_v0!(v, system)
                TrixiParticles.write_u0!(u, system)
            end

            return v_ode, u_ode
        end

        function kick_once(semi)
            v_ode, u_ode = create_ode_state(semi)
            dv_ode = similar(v_ode)

            TrixiParticles.kick!(dv_ode, v_ode, u_ode, semi, 0.0)

            return dv_ode
        end

        function system_dv(dv_ode, semi, system_index)
            system = semi.systems[system_index]
            return Array(TrixiParticles.wrap_v(dv_ode, system, semi, system_index))
        end

        matched_system_pairs(system_index,
                             neighbor_index) = !((system_index == 1 &&
                                                  neighbor_index == 4) ||
                                                 (system_index == 4 &&
                                                  neighbor_index == 1) ||
                                                 (system_index == 2 &&
                                                  neighbor_index == 3) ||
                                                 (system_index == 3 &&
                                                  neighbor_index == 2))

        function make_semi(nhs_factory, systems...; filtered=false)
            neighborhood_search = nhs_factory()

            if filtered
                return Semidiscretization(systems...; neighborhood_search,
                                          system_interaction=matched_system_pairs)
            end

            return Semidiscretization(systems...; neighborhood_search)
        end

        for (test_name, nhs_factory) in (("without neighborhood search", () -> nothing),
             ("with grid neighborhood search",
              () -> GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())))
            @testset "$test_name" begin
                semi_full = let
                    fluid_a, fluid_b, boundary_a, boundary_b = make_systems()
                    make_semi(nhs_factory, fluid_a, fluid_b, boundary_a, boundary_b)
                end

                semi_filtered = let
                    fluid_a, fluid_b, boundary_a, boundary_b = make_systems()
                    make_semi(nhs_factory, fluid_a, fluid_b, boundary_a, boundary_b;
                              filtered=true)
                end

                semi_a = let
                    fluid_a, fluid_b, boundary_a, boundary_b = make_systems()
                    make_semi(nhs_factory, fluid_a, fluid_b, boundary_a)
                end

                semi_b = let
                    fluid_a, fluid_b, boundary_a, boundary_b = make_systems()
                    make_semi(nhs_factory, fluid_a, fluid_b, boundary_b)
                end

                semi_boundary_a = let
                    fluid_a, fluid_b, boundary_a, boundary_b = make_systems()
                    make_semi(nhs_factory, fluid_a, boundary_a)
                end

                semi_boundary_b = let
                    fluid_a, fluid_b, boundary_a, boundary_b = make_systems()
                    make_semi(nhs_factory, fluid_b, boundary_b)
                end

                @test !TrixiParticles.has_system_interaction(semi_filtered, 1, 4)
                @test !TrixiParticles.has_system_interaction(semi_filtered, 4, 1)
                @test !TrixiParticles.has_system_interaction(semi_filtered, 2, 3)
                @test !TrixiParticles.has_system_interaction(semi_filtered, 3, 2)
                @test TrixiParticles.has_system_interaction(semi_filtered, 1, 2)
                @test TrixiParticles.has_system_interaction(semi_filtered, 3, 1)
                @test TrixiParticles.has_system_interaction(semi_filtered, 4, 2)

                dv_full = kick_once(semi_full)
                dv_filtered = kick_once(semi_filtered)
                dv_a = kick_once(semi_a)
                dv_b = kick_once(semi_b)
                dv_boundary_a = kick_once(semi_boundary_a)
                dv_boundary_b = kick_once(semi_boundary_b)

                @test system_dv(dv_filtered, semi_filtered, 1) ≈ system_dv(dv_a, semi_a, 1)
                @test system_dv(dv_filtered, semi_filtered, 2) ≈ system_dv(dv_b, semi_b, 2)
                @test system_dv(dv_filtered, semi_filtered, 3) ≈ system_dv(dv_boundary_a,
                                semi_boundary_a, 2)
                @test system_dv(dv_filtered, semi_filtered, 4) ≈ system_dv(dv_boundary_b,
                                semi_boundary_b, 2)

                @test !isapprox(system_dv(dv_filtered, semi_filtered, 3),
                                system_dv(dv_full, semi_full, 3))
                @test !isapprox(system_dv(dv_filtered, semi_filtered, 4),
                                system_dv(dv_full, semi_full, 4))
            end
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
