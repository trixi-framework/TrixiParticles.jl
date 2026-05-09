# Use `@trixi_testset` to isolate the mock functions in a separate namespace
@trixi_testset "Semidiscretization" begin
    # Mock systems
    struct System1 <: TrixiParticles.AbstractStructureSystem{3} end
    struct System2 <: TrixiParticles.AbstractStructureSystem{3} end

    system1 = System1()
    system2 = System2()

    Base.eltype(::System1) = Float64
    TrixiParticles.coordinates_eltype(::System1) = Float32
    TrixiParticles.u_nvariables(::System1) = 3
    TrixiParticles.u_nvariables(::System2) = 3
    TrixiParticles.v_nvariables(::System1) = 3
    TrixiParticles.v_nvariables(::System2) = 4
    TrixiParticles.nparticles(::System1) = 2
    TrixiParticles.nparticles(::System2) = 3
    TrixiParticles.n_integrated_particles(::System1) = 2
    TrixiParticles.n_integrated_particles(::System2) = 3

    TrixiParticles.compact_support(::System1, neighbor) = 0.2
    TrixiParticles.compact_support(::System2, neighbor) = 0.2

    @testset verbose=true "Constructor" begin
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        # Verification
        @test semi.ranges_u == (1:6, 7:15)
        @test semi.ranges_v == (1:6, 7:18)
        @test semi.interaction_matrix == trues(2, 2)

        nhs = [TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
               eachpoint=1:2)
               TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
               eachpoint=1:2);;
               TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
               eachpoint=1:3)
               TrixiParticles.TrivialNeighborhoodSearch{3}(search_radius=0.2,
               eachpoint=1:3)]
        @test semi.neighborhood_searches == nhs

        @test_throws ArgumentError Semidiscretization(system1, system2;
                                                      neighborhood_search=nothing,
                                                      interaction_matrix=trues(1, 1))
        @test_throws ArgumentError Semidiscretization(system1, system2;
                                                      neighborhood_search=nothing,
                                                      interaction_matrix=Any[true 1;
                                                                             true true])
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
            structure_system1 = TotalLagrangianSPHSystem(ic;
                                                         smoothing_kernel=kernel,
                                                         smoothing_length=1.0,
                                                         young_modulus=1.0,
                                                         poisson_ratio=1.0)

            error_str = "a boundary model for `TotalLagrangianSPHSystem` must be " *
                        "specified when simulating a fluid-structure interaction."
            @test_throws ArgumentError(error_str) Semidiscretization(fluid_system,
                                                                     structure_system1,
                                                                     neighborhood_search=nothing)

            # FSI with boundary model
            structure_system2 = TotalLagrangianSPHSystem(ic;
                                                         smoothing_kernel=kernel,
                                                         smoothing_length=1.0,
                                                         young_modulus=1.0,
                                                         poisson_ratio=1.0,
                                                         boundary_model=model_a)
            structure_system2 = TrixiParticles.initialize_self_interaction_nhs(structure_system2,
                                                                               nothing,
                                                                               nothing)

            @test_nowarn TrixiParticles.check_configuration((structure_system2,
                                                             fluid_system),
                                                            nothing)

            # FSI with wrong boundary model
            structure_system3 = TotalLagrangianSPHSystem(ic;
                                                         smoothing_kernel=kernel,
                                                         smoothing_length=1.0,
                                                         young_modulus=1.0,
                                                         poisson_ratio=1.0,
                                                         boundary_model=model_b)

            error_str = "`BoundaryModelDummyParticles` with density calculator " *
                        "`ContinuityDensity` is not yet supported for a `TotalLagrangianSPHSystem`"
            @test_throws ArgumentError(error_str) Semidiscretization(structure_system3,
                                                                     fluid_system,
                                                                     neighborhood_search=nothing)

            # FSI with wrong boundary model
            structure_system4 = TotalLagrangianSPHSystem(ic;
                                                         smoothing_kernel=kernel,
                                                         smoothing_length=1.0,
                                                         young_modulus=1.0,
                                                         poisson_ratio=1.0,
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
            fluid_system = WeaklyCompressibleSPHSystem(ic; smoothing_kernel=kernel,
                                                       smoothing_length=1.0,
                                                       density_calculator=SummationDensity(),
                                                       state_equation=nothing)

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

            fluid_a = WeaklyCompressibleSPHSystem(fluid_a_ic;
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation, smoothing_kernel=kernel,
                                                  smoothing_length)
            fluid_b = WeaklyCompressibleSPHSystem(fluid_b_ic;
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation, smoothing_kernel=kernel,
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
            return Array(TrixiParticles.wrap_v(dv_ode, system, semi))
        end

        function matched_phase_interaction_matrix()
            interaction_matrix = trues(4, 4)
            interaction_matrix[1, 4] = false
            interaction_matrix[4, 1] = false
            interaction_matrix[2, 3] = false
            interaction_matrix[3, 2] = false

            return interaction_matrix
        end

        function make_semi(nhs_factory, systems...; filtered=false)
            neighborhood_search = nhs_factory()

            if filtered
                return Semidiscretization(systems...; neighborhood_search,
                                          interaction_matrix=matched_phase_interaction_matrix())
            end

            return Semidiscretization(systems...; neighborhood_search)
        end

        struct TestInterphaseDrag{T}
            coefficient::T
        end

        struct TestKeywordInteraction
            integrate_tlsph_seen::Base.RefValue{Bool}
        end

        function (drag::TestInterphaseDrag)(dv, v_system, u_system, v_neighbor,
                                            u_neighbor, system, neighbor, semi)
            system_coords = TrixiParticles.current_coordinates(u_system, system)
            neighbor_coords = TrixiParticles.current_coordinates(u_neighbor, neighbor)

            TrixiParticles.foreach_point_neighbor(system, neighbor, system_coords,
                                                  neighbor_coords, semi) do particle,
                                                                            neighbor_particle,
                                                                            pos_diff,
                                                                            distance
                v_a = TrixiParticles.current_velocity(v_system, system, particle)
                v_b = TrixiParticles.current_velocity(v_neighbor, neighbor,
                                                      neighbor_particle)
                rho_a = TrixiParticles.current_density(v_system, system, particle)
                rho_b = TrixiParticles.current_density(v_neighbor, neighbor,
                                                       neighbor_particle)
                m_b = TrixiParticles.hydrodynamic_mass(neighbor, neighbor_particle)
                kernel_value = TrixiParticles.smoothing_kernel(system, distance, particle)

                acceleration = drag.coefficient * m_b / (rho_a * rho_b) *
                               kernel_value * (v_b - v_a)

                for i in 1:ndims(system)
                    dv[i, particle] += acceleration[i]
                end
            end

            return dv
        end

        function (interaction::TestKeywordInteraction)(dv, v_system, u_system,
                                                       v_neighbor, u_neighbor,
                                                       system, neighbor, semi;
                                                       integrate_tlsph=false)
            interaction.integrate_tlsph_seen[] = integrate_tlsph
            return dv
        end

        function make_drag_semi(nhs_factory, systems...; drag=false)
            neighborhood_search = nhs_factory()

            if drag
                interaction_matrix = Any[true TestInterphaseDrag(500.0);
                                         TestInterphaseDrag(500.0) true]
            else
                interaction_matrix = Bool[true false;
                                          false true]
            end

            return Semidiscretization(systems...; neighborhood_search,
                                      interaction_matrix=interaction_matrix)
        end

        function make_pressure_interpolation_systems()
            fluid_a_ic = make_particle(0.0, 1000.0, (0.0, 0.0))
            fluid_b_ic = make_particle(0.25, 1030.0, (0.0, 0.0))
            boundary_ic = make_particle(0.125, 1000.0, (0.0, 0.0))

            fluid_a = WeaklyCompressibleSPHSystem(fluid_a_ic;
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation, smoothing_kernel=kernel,
                                                  smoothing_length)
            fluid_b = WeaklyCompressibleSPHSystem(fluid_b_ic;
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation, smoothing_kernel=kernel,
                                                  smoothing_length)

            boundary_model = BoundaryModelDummyParticles(boundary_ic.density,
                                                         boundary_ic.mass,
                                                         AdamiPressureExtrapolation(;
                                                             allow_loop_flipping=false),
                                                         kernel, smoothing_length;
                                                         state_equation=state_equation,
                                                         correction=nothing)
            boundary = WallBoundarySystem(boundary_ic, boundary_model)

            return fluid_a, fluid_b, boundary
        end

        function updated_boundary_pressure(semi)
            v_ode, u_ode = create_ode_state(semi)

            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

            return copy(last(semi.systems).boundary_model.pressure)
        end

        # Disable only the cross-coupling between the mismatched fluid/boundary pairs:
        # fluid A must ignore boundary B and fluid B must ignore boundary A.
        # The filtered four-system RHS is compared against reduced semidiscretizations that
        # keep only the allowed neighbors for each system, which makes the intended effect explicit.
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

                filtered_systems = semi_filtered.systems
                @test !TrixiParticles.has_system_interaction(filtered_systems[1],
                                                             filtered_systems[4],
                                                             semi_filtered)
                @test !TrixiParticles.has_system_interaction(filtered_systems[4],
                                                             filtered_systems[1],
                                                             semi_filtered)
                @test !TrixiParticles.has_system_interaction(filtered_systems[2],
                                                             filtered_systems[3],
                                                             semi_filtered)
                @test !TrixiParticles.has_system_interaction(filtered_systems[3],
                                                             filtered_systems[2],
                                                             semi_filtered)
                @test TrixiParticles.has_system_interaction(filtered_systems[1],
                                                            filtered_systems[2],
                                                            semi_filtered)
                @test TrixiParticles.has_system_interaction(filtered_systems[3],
                                                            filtered_systems[1],
                                                            semi_filtered)
                @test TrixiParticles.has_system_interaction(filtered_systems[4],
                                                            filtered_systems[2],
                                                            semi_filtered)

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

        @testset "dummy boundary pressure respects interaction matrix" begin
            semi_filtered = let
                fluid_a, fluid_b, boundary = make_pressure_interpolation_systems()
                interaction_matrix = trues(3, 3)
                interaction_matrix[3, 2] = false

                Semidiscretization(fluid_a, fluid_b, boundary;
                                   neighborhood_search=nothing, interaction_matrix)
            end

            semi_reduced = let
                fluid_a, _, boundary = make_pressure_interpolation_systems()
                Semidiscretization(fluid_a, boundary; neighborhood_search=nothing)
            end

            semi_full = let
                fluid_a, fluid_b, boundary = make_pressure_interpolation_systems()
                Semidiscretization(fluid_a, fluid_b, boundary; neighborhood_search=nothing)
            end

            pressure_filtered = updated_boundary_pressure(semi_filtered)
            pressure_reduced = updated_boundary_pressure(semi_reduced)
            pressure_full = updated_boundary_pressure(semi_full)

            @test pressure_filtered ≈ pressure_reduced
            @test !isapprox(pressure_filtered, pressure_full)
        end

        @testset verbose=true "callable interphase drag" begin
            for (test_name, nhs_factory) in (("without neighborhood search", () -> nothing),
                                             ("with grid neighborhood search",
                                              () -> GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())))
                @testset "$test_name" begin
                    semi_skipped = let
                        fluid_a, fluid_b, _, _ = make_systems()
                        make_drag_semi(nhs_factory, fluid_a, fluid_b)
                    end

                    semi_drag = let
                        fluid_a, fluid_b, _, _ = make_systems()
                        make_drag_semi(nhs_factory, fluid_a, fluid_b; drag=true)
                    end

                    drag_systems = semi_drag.systems
                    @test TrixiParticles.has_system_interaction(drag_systems[1],
                                                                drag_systems[2],
                                                                semi_drag)
                    @test TrixiParticles.has_system_interaction(drag_systems[2],
                                                                drag_systems[1],
                                                                semi_drag)
                    @test semi_drag.interaction_matrix[1, 2] isa TestInterphaseDrag
                    @test semi_drag.interaction_matrix[2, 1] isa TestInterphaseDrag

                    dv_skipped = kick_once(semi_skipped)
                    dv_drag = kick_once(semi_drag)

                    delta_fluid_a = system_dv(dv_drag, semi_drag, 1) -
                                    system_dv(dv_skipped, semi_skipped, 1)
                    delta_fluid_b = system_dv(dv_drag, semi_drag, 2) -
                                    system_dv(dv_skipped, semi_skipped, 2)

                    # Fluid A starts with velocity (0.4, -0.1), fluid B with (-0.2, 0.3).
                    # The drag interaction accelerates both phases toward each other.
                    @test delta_fluid_a[1, 1] < 0
                    @test delta_fluid_a[2, 1] > 0
                    @test delta_fluid_b[1, 1] > 0
                    @test delta_fluid_b[2, 1] < 0
                end
            end
        end

        @testset "split custom interaction receives keyword arguments" begin
            integrate_tlsph_seen = Ref(false)
            fluid_a, fluid_b, _, _ = make_systems()
            interaction = TestKeywordInteraction(integrate_tlsph_seen)
            interaction_matrix = Any[false interaction;
                                     false false]

            semi = Semidiscretization(fluid_a, fluid_b; neighborhood_search=nothing,
                                      interaction_matrix)
            semi_split = Semidiscretization(fluid_a; neighborhood_search=nothing)

            v_ode, u_ode = create_ode_state(semi)
            v_ode_split, u_ode_split = create_ode_state(semi_split)
            dv_ode_split = zero(v_ode_split)

            TrixiParticles.system_interaction_split!(dv_ode_split, v_ode, u_ode, semi,
                                                     v_ode_split, u_ode_split, semi_split)

            @test integrate_tlsph_seen[]
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
        function Base.getproperty(::System1, name::Symbol)
            if name == :acceleration
                return SVector(0.0, 0.0, 0.0)
            end
            error("property $name not defined for `System1`")
        end
        function Base.getproperty(::System2, name::Symbol)
            if name == :acceleration
                return SVector(0.0, 0.0, 1.0)
            end
            error("property $name not defined for `System2`")
        end
        TrixiParticles.source_terms(::System1) = SourceTermDamping(damping_coefficient=0.1)
        TrixiParticles.source_terms(::System2) = nothing
        TrixiParticles.current_density(v, system::System1, particle) = 0.0
        TrixiParticles.current_pressure(v, system::System1, particle) = 0.0

        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        dv_ode = zeros(3 * 2 + 4 * 3)
        du_ode = zeros(3 * 2 + 3 * 3)
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
        @test dv2 == vcat(zeros(2, 3), ones(1, 3), zeros(1, 3))
    end

    @testset verbose=true "drift!" begin
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        du_ode = fill(NaN, 3 * 2 + 3 * 3)
        u_ode = zeros(3 * 2 + 3 * 3)

        v1 = [1.0 2.0
              3.0 4.0
              5.0 6.0]
        v2 = [10.0 11.0 12.0
              20.0 21.0 22.0
              30.0 31.0 32.0
              -999.0 -999.0 -999.0]
        v_ode = vcat(vec(v1), vec(v2))

        returned = TrixiParticles.drift!(du_ode, v_ode, u_ode, (; semi), 0.0)
        @test returned === du_ode

        du1 = TrixiParticles.wrap_u(du_ode, system1, semi)
        @test du1 == v1

        du2 = TrixiParticles.wrap_u(du_ode, system2, semi)
        @test du2 == v2[1:3, :]
    end
end
