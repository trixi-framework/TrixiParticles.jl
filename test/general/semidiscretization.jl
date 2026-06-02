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

    function TrixiParticles.interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
                                      system::System1, neighbor::System1, semi; kwargs...)
        dv[1, 1] += 1
        return dv
    end

    function TrixiParticles.interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
                                      system::System1, neighbor::System2, semi; kwargs...)
        dv[1, 1] += 10
        return dv
    end

    function TrixiParticles.interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
                                      system::System2, neighbor::System1, semi; kwargs...)
        dv[1, 1] += 100
        return dv
    end

    function TrixiParticles.interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
                                      system::System2, neighbor::System2, semi; kwargs...)
        dv[1, 1] += 1000
        return dv
    end

    @testset verbose=true "Constructor" begin
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        # Verification
        @test semi.ranges_u == (1:6, 7:15)
        @test semi.ranges_v == (1:6, 7:18)

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

    @testset verbose=true "Interaction Matrix" begin
        struct TestInteraction
            calls::Base.RefValue{Int}
        end
        TestInteraction() = TestInteraction(Ref(0))

        function (interaction::TestInteraction)(dv, v_system, u_system, v_neighbor,
                                                u_neighbor, system, neighbor, semi;
                                                kwargs...)
            interaction.calls[] += 1
            dv[1, 1] += 50
            return dv
        end

        function zero_ode_state(semi)
            n_u = sum(TrixiParticles.u_nvariables(system) *
                      TrixiParticles.n_integrated_particles(system)
                      for system in semi.systems)
            n_v = sum(TrixiParticles.v_nvariables(system) *
                      TrixiParticles.n_integrated_particles(system)
                      for system in semi.systems)

            v_ode = zeros(n_v)
            u_ode = zeros(n_u)
            dv_ode = zero(v_ode)

            return v_ode, u_ode, dv_ode
        end

        function initialized_ode_state(semi)
            v_ode, u_ode, dv_ode = zero_ode_state(semi)

            TrixiParticles.foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
                TrixiParticles.write_v0!(v, system)
                TrixiParticles.write_u0!(u, system)
            end

            return v_ode, u_ode, dv_ode
        end

        function system_dv(dv_ode, semi, system_index)
            system = semi.systems[system_index]
            return Array(TrixiParticles.wrap_v(dv_ode, system, semi))
        end

        function make_tlsph_fluid_systems()
            kernel = SchoenbergCubicSplineKernel{2}()
            smoothing_length = 1.0
            state_equation = StateEquationCole(sound_speed=10.0,
                                               reference_density=1000.0,
                                               exponent=7)

            structure_ic = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                            velocity=reshape([0.0, 0.0], 2, 1),
                                            density=[1000.0],
                                            particle_spacing=1.0)
            boundary_model = BoundaryModelDummyParticles(structure_ic.density,
                                                         structure_ic.mass,
                                                         SummationDensity(), kernel,
                                                         smoothing_length;
                                                         state_equation,
                                                         correction=nothing)
            structure = TotalLagrangianSPHSystem(structure_ic;
                                                 smoothing_kernel=kernel,
                                                 smoothing_length,
                                                 young_modulus=1.0,
                                                 poisson_ratio=0.3,
                                                 boundary_model)

            fluid_ic = InitialCondition(; coordinates=reshape([0.5, 0.0], 2, 1),
                                        velocity=reshape([0.0, 0.0], 2, 1),
                                        density=[1000.0],
                                        particle_spacing=1.0)
            fluid = WeaklyCompressibleSPHSystem(fluid_ic;
                                                density_calculator=ContinuityDensity(),
                                                state_equation,
                                                smoothing_kernel=kernel,
                                                smoothing_length)

            return structure, fluid
        end

        function make_shepard_fluid_system(x_coordinate)
            kernel = SchoenbergCubicSplineKernel{2}()
            smoothing_length = 1.0
            state_equation = StateEquationCole(sound_speed=10.0,
                                               reference_density=1000.0,
                                               exponent=7)
            initial_condition = InitialCondition(;
                                                 coordinates=reshape([x_coordinate, 0.0],
                                                                     2, 1),
                                                 velocity=reshape([0.0, 0.0], 2, 1),
                                                 density=[1000.0],
                                                 particle_spacing=1.0)
            system = WeaklyCompressibleSPHSystem(initial_condition;
                                                 density_calculator=SummationDensity(),
                                                 correction=ShepardKernelCorrection(),
                                                 state_equation,
                                                 smoothing_kernel=kernel,
                                                 smoothing_length)
            system.cache.density .= initial_condition.density

            return system
        end

        function shepard_correction_coefficient(systems, interaction_matrix)
            semi = Semidiscretization(systems...; neighborhood_search=nothing,
                                      interaction_matrix)
            v_ode, u_ode, _ = initialized_ode_state(semi)
            system = semi.systems[1]
            u = TrixiParticles.wrap_u(u_ode, system, semi)

            TrixiParticles.compute_correction_values!(system,
                                                      TrixiParticles.system_correction(system),
                                                      u, v_ode, u_ode, semi)

            return copy(system.cache.kernel_correction_coefficient), semi
        end

        @testset "constructor" begin
            semi = Semidiscretization(system1, system2, neighborhood_search=nothing)
            @test semi.interaction_matrix == trues(2, 2)

            @test_throws ArgumentError Semidiscretization(system1, system2;
                                                          neighborhood_search=nothing,
                                                          interaction_matrix=trues(1, 1))

            @test_throws ArgumentError Semidiscretization(system1, system2;
                                                          neighborhood_search=nothing,
                                                          interaction_matrix=Any[true 1;
                                                                                 false true])

            semi_any_bool = Semidiscretization(system1, system2;
                                               neighborhood_search=nothing,
                                               interaction_matrix=Any[true false;
                                                                      false true])
            @test semi_any_bool.interaction_matrix isa Matrix{Bool}
            @test semi_any_bool.interaction_matrix == [true false; false true]

            matrix_parent = trues(3, 3)
            interaction_matrix_view = @view matrix_parent[1:2, 1:2]
            semi_matrix_view = Semidiscretization(system1, system2;
                                                  neighborhood_search=nothing,
                                                  interaction_matrix=interaction_matrix_view)
            matrix_parent[1, 2] = false
            @test semi_matrix_view.interaction_matrix isa Matrix{Bool}
            @test axes(semi_matrix_view.interaction_matrix) == (Base.OneTo(2), Base.OneTo(2))
            @test semi_matrix_view.interaction_matrix == trues(2, 2)

            abstract_union_matrix = Matrix{Union{Bool, Function}}(trues(2, 2))
            semi_abstract_union = Semidiscretization(system1, system2;
                                                     neighborhood_search=nothing,
                                                     interaction_matrix=abstract_union_matrix)
            @test semi_abstract_union.interaction_matrix isa Matrix{Bool}
            @test semi_abstract_union.interaction_matrix == trues(2, 2)

            interaction = TestInteraction()
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(trues(2, 2))
            interaction_matrix[1, 2] = interaction
            semi_custom = Semidiscretization(system1, system2;
                                             neighborhood_search=nothing,
                                             interaction_matrix)
            interaction_matrix[1, 2] = false

            @test semi_custom.interaction_matrix[1, 2] === interaction

            interaction_any = TestInteraction()
            interaction_matrix_any = Any[true interaction_any;
                                         false true]
            semi_any_custom = Semidiscretization(system1, system2;
                                                 neighborhood_search=nothing,
                                                 interaction_matrix=interaction_matrix_any)
            interaction_matrix_any[1, 2] = false

            @test eltype(semi_any_custom.interaction_matrix) ==
                  Union{Bool, typeof(interaction_any)}
            @test semi_any_custom.interaction_matrix[1, 2] === interaction_any
        end

        @testset "disabled pairs skip ordered RHS dispatch" begin
            interaction_matrix = Bool[true false
                                      true true]
            semi = Semidiscretization(system1, system2; neighborhood_search=nothing,
                                      interaction_matrix)
            v_ode, u_ode, dv_ode = zero_ode_state(semi)

            TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi)

            @test !TrixiParticles.has_system_interaction(semi.systems[1],
                                                         semi.systems[2], semi)
            @test TrixiParticles.has_system_interaction(semi.systems[2],
                                                        semi.systems[1], semi)
            @test system_dv(dv_ode, semi, 1)[1, 1] == 1
            @test system_dv(dv_ode, semi, 2)[1, 1] == 1100
        end

        @testset "callable entries replace ordered RHS dispatch" begin
            interaction = TestInteraction()
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(trues(2, 2))
            interaction_matrix[1, 2] = interaction
            interaction_matrix[2, 1] = false

            semi = Semidiscretization(system1, system2; neighborhood_search=nothing,
                                      interaction_matrix)
            v_ode, u_ode, dv_ode = zero_ode_state(semi)

            TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi)

            @test interaction.calls[] == 1
            @test semi.interaction_matrix[1, 2] === interaction
            @test system_dv(dv_ode, semi, 1)[1, 1] == 51
            @test system_dv(dv_ode, semi, 2)[1, 1] == 1000
        end

        @testset "disabled pairs skip correction values" begin
            all_enabled_coefficient,
            _ = shepard_correction_coefficient((make_shepard_fluid_system(0.0),
                                                make_shepard_fluid_system(0.5)),
                                               trues(2, 2))

            filtered_matrix = Bool[true false
                                   true true]
            filtered_coefficient,
            semi_filtered = shepard_correction_coefficient((make_shepard_fluid_system(0.0),
                                                            make_shepard_fluid_system(0.5)),
                                                           filtered_matrix)

            reference_coefficient,
            _ = shepard_correction_coefficient((make_shepard_fluid_system(0.0),),
                                               trues(1, 1))

            @test !TrixiParticles.has_system_interaction(semi_filtered.systems[1],
                                                         semi_filtered.systems[2],
                                                         semi_filtered)
            @test only(all_enabled_coefficient) > only(filtered_coefficient)
            @test isapprox(only(filtered_coefficient), only(reference_coefficient))
        end

        @testset "split interaction uses original interaction matrix" begin
            structure, fluid = make_tlsph_fluid_systems()
            interaction = TestInteraction()
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(trues(2, 2))
            interaction_matrix[1, 2] = interaction

            semi = Semidiscretization(structure, fluid; neighborhood_search=nothing,
                                      interaction_matrix)
            semi_split = Semidiscretization(semi.systems[1]; neighborhood_search=nothing)
            v_ode, u_ode, _ = initialized_ode_state(semi)
            _, _, dv_ode_split = initialized_ode_state(semi_split)

            semi.integrate_tlsph[] = false
            TrixiParticles.system_interaction!(zero(v_ode), v_ode, u_ode, semi)
            @test interaction.calls[] == 0

            TrixiParticles.other_interaction_split!(dv_ode_split, semi, v_ode, u_ode,
                                                    semi_split)

            @test interaction.calls[] == 1
            @test system_dv(dv_ode_split, semi_split, 1)[1, 1] == 50
        end

        @testset "split interaction separates TLSPH self and cross interactions" begin
            structure1 = first(make_tlsph_fluid_systems())
            structure2 = first(make_tlsph_fluid_systems())
            self_interaction = TestInteraction()
            cross_interaction = TestInteraction()
            interaction_matrix = Matrix{Union{Bool, typeof(self_interaction)}}(falses(2, 2))
            interaction_matrix[1, 1] = self_interaction
            interaction_matrix[1, 2] = cross_interaction

            semi = Semidiscretization(structure1, structure2; neighborhood_search=nothing,
                                      interaction_matrix)
            semi_split = Semidiscretization(semi.systems...; neighborhood_search=nothing)
            v_ode, u_ode, _ = initialized_ode_state(semi)
            v_ode_split, u_ode_split, dv_ode_split = initialized_ode_state(semi_split)

            semi.integrate_tlsph[] = false
            TrixiParticles.self_interaction_split!(dv_ode_split, v_ode_split,
                                                   u_ode_split, semi_split, semi)

            @test self_interaction.calls[] == 1
            @test cross_interaction.calls[] == 0
            @test system_dv(dv_ode_split, semi_split, 1)[1, 1] == 50
            @test system_dv(dv_ode_split, semi_split, 2)[1, 1] == 0

            fill!(dv_ode_split, 0)
            TrixiParticles.other_interaction_split!(dv_ode_split, semi, v_ode, u_ode,
                                                    semi_split)

            @test cross_interaction.calls[] == 1
            @test system_dv(dv_ode_split, semi_split, 1)[1, 1] == 50
            @test system_dv(dv_ode_split, semi_split, 2)[1, 1] == 0
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

        struct ShowInteraction end
        (::ShowInteraction)(args...; kwargs...) = nothing

        interaction = ShowInteraction()
        interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(trues(2, 2))
        interaction_matrix[1, 2] = false
        interaction_matrix[2, 1] = interaction
        semi_custom = Semidiscretization(system1, system2; neighborhood_search=nothing,
                                         interaction_matrix)

        show_custom_compact = "Semidiscretization($System1(), $System2(), " *
                              "neighborhood_search=TrivialNeighborhoodSearch, " *
                              "interaction_matrix=1 disabled, 1 custom)"
        @test repr(semi_custom) == show_custom_compact

        show_custom_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ Semidiscretization                                                                               │
        │ ══════════════════                                                                               │
        │ #spatial dimensions: ………………………… 3                                                                │
        │ #systems: ……………………………………………………… 2                                                                │
        │ neighborhood search: ………………………… TrivialNeighborhoodSearch                                        │
        │ total #particles: ………………………………… 5                                                                │
        │ eltype: …………………………………………………………… Float64                                                          │
        │ coordinates eltype: …………………………… Float32                                                          │
        │ interaction matrix: …………………………… 1 disabled, 1 custom                                             │
        │ disabled pairs: ……………………………………… 1 -> 2                                                           │
        │ custom pairs: …………………………………………… 2 -> 1 (ShowInteraction)                                         │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", semi_custom) == show_custom_box
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
