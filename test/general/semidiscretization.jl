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
        @test_throws ArgumentError Semidiscretization(system1, system2;
                                                      neighborhood_search=nothing,
                                                      interaction_matrix=Any[true false;
                                                                             false true])
        abstract_union_matrix = Matrix{Union{Bool, Function}}(trues(2, 2))
        @test_throws ArgumentError Semidiscretization(system1, system2;
                                                      neighborhood_search=nothing,
                                                      interaction_matrix=abstract_union_matrix)

        closure_interaction = let calls = Ref(0)
            (dv, v_system, u_system, v_neighbor, u_neighbor, system, neighbor, semi;
             kwargs...) -> (calls[] += 1; dv)
        end
        closure_matrix = Matrix{Union{Bool, typeof(closure_interaction)}}(trues(2, 2))
        closure_matrix[1, 2] = closure_interaction
        semi_closure = Semidiscretization(system1, system2;
                                          neighborhood_search=nothing,
                                          interaction_matrix=closure_matrix)
        @test semi_closure.interaction_matrix[1, 2] === closure_interaction

        interaction_matrix = trues(2, 2)
        semi_copied_matrix = Semidiscretization(system1, system2;
                                                neighborhood_search=nothing,
                                                interaction_matrix)
        interaction_matrix[1, 2] = false
        @test semi_copied_matrix.interaction_matrix == trues(2, 2)
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

            p = (; semi, split_integration_data=nothing)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, p, 0.0)

            return dv_ode
        end

        function system_dv(dv_ode, semi, system_index)
            system = semi.systems[system_index]
            return Array(TrixiParticles.wrap_v(dv_ode, system, semi))
        end

        struct TestInterphaseDrag{T}
            coefficient::T
        end

        struct TestNoOpInteraction end

        struct TestCountingInteraction
            calls::Base.RefValue{Int}
            integrate_tlsph_seen::Base.RefValue{Bool}
        end
        TestCountingInteraction(calls) = TestCountingInteraction(calls, Ref(false))

        function (drag::TestInterphaseDrag)(dv, v_system, u_system, v_neighbor,
                                            u_neighbor, system, neighbor, semi; kwargs...)
            system_coords = TrixiParticles.current_coordinates(u_system, system)
            neighbor_coords = TrixiParticles.current_coordinates(u_neighbor, neighbor)

            TrixiParticles.foreach_point_neighbor(system, neighbor, system_coords,
                                                  neighbor_coords,
                                                  semi) do particle,
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

        function (::TestNoOpInteraction)(dv, v_system, u_system, v_neighbor,
                                         u_neighbor, system, neighbor, semi; kwargs...)
            return dv
        end

        function (interaction::TestCountingInteraction)(dv, v_system, u_system, v_neighbor,
                                                        u_neighbor, system, neighbor, semi;
                                                        integrate_tlsph=false, kwargs...)
            interaction.calls[] += 1
            interaction.integrate_tlsph_seen[] |= integrate_tlsph
            dv[1, 1] += 1

            return dv
        end

        function make_drag_semi(nhs_factory, systems...; drag=false)
            neighborhood_search = nhs_factory()

            if drag
                interaction = TestInterphaseDrag(500.0)
                interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(trues(2, 2))
                interaction_matrix[1, 2] = interaction
                interaction_matrix[2, 1] = interaction
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

        function make_moving_pressure_interpolation_systems()
            fluid_ic = make_particle(10.0, 1030.0, (0.0, 0.0))
            boundary_ic = make_particle(0.0, 1000.0, (0.0, 0.0))

            fluid = WeaklyCompressibleSPHSystem(fluid_ic;
                                                density_calculator=ContinuityDensity(),
                                                state_equation, smoothing_kernel=kernel,
                                                smoothing_length)

            boundary_model = BoundaryModelDummyParticles(boundary_ic.density,
                                                         boundary_ic.mass,
                                                         AdamiPressureExtrapolation(),
                                                         kernel, smoothing_length;
                                                         state_equation=state_equation,
                                                         correction=nothing)
            boundary = WallBoundarySystem(boundary_ic, boundary_model)

            return fluid, boundary
        end

        function make_summation_density_systems()
            fluid_a_ic = make_particle(0.0, 1000.0, (0.0, 0.0))
            fluid_b_ic = make_particle(0.2, 1000.0, (0.0, 0.0))

            fluid_a = WeaklyCompressibleSPHSystem(fluid_a_ic;
                                                  density_calculator=SummationDensity(),
                                                  state_equation, smoothing_kernel=kernel,
                                                  smoothing_length)
            fluid_b = WeaklyCompressibleSPHSystem(fluid_b_ic;
                                                  density_calculator=SummationDensity(),
                                                  state_equation, smoothing_kernel=kernel,
                                                  smoothing_length)

            return fluid_a, fluid_b
        end

        function make_unlinked_tlsph_fluid_systems()
            structure_ic = make_particle(0.0, 1000.0, (0.0, 0.0))
            structure = TotalLagrangianSPHSystem(structure_ic;
                                                 smoothing_kernel=kernel,
                                                 smoothing_length,
                                                 young_modulus=1.0,
                                                 poisson_ratio=0.3)

            fluid_ic = make_particle(0.5, 1030.0, (0.0, 0.0))
            fluid = WeaklyCompressibleSPHSystem(fluid_ic;
                                                density_calculator=ContinuityDensity(),
                                                state_equation,
                                                smoothing_kernel=kernel,
                                                smoothing_length)

            return structure, fluid
        end

        function make_tlsph_fluid_systems()
            structure_ic = make_particle(0.0, 1000.0, (0.0, 0.0))
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

            fluid_ic = make_particle(0.5, 1030.0, (0.0, 0.0))
            fluid = WeaklyCompressibleSPHSystem(fluid_ic;
                                                density_calculator=ContinuityDensity(),
                                                state_equation,
                                                smoothing_kernel=kernel,
                                                smoothing_length)

            return structure, fluid
        end

        function make_edac_average_pressure_system(x, pressure)
            ic = InitialCondition(; coordinates=reshape([x, 0.0], 2, 1),
                                  velocity=reshape([0.0, 0.0], 2, 1),
                                  density=[1000.0], pressure=[pressure],
                                  particle_spacing=1.0)

            return EntropicallyDampedSPHSystem(ic; smoothing_kernel=kernel,
                                               smoothing_length,
                                               sound_speed=10.0,
                                               average_pressure_reduction=true)
        end

        function updated_boundary_pressure(semi)
            v_ode, u_ode = create_ode_state(semi)

            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

            return copy(last(semi.systems).boundary_model.pressure)
        end

        function updated_boundary_pressure_after_moving_fluid(semi)
            v_ode, u_ode = create_ode_state(semi)
            TrixiParticles.initialize_neighborhood_searches!(semi)

            fluid = first(semi.systems)
            u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
            u_fluid[:, 1] .= (0.1, 0.0)

            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

            return copy(last(semi.systems).boundary_model.pressure)
        end

        function updated_system_density(semi, system_index)
            v_ode, u_ode = create_ode_state(semi)

            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

            system = semi.systems[system_index]
            v = TrixiParticles.wrap_v(v_ode, system, semi)

            return TrixiParticles.current_density(v, system, 1)
        end

        function updated_average_pressure(semi, system_index)
            v_ode, u_ode = create_ode_state(semi)
            TrixiParticles.initialize_neighborhood_searches!(semi)
            TrixiParticles.update_nhs!(semi, u_ode)

            system = semi.systems[system_index]
            TrixiParticles.update_average_pressure!(system,
                                                    system.average_pressure_reduction,
                                                    v_ode, u_ode, semi)

            return system.cache.pressure_average[1], system.cache.neighbor_counter[1]
        end

        @testset "disabled pairs skip ordered RHS dispatch" begin
            interaction_matrix = Bool[true false
                                      true true]
            semi = Semidiscretization(system1, system2; neighborhood_search=nothing,
                                      interaction_matrix)

            v_ode = zeros(sum(TrixiParticles.v_nvariables(system) *
                              TrixiParticles.n_integrated_particles(system)
                              for system in semi.systems))
            u_ode = zeros(sum(TrixiParticles.u_nvariables(system) *
                              TrixiParticles.n_integrated_particles(system)
                              for system in semi.systems))
            dv_ode = zero(v_ode)

            TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi)

            @test system_dv(dv_ode, semi, 1)[1, 1] == 1
            @test system_dv(dv_ode, semi, 2)[1, 1] == 1100
        end

        @testset "generic neighbor iteration ignores interaction matrix" begin
            fluid_a, fluid_b = make_summation_density_systems()
            interaction_matrix = Bool[true false
                                      false true]
            semi = Semidiscretization(fluid_a, fluid_b; neighborhood_search=nothing,
                                      interaction_matrix)
            v_ode, u_ode = create_ode_state(semi)

            system, neighbor = semi.systems
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            u_neighbor = TrixiParticles.wrap_u(u_ode, neighbor, semi)
            system_coords = TrixiParticles.current_coordinates(u, system)
            neighbor_coords = TrixiParticles.current_coordinates(u_neighbor, neighbor)
            counter = Ref(0)

            TrixiParticles.foreach_point_neighbor(system, neighbor, system_coords,
                                                  neighbor_coords, semi) do particle,
                                                                              neighbor_particle,
                                                                              pos_diff,
                                                                              distance
                counter[] += 1
            end

            @test counter[] == 1
        end

        @testset "dummy boundary pressure ignores interaction matrix" begin
            semi_filtered = let
                fluid_a, fluid_b, boundary = make_pressure_interpolation_systems()
                interaction_matrix = trues(3, 3)
                interaction_matrix[3, 2] = false

                Semidiscretization(fluid_a, fluid_b, boundary;
                                   neighborhood_search=nothing, interaction_matrix)
            end

            semi_full = let
                fluid_a, fluid_b, boundary = make_pressure_interpolation_systems()
                Semidiscretization(fluid_a, fluid_b, boundary; neighborhood_search=nothing)
            end

            pressure_filtered = updated_boundary_pressure(semi_filtered)
            pressure_full = updated_boundary_pressure(semi_full)

            @test pressure_filtered ≈ pressure_full
        end

        @testset "asymmetric boundary pressure updates reverse neighborhood search" begin
            semi_filtered = let
                fluid, boundary = make_moving_pressure_interpolation_systems()
                interaction_matrix = trues(2, 2)
                interaction_matrix[1, 2] = false

                Semidiscretization(fluid, boundary;
                                   neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()),
                                   interaction_matrix)
            end

            semi_full = let
                fluid, boundary = make_moving_pressure_interpolation_systems()
                Semidiscretization(fluid, boundary;
                                   neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()))
            end

            pressure_filtered = updated_boundary_pressure_after_moving_fluid(semi_filtered)
            pressure_full = updated_boundary_pressure_after_moving_fluid(semi_full)

            @test only(pressure_filtered) > 0
            @test pressure_filtered ≈ pressure_full
        end

        @testset "summation density ignores interaction matrix" begin
            semi_filtered = let
                fluid_a, fluid_b = make_summation_density_systems()
                interaction_matrix = trues(2, 2)
                interaction_matrix[1, 2] = false

                Semidiscretization(fluid_a, fluid_b; neighborhood_search=nothing,
                                   interaction_matrix)
            end

            semi_full = let
                fluid_a, fluid_b = make_summation_density_systems()
                Semidiscretization(fluid_a, fluid_b; neighborhood_search=nothing)
            end

            density_filtered = updated_system_density(semi_filtered, 1)
            density_full = updated_system_density(semi_full, 1)

            @test density_filtered ≈ density_full
        end

        @testset "average pressure reduction ignores interaction matrix" begin
            semi_filtered = let
                system_a = make_edac_average_pressure_system(0.0, 10.0)
                system_b = make_edac_average_pressure_system(0.25, 100.0)
                interaction_matrix = trues(2, 2)
                interaction_matrix[1, 2] = false

                Semidiscretization(system_a, system_b; neighborhood_search=nothing,
                                   interaction_matrix)
            end

            semi_full = let
                system_a = make_edac_average_pressure_system(0.0, 10.0)
                system_b = make_edac_average_pressure_system(0.25, 100.0)
                Semidiscretization(system_a, system_b; neighborhood_search=nothing)
            end

            pressure_filtered,
            neighbor_count_filtered = updated_average_pressure(semi_filtered,
                                                               1)
            pressure_full, neighbor_count_full = updated_average_pressure(semi_full, 1)

            @test neighbor_count_filtered == neighbor_count_full
            @test pressure_filtered ≈ pressure_full
            @test neighbor_count_full == 2
            @test pressure_full ≈ 55.0
        end

        @testset "disabled pairs do not bypass normal configuration" begin
            structure, fluid = make_unlinked_tlsph_fluid_systems()
            interaction_matrix = Bool[true false
                                      false true]

            @test_throws ArgumentError Semidiscretization(structure, fluid;
                                                          neighborhood_search=nothing,
                                                          interaction_matrix)

            interaction = TestNoOpInteraction()
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(2, 2))
            interaction_matrix[2, 1] = interaction
            @test_throws ArgumentError Semidiscretization(structure, fluid;
                                                          neighborhood_search=nothing,
                                                          interaction_matrix)
        end

        @testset "custom TLSPH interactions use split-aware dispatch paths" begin
            structure, _ = make_unlinked_tlsph_fluid_systems()
            calls = Ref(0)
            integrate_tlsph_seen = Ref(false)
            interaction = TestCountingInteraction(calls, integrate_tlsph_seen)
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(trues(1, 1))
            interaction_matrix[1, 1] = interaction

            semi = Semidiscretization(structure; neighborhood_search=nothing,
                                      interaction_matrix)
            v_ode, u_ode = create_ode_state(semi)
            dv_ode = zero(v_ode)

            semi.integrate_tlsph[] = false
            TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi)
            @test calls[] == 0
            @test !integrate_tlsph_seen[]
            @test iszero(dv_ode)

            TrixiParticles.self_interaction_split!(dv_ode, v_ode, u_ode, semi, semi)
            @test calls[] == 1
            @test integrate_tlsph_seen[]
            @test system_dv(dv_ode, semi, 1)[1, 1] == 1

            work = Ref(0.0)
            dv = zeros(TrixiParticles.v_nvariables(semi.systems[1]),
                       TrixiParticles.nparticles(semi.systems[1]))
            TrixiParticles.update_mechanical_work_calculator!(work, semi.systems[1],
                                                              1:1, false, dv, v_ode,
                                                              u_ode, semi, 0.0, 1.0)
            @test calls[] == 2
        end

        @testset "custom TLSPH-fluid interaction is called by split callback" begin
            structure, fluid = make_tlsph_fluid_systems()
            calls = Ref(0)
            integrate_tlsph_seen = Ref(false)
            interaction = TestCountingInteraction(calls, integrate_tlsph_seen)
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(trues(2, 2))
            interaction_matrix[1, 2] = interaction

            semi = Semidiscretization(structure, fluid;
                                      neighborhood_search=nothing,
                                      interaction_matrix)
            ode = semidiscretize(semi, (0.0, 1.0e-4))

            split_integration = SplitIntegrationCallback(SymplecticPositionVerlet();
                                                         dt=1.0e-5)
            sol = TrixiParticles.SciMLBase.solve(ode, SymplecticPositionVerlet();
                                                 dt=1.0e-4, save_everystep=false,
                                                 maxiters=10, callback=split_integration)

            @test sol.retcode == TrixiParticles.SciMLBase.ReturnCode.Success
            @test calls[] > 0
            @test integrate_tlsph_seen[]
        end

        @testset verbose=true "callable interphase drag" begin
            for (test_name, nhs_factory) in
                (("without neighborhood search", () -> nothing),
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
