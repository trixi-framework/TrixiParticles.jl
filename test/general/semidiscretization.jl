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
        using OrdinaryDiffEqCore: ReturnCode, solve
        using OrdinaryDiffEqLowStorageRK: RDPK3SpFSAL35

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

        struct TestDefaultWrapperInteraction
            integrate_tlsph_seen::Base.RefValue{Bool}
        end

        struct TestSplitSuppressedInteraction
            called::Base.RefValue{Bool}
        end

        struct TestSelfNeighborCountingInteraction
            neighbor_count::Base.RefValue{Int}
        end

        struct TestNoOpNHSSystem <: TrixiParticles.AbstractFluidSystem{2}
            initial_condition::InitialCondition
            surface_tension::Nothing
            surface_normal_method::Nothing
        end

        struct TestNoOpInteraction end

        Base.eltype(system::TestNoOpNHSSystem) = eltype(system.initial_condition)
        TrixiParticles.nparticles(system::TestNoOpNHSSystem) = 1
        TrixiParticles.n_integrated_particles(system::TestNoOpNHSSystem) = 1
        TrixiParticles.u_nvariables(::TestNoOpNHSSystem) = 2
        TrixiParticles.v_nvariables(::TestNoOpNHSSystem) = 0
        TrixiParticles.compact_support(::TestNoOpNHSSystem,
                                       ::TestNoOpNHSSystem) = 0.0
        TrixiParticles.compact_support(::TestNoOpInteraction,
                                       ::TestNoOpNHSSystem,
                                       ::TestNoOpNHSSystem) = 0.75
        TrixiParticles.compact_support(::TestNoOpInteraction,
                                       ::TrixiParticles.AbstractFluidSystem,
                                       ::TotalLagrangianSPHSystem) = 0.0

        function TrixiParticles.write_u0!(u0, system::TestNoOpNHSSystem)
            u0[:, :] .= system.initial_condition.coordinates
            return u0
        end

        TrixiParticles.write_v0!(v0, ::TestNoOpNHSSystem) = v0

        function TrixiParticles.update_nhs!(neighborhood_search,
                                            ::TestNoOpNHSSystem,
                                            ::TestNoOpNHSSystem,
                                            u_system, u_neighbor, semi)
            return neighborhood_search
        end

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

        function (interaction::TestKeywordInteraction)(dv, v_system, u_system,
                                                       v_neighbor, u_neighbor,
                                                       system, neighbor, semi;
                                                       integrate_tlsph=false)
            interaction.integrate_tlsph_seen[] = integrate_tlsph
            return dv
        end

        function (interaction::TestDefaultWrapperInteraction)(dv, v_system, u_system,
                                                              v_neighbor, u_neighbor,
                                                              system, neighbor, semi;
                                                              integrate_tlsph=false,
                                                              kwargs...)
            interaction.integrate_tlsph_seen[] = integrate_tlsph
            return TrixiParticles.interact!(dv, v_system, u_system, v_neighbor,
                                            u_neighbor, system, neighbor, semi;
                                            integrate_tlsph, kwargs...)
        end

        function (interaction::TestSplitSuppressedInteraction)(dv, v_system, u_system,
                                                               v_neighbor, u_neighbor,
                                                               system, neighbor, semi;
                                                               kwargs...)
            interaction.called[] = true
            dv[1, 1] += 1
            return dv
        end

        function (::TestNoOpInteraction)(dv, v_system, u_system, v_neighbor,
                                         u_neighbor, system, neighbor, semi; kwargs...)
            return dv
        end

        function (interaction::TestSelfNeighborCountingInteraction)(dv, v_system, u_system,
                                                                    v_neighbor, u_neighbor,
                                                                    system, neighbor, semi;
                                                                    kwargs...)
            interaction.neighbor_count[] = 0
            system_coords = TrixiParticles.current_coordinates(u_system, system)
            neighbor_coords = TrixiParticles.current_coordinates(u_neighbor, neighbor)

            TrixiParticles.foreach_point_neighbor(system, neighbor, system_coords,
                                                  neighbor_coords, semi;
                                                  parallelization_backend=SerialBackend()) do particle,
                                                                                                neighbor_particle,
                                                                                                pos_diff,
                                                                                                distance
                system === neighbor && particle == neighbor_particle && return
                interaction.neighbor_count[] += 1
            end

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

        function make_tlsph_fluid_systems()
            structure_ic = make_particle(0.0, 1000.0, (0.0, 0.0))
            boundary_model = BoundaryModelDummyParticles(structure_ic.density,
                                                         structure_ic.mass,
                                                         SummationDensity(), kernel,
                                                         smoothing_length;
                                                         state_equation)
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

        function make_tlsph_single_particle_system(x)
            initial_condition = make_particle(x, 1000.0, (0.0, 0.0))

            return TotalLagrangianSPHSystem(initial_condition;
                                            smoothing_kernel=kernel,
                                            smoothing_length,
                                            young_modulus=1.0,
                                            poisson_ratio=0.3)
        end

        function make_tlsph_pair_system()
            coordinates = [0.0 0.25
                           0.0 0.0]
            velocity = zeros(2, 2)
            density = fill(1000.0, 2)
            initial_condition = InitialCondition(; coordinates, velocity, density,
                                                 particle_spacing=0.25)

            return TotalLagrangianSPHSystem(initial_condition;
                                            smoothing_kernel=kernel,
                                            smoothing_length,
                                            young_modulus=1.0,
                                            poisson_ratio=0.3)
        end

        function make_tlsph_far_pair_system()
            coordinates = [0.0 10.0
                           0.0 0.0]
            velocity = zeros(2, 2)
            density = fill(1000.0, 2)
            initial_condition = InitialCondition(; coordinates, velocity, density,
                                                 particle_spacing=0.25)

            return TotalLagrangianSPHSystem(initial_condition;
                                            smoothing_kernel=kernel,
                                            smoothing_length,
                                            young_modulus=1.0,
                                            poisson_ratio=0.3)
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

        function count_neighbors_after_moving_second_system(semi)
            _, u_ode = create_ode_state(semi)
            TrixiParticles.initialize_neighborhood_searches!(semi)

            first_system, second_system = semi.systems
            u_first = TrixiParticles.wrap_u(u_ode, first_system, semi)
            u_second = TrixiParticles.wrap_u(u_ode, second_system, semi)
            u_second[:, 1] .= (0.25, 0.0)

            TrixiParticles.update_nhs!(semi, u_ode)

            count = Ref(0)
            TrixiParticles.foreach_point_neighbor(first_system, second_system,
                                                  TrixiParticles.current_coordinates(u_first,
                                                                                     first_system),
                                                  TrixiParticles.current_coordinates(u_second,
                                                                                     second_system),
                                                  semi) do particle, neighbor,
                                                           pos_diff, distance
                count[] += 1
            end

            return count[]
        end

        function count_tlsph_initial_self_neighbors(system)
            coordinates = TrixiParticles.initial_coordinates(system)
            count = Ref(0)

            for particle in TrixiParticles.each_integrated_particle(system)
                TrixiParticles.foreach_neighbor(coordinates, coordinates,
                                                system.self_interaction_nhs,
                                                SerialBackend(), particle) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
                    count[] += 1
                end
            end

            return count[]
        end

        function count_tlsph_current_self_neighbors(semi, system, u)
            coordinates = TrixiParticles.current_coordinates(u, system)
            count = Ref(0)

            TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                                  semi) do particle, neighbor, pos_diff,
                                                           distance
                count[] += 1
            end

            return count[]
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

        @testset "summation density respects interaction matrix" begin
            semi_filtered = let
                fluid_a, fluid_b = make_summation_density_systems()
                interaction_matrix = trues(2, 2)
                interaction_matrix[1, 2] = false

                Semidiscretization(fluid_a, fluid_b; neighborhood_search=nothing,
                                   interaction_matrix)
            end

            semi_reduced = let
                fluid_a, _ = make_summation_density_systems()
                Semidiscretization(fluid_a; neighborhood_search=nothing)
            end

            semi_full = let
                fluid_a, fluid_b = make_summation_density_systems()
                Semidiscretization(fluid_a, fluid_b; neighborhood_search=nothing)
            end

            density_filtered = updated_system_density(semi_filtered, 1)
            density_reduced = updated_system_density(semi_reduced, 1)
            density_full = updated_system_density(semi_full, 1)

            @test density_filtered ≈ density_reduced
            @test !isapprox(density_filtered, density_full)
        end

        @testset "average pressure reduction respects ordered interaction matrix" begin
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

            @test neighbor_count_filtered == 1
            @test pressure_filtered ≈ 10.0
            @test neighbor_count_full == 2
            @test pressure_full ≈ 55.0
        end

        @testset "callable interaction updates otherwise unused neighborhood search" begin
            first_ic = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                        density=[1000.0], particle_spacing=1.0)
            second_ic = InitialCondition(; coordinates=reshape([10.0, 0.0], 2, 1),
                                         density=[1000.0], particle_spacing=1.0)

            first_system = TestNoOpNHSSystem(first_ic, nothing, nothing)
            second_system = TestNoOpNHSSystem(second_ic, nothing, nothing)
            interaction = TestNoOpInteraction()
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(2, 2))
            interaction_matrix[1, 2] = interaction

            semi = Semidiscretization(first_system, second_system;
                                      neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()),
                                      interaction_matrix)

            @test count_neighbors_after_moving_second_system(semi) == 1
        end

        @testset "pair neighborhood search uses largest enabled support" begin
            first_ic = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                        density=[1000.0], particle_spacing=1.0)
            second_ic = InitialCondition(; coordinates=reshape([10.0, 0.0], 2, 1),
                                         density=[1000.0], particle_spacing=1.0)

            first_system = TestNoOpNHSSystem(first_ic, nothing, nothing)
            second_system = TestNoOpNHSSystem(second_ic, nothing, nothing)
            interaction = TestNoOpInteraction()
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(trues(2, 2))
            interaction_matrix[2, 1] = interaction

            semi = Semidiscretization(first_system, second_system;
                                      neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()),
                                      interaction_matrix)

            search = TrixiParticles.get_neighborhood_search(first_system, second_system,
                                                            semi)
            @test TrixiParticles.PointNeighbors.search_radius(search) ≈ 0.75
        end

        @testset "disabled pairs skip configuration and neighborhood setup" begin
            structure, fluid = make_unlinked_tlsph_fluid_systems()
            interaction_matrix = Bool[true false
                                      false true]

            semi = Semidiscretization(structure, fluid; neighborhood_search=nothing,
                                      interaction_matrix)
            @test length(semi.systems) == 2

            interaction_matrix[2, 1] = true
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

        @testset "custom tlsph self interaction preserves initial self search" begin
            structure = make_tlsph_pair_system()
            interaction = TestNoOpInteraction()
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(1, 1))
            interaction_matrix[1, 1] = interaction

            semi = Semidiscretization(structure;
                                      neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()),
                                      interaction_matrix)
            system = only(semi.systems)
            initial_neighbor_count = count_tlsph_initial_self_neighbors(system)
            @test initial_neighbor_count > 2

            TrixiParticles.initialize_neighborhood_searches!(semi)
            _, u_ode = create_ode_state(semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            system.current_coordinates[:, 2] .= (10.0, 0.0)

            TrixiParticles.update_nhs!(semi, u_ode)

            @test count_tlsph_initial_self_neighbors(system) == initial_neighbor_count
            @test count_tlsph_current_self_neighbors(semi, system, u) <
                  initial_neighbor_count
        end

        @testset "large integrator suppresses split custom structure interaction" begin
            structure, fluid = make_tlsph_fluid_systems()
            interaction = TestSplitSuppressedInteraction(Ref(false))
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(2, 2))
            interaction_matrix[1, 2] = interaction

            semi = Semidiscretization(structure, fluid; neighborhood_search=nothing,
                                      interaction_matrix)
            semi.integrate_tlsph[] = false

            dv_ode = kick_once(semi)

            @test !interaction.called[]
            @test iszero(maximum(abs, system_dv(dv_ode, semi, 1)))
        end

        @testset "split custom tlsph self interaction updates current self search" begin
            structure = make_tlsph_far_pair_system()
            interaction = TestSelfNeighborCountingInteraction(Ref(0))
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(1, 1))
            interaction_matrix[1, 1] = interaction

            semi = Semidiscretization(structure;
                                      neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()),
                                      interaction_matrix)
            ode = semidiscretize(semi, (0.0, 1.0e-3))
            semi = ode.p.semi
            system = only(semi.systems)
            semi_split = Semidiscretization(system; neighborhood_search=nothing,
                                            parallelization_backend=semi.parallelization_backend)
            v_ode_split, u_ode_split = map(Array, ode.u0.x)
            u_split = TrixiParticles.wrap_u(u_ode_split, system, semi_split)
            u_split[:, 2] .= (0.25, 0.0)

            TrixiParticles.update_systems_split!(semi_split, semi, v_ode_split,
                                                 u_ode_split, 0.0)
            dv_ode_split = zero(v_ode_split)
            TrixiParticles.self_interaction_split!(dv_ode_split, v_ode_split,
                                                   u_ode_split, semi_split, semi)

            @test interaction.neighbor_count[] > 0
        end

        @testset "split custom tlsph cross interaction is not frozen" begin
            structure_a = make_tlsph_single_particle_system(0.0)
            structure_b = make_tlsph_single_particle_system(0.5)
            interaction = TestSplitSuppressedInteraction(Ref(false))
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(2, 2))
            interaction_matrix[1, 2] = interaction

            semi = Semidiscretization(structure_a, structure_b;
                                      neighborhood_search=nothing,
                                      interaction_matrix)
            ode = semidiscretize(semi, (0.0, 1.0e-3))
            semi = ode.p.semi
            semi_split = Semidiscretization(semi.systems...; neighborhood_search=nothing,
                                            parallelization_backend=semi.parallelization_backend)
            v_ode, u_ode = map(Array, ode.u0.x)
            dv_ode_split = zero(v_ode)

            TrixiParticles.other_interaction_split!(dv_ode_split, semi, v_ode, u_ode,
                                                    semi_split)
            @test !interaction.called[]

            TrixiParticles.self_interaction_split!(dv_ode_split, v_ode, u_ode,
                                                   semi_split, semi)
            @test interaction.called[]
        end

        @testset "split custom tlsph cross interaction updates current search" begin
            structure_a = make_tlsph_single_particle_system(0.0)
            structure_b = make_tlsph_single_particle_system(10.0)
            interaction = TestSelfNeighborCountingInteraction(Ref(0))
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(2, 2))
            interaction_matrix[1, 2] = interaction

            semi = Semidiscretization(structure_a, structure_b;
                                      neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()),
                                      interaction_matrix)
            ode = semidiscretize(semi, (0.0, 1.0e-3))
            semi = ode.p.semi
            semi_split = Semidiscretization(semi.systems...; neighborhood_search=nothing,
                                            parallelization_backend=semi.parallelization_backend)
            v_ode_split, u_ode_split = map(Array, ode.u0.x)
            u_b = TrixiParticles.wrap_u(u_ode_split, semi.systems[2], semi_split)
            u_b[:, 1] .= (0.25, 0.0)

            TrixiParticles.update_systems_split!(semi_split, semi, v_ode_split,
                                                 u_ode_split, 0.0)
            dv_ode_split = zero(v_ode_split)
            TrixiParticles.self_interaction_split!(dv_ode_split, v_ode_split,
                                                   u_ode_split, semi_split, semi)

            @test interaction.neighbor_count[] > 0
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

        @testset "split custom interaction receives keyword arguments through callback" begin
            integrate_tlsph_seen = Ref(false)
            structure, fluid = make_tlsph_fluid_systems()
            interaction = TestKeywordInteraction(integrate_tlsph_seen)
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(2, 2))
            interaction_matrix[1, 2] = interaction

            semi = Semidiscretization(structure, fluid; neighborhood_search=nothing,
                                      interaction_matrix)
            ode = semidiscretize(semi, (0.0, 1.0e-3))
            split_integration = SplitIntegrationCallback(RDPK3SpFSAL35(); dt=1.0e-4)
            sol = solve(ode, RDPK3SpFSAL35(); dt=1.0e-3, adaptive=false,
                        save_everystep=false, callback=split_integration)

            @test sol.retcode == ReturnCode.Success
            @test integrate_tlsph_seen[]
        end

        @testset "split wrapped default interaction forwards keyword arguments" begin
            integrate_tlsph_seen = Ref(false)
            structure, fluid = make_tlsph_fluid_systems()
            interaction = TestDefaultWrapperInteraction(integrate_tlsph_seen)
            interaction_matrix = Matrix{Union{Bool, typeof(interaction)}}(falses(2, 2))
            interaction_matrix[1, 2] = interaction

            semi = Semidiscretization(structure, fluid; neighborhood_search=nothing,
                                      interaction_matrix)
            ode = semidiscretize(semi, (0.0, 1.0e-3))
            split_integration = SplitIntegrationCallback(RDPK3SpFSAL35(); dt=1.0e-4)
            sol = solve(ode, RDPK3SpFSAL35(); dt=1.0e-3, adaptive=false,
                        save_everystep=false, callback=split_integration)

            @test sol.retcode == ReturnCode.Success
            @test integrate_tlsph_seen[]

            v_ode, _ = sol.u[end].x
            v_structure = TrixiParticles.wrap_v(v_ode, first(semi.systems), semi)
            @test maximum(abs, Array(v_structure)) > 0
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
