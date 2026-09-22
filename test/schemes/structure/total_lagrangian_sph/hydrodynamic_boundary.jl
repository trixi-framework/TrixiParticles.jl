@testset "Hydrodynamic boundary particles" begin
    function coupled_rhs(fluid_kind, hydrodynamic_boundary_particles,
                         parallelization_backend)
        spacing = 0.1
        fluid_density = 1000.0
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = spacing
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=fluid_density,
                                           exponent=1.0)

        fluid_initial = InitialCondition(; coordinates=reshape([0.0, 0.08], 2, 1),
                                         velocity=reshape([0.0, -1.0], 2, 1),
                                         density=1100.0, pressure=10_000.0,
                                         particle_spacing=spacing)
        fluid = if fluid_kind === :wcsph
            WeaklyCompressibleSPHSystem(fluid_initial; smoothing_kernel,
                                        smoothing_length,
                                        density_calculator=ContinuityDensity(),
                                        state_equation)
        else
            EntropicallyDampedSPHSystem(fluid_initial; smoothing_kernel,
                                        smoothing_length,
                                        density_calculator=ContinuityDensity(),
                                        sound_speed=10.0)
        end

        structure_initial = InitialCondition(; coordinates=[0.0 0.06
                                                            0.0 0.0],
                                             density=1200.0,
                                             particle_spacing=spacing)
        hydrodynamic_mass = fill(fluid_density * spacing^2, 2)
        boundary_model = BoundaryModelDummyParticles(fill(fluid_density, 2),
                                                     hydrodynamic_mass,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel,
                                                     smoothing_length;
                                                     state_equation)
        structure = TotalLagrangianSPHSystem(structure_initial; smoothing_kernel,
                                             smoothing_length,
                                             young_modulus=0.0,
                                             poisson_ratio=0.0,
                                             boundary_model,
                                             hydrodynamic_boundary_particles)

        semi = Semidiscretization(fluid, structure; neighborhood_search=nothing,
                                  parallelization_backend)
        ode = semidiscretize(semi, (0.0, 0.01); reset_threads=false)
        fluid, structure = ode.p.semi.systems
        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

        dv_fluid = Array(TrixiParticles.wrap_v(dv_ode, fluid, ode.p.semi))
        dv_structure = Array(TrixiParticles.wrap_v(dv_ode, structure, ode.p.semi))
        fluid_force = Array(fluid.mass)[1] * dv_fluid[1:2, 1]
        structure_force = vec(sum(Array(structure.mass)' .* dv_structure[1:2, :]; dims=2))

        return (; dv_fluid, dv_structure, fluid_force, structure_force,
                pressure=Array(structure.boundary_model.pressure),
                mask=Array(structure.hydrodynamic_boundary))
    end

    @testset "$fluid_kind" for fluid_kind in (:wcsph, :edac)
        default_result = coupled_rhs(fluid_kind, nothing, SerialBackend())
        full_result = coupled_rhs(fluid_kind, 1:2, SerialBackend())
        subset_result = coupled_rhs(fluid_kind, [1], SerialBackend())
        empty_result = coupled_rhs(fluid_kind, Int[], SerialBackend())

        @test default_result.mask == [true, true]
        @test full_result.mask == [true, true]
        @test subset_result.mask == [true, false]
        @test empty_result.mask == [false, false]
        @test default_result.dv_fluid ≈ full_result.dv_fluid
        @test default_result.dv_structure ≈ full_result.dv_structure

        @test norm(subset_result.fluid_force) > eps()
        @test norm(subset_result.dv_structure[:, 1]) > eps()
        @test subset_result.dv_structure[:, 2] ≈ zeros(2)
        @test subset_result.pressure[2] == 0
        @test subset_result.fluid_force≈-subset_result.structure_force rtol=5e-13 atol=5e-13

        @test empty_result.dv_fluid[1:2, :] ≈ zeros(2, 1)
        @test empty_result.dv_structure ≈ zeros(2, 2)
        @test empty_result.pressure == zeros(2)

        kernel_result = coupled_rhs(fluid_kind, [1],
                                    TrixiParticles.KernelAbstractions.CPU())
        @test kernel_result.dv_fluid ≈ subset_result.dv_fluid
        @test kernel_result.dv_structure ≈ subset_result.dv_structure
        @test kernel_result.pressure ≈ subset_result.pressure
    end

    @testset "Density and pressure" begin
        function density_state(hydrodynamic_boundary_particles)
            spacing = 0.1
            density = 1000.0
            smoothing_kernel = WendlandC2Kernel{2}()
            smoothing_length = spacing
            state_equation = StateEquationCole(; sound_speed=10.0,
                                               reference_density=density,
                                               exponent=1.0)
            fluid_initial = InitialCondition(; coordinates=reshape([0.0, 0.08], 2, 1),
                                             density, particle_spacing=spacing)
            fluid = WeaklyCompressibleSPHSystem(fluid_initial; smoothing_kernel,
                                                smoothing_length,
                                                density_calculator=SummationDensity(),
                                                state_equation)

            structure_initial = InitialCondition(; coordinates=[0.0 0.06
                                                                0.0 0.0],
                                                 density=1200.0,
                                                 particle_spacing=spacing)
            boundary_model = BoundaryModelDummyParticles(fill(density, 2),
                                                         fill(density * spacing^2, 2),
                                                         SummationDensity(),
                                                         smoothing_kernel,
                                                         smoothing_length;
                                                         state_equation)
            structure = TotalLagrangianSPHSystem(structure_initial; smoothing_kernel,
                                                 smoothing_length,
                                                 young_modulus=0.0,
                                                 poisson_ratio=0.0,
                                                 boundary_model,
                                                 hydrodynamic_boundary_particles)
            semi = Semidiscretization(fluid, structure; neighborhood_search=nothing,
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01); reset_threads=false)
            fluid, structure = ode.p.semi.systems
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)
            v_structure = TrixiParticles.wrap_v(v_ode, structure, ode.p.semi)
            u_structure = TrixiParticles.wrap_u(u_ode, structure, ode.p.semi)
            TrixiParticles.compute_density!(structure.boundary_model,
                                            SummationDensity(), structure,
                                            v_structure, u_structure, v_ode, u_ode,
                                            ode.p.semi)
            TrixiParticles.compute_pressure!(structure.boundary_model,
                                             SummationDensity(), structure,
                                             v_structure, u_structure, v_ode, u_ode,
                                             ode.p.semi)

            return (; fluid_density=copy(fluid.cache.density),
                    structure_density=copy(structure.boundary_model.cache.density),
                    structure_pressure=copy(structure.boundary_model.pressure))
        end

        default_state = density_state(nothing)
        full_state = density_state(1:2)
        subset_state = density_state([1])
        empty_state = density_state(Int[])

        @test default_state == full_state
        @test default_state.fluid_density[1] > subset_state.fluid_density[1] >
              empty_state.fluid_density[1]
        @test subset_state.structure_density[1] > 0
        @test subset_state.structure_density[2] == 0
        @test subset_state.structure_pressure[2] == 0
        @test empty_state.structure_density == zeros(2)
        @test empty_state.structure_pressure == zeros(2)
    end

    @testset "EDAC pressure averaging" begin
        function pressure_average(hydrodynamic_boundary_particles)
            spacing = 0.1
            density = 1000.0
            smoothing_kernel = WendlandC2Kernel{2}()
            smoothing_length = spacing
            fluid_initial = InitialCondition(; coordinates=reshape([0.0, 0.08], 2, 1),
                                             density, pressure=6.0,
                                             particle_spacing=spacing)
            fluid = EntropicallyDampedSPHSystem(fluid_initial; smoothing_kernel,
                                                smoothing_length,
                                                density_calculator=ContinuityDensity(),
                                                sound_speed=10.0,
                                                average_pressure_reduction=true)

            structure_initial = InitialCondition(; coordinates=[0.0 0.06
                                                                0.0 0.0],
                                                 density=1200.0,
                                                 particle_spacing=spacing)
            boundary_model = BoundaryModelDummyParticles(fill(density, 2),
                                                         fill(density * spacing^2, 2),
                                                         AdamiPressureExtrapolation(),
                                                         smoothing_kernel,
                                                         smoothing_length)
            structure = TotalLagrangianSPHSystem(structure_initial; smoothing_kernel,
                                                 smoothing_length,
                                                 young_modulus=0.0,
                                                 poisson_ratio=0.0,
                                                 boundary_model,
                                                 hydrodynamic_boundary_particles)
            semi = Semidiscretization(fluid, structure; neighborhood_search=nothing,
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01); reset_threads=false)
            fluid, structure = ode.p.semi.systems
            v_ode, u_ode = ode.u0.x
            v_fluid = TrixiParticles.wrap_v(v_ode, fluid, ode.p.semi)
            v_fluid[3, 1] = 6.0
            structure.boundary_model.pressure .= [12.0, 30.0]
            TrixiParticles.update_nhs!(ode.p.semi, u_ode)
            TrixiParticles.update_average_pressure!(fluid, Val(true), v_ode, u_ode,
                                                    ode.p.semi)

            return fluid.cache.pressure_average[1], fluid.cache.neighbor_counter[1]
        end

        @test pressure_average(nothing) == (16.0, 3)
        @test pressure_average(1:2) == (16.0, 3)
        @test pressure_average([1]) == (9.0, 2)
        @test pressure_average(Int[]) == (6.0, 1)
    end

    @testset "Particle shifting" begin
        function shifting_velocity(hydrodynamic_boundary_particles, shifting_technique)
            spacing = 0.1
            density = 1000.0
            smoothing_kernel = WendlandC2Kernel{2}()
            smoothing_length = spacing
            state_equation = StateEquationCole(; sound_speed=10.0,
                                               reference_density=density,
                                               exponent=1.0)
            fluid_initial = InitialCondition(; coordinates=reshape([0.0, 0.08], 2, 1),
                                             velocity=reshape([1.0, 0.0], 2, 1),
                                             density, particle_spacing=spacing)
            fluid = WeaklyCompressibleSPHSystem(fluid_initial; smoothing_kernel,
                                                smoothing_length,
                                                density_calculator=ContinuityDensity(),
                                                state_equation, shifting_technique)

            structure_initial = InitialCondition(; coordinates=[0.0 0.06
                                                                0.0 0.0],
                                                 density=1200.0,
                                                 particle_spacing=spacing)
            boundary_model = BoundaryModelDummyParticles(fill(density, 2),
                                                         fill(density * spacing^2, 2),
                                                         AdamiPressureExtrapolation(),
                                                         smoothing_kernel,
                                                         smoothing_length;
                                                         state_equation)
            structure = TotalLagrangianSPHSystem(structure_initial; smoothing_kernel,
                                                 smoothing_length,
                                                 young_modulus=0.0,
                                                 poisson_ratio=0.0,
                                                 boundary_model,
                                                 hydrodynamic_boundary_particles)
            semi = Semidiscretization(fluid, structure; neighborhood_search=nothing,
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01); reset_threads=false)
            fluid = ode.p.semi.systems[1]
            v_ode, u_ode = ode.u0.x
            v = TrixiParticles.wrap_v(v_ode, fluid, ode.p.semi)
            u = TrixiParticles.wrap_u(u_ode, fluid, ode.p.semi)
            TrixiParticles.update_nhs!(ode.p.semi, u_ode)
            TrixiParticles.update_shifting!(fluid, shifting_technique, v, u,
                                            v_ode, u_ode, ode.p.semi)
            return Array(fluid.cache.delta_v)
        end

        shifting_techniques = (ConsistentShiftingSun2019(),
                               TransportVelocityAdami(background_pressure=10_000.0))
        for shifting_technique in shifting_techniques
            full_velocity = shifting_velocity(1:2, shifting_technique)
            subset_velocity = shifting_velocity([1], shifting_technique)
            empty_velocity = shifting_velocity(Int[], shifting_technique)

            @test norm(full_velocity) > eps()
            @test norm(subset_velocity) > eps()
            @test empty_velocity == zeros(2, 1)
        end
    end

    @testset "Interpolation" begin
        function interpolation_counts(hydrodynamic_boundary_particles)
            spacing = 0.1
            density = 1000.0
            smoothing_kernel = WendlandC2Kernel{2}()
            smoothing_length = spacing
            state_equation = StateEquationCole(; sound_speed=10.0,
                                               reference_density=density,
                                               exponent=1.0)
            fluid_initial = InitialCondition(; coordinates=reshape([0.0, 0.08], 2, 1),
                                             density, particle_spacing=spacing)
            fluid = WeaklyCompressibleSPHSystem(fluid_initial; smoothing_kernel,
                                                smoothing_length,
                                                density_calculator=ContinuityDensity(),
                                                state_equation)

            structure_initial = InitialCondition(; coordinates=[0.0 0.06
                                                                0.0 0.0],
                                                 density=1200.0,
                                                 particle_spacing=spacing)
            boundary_model = BoundaryModelDummyParticles(fill(density, 2),
                                                         fill(density * spacing^2, 2),
                                                         AdamiPressureExtrapolation(),
                                                         smoothing_kernel,
                                                         smoothing_length;
                                                         state_equation)
            structure = TotalLagrangianSPHSystem(structure_initial; smoothing_kernel,
                                                 smoothing_length,
                                                 young_modulus=0.0,
                                                 poisson_ratio=0.0,
                                                 boundary_model,
                                                 hydrodynamic_boundary_particles)
            semi = Semidiscretization(fluid, structure; neighborhood_search=nothing,
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01); reset_threads=false)
            fluid, structure = ode.p.semi.systems
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)

            function interpolation_searches(ref_system)
                search_radius = TrixiParticles.compact_support(ref_system.smoothing_kernel,
                                                               smoothing_length)
                return map(ode.p.semi.systems) do system
                    TrivialNeighborhoodSearch{2}(; search_radius,
                                                 eachpoint=TrixiParticles.each_active_particle(system))
                end
            end

            fluid_result = TrixiParticles.interpolate_points([0.0; 0.08;;], ode.p.semi,
                                                             fluid, v_ode, u_ode,
                                                             interpolation_searches(fluid);
                                                             cut_off_bnd=false,
                                                             include_wall_velocity=true)
            structure_result = TrixiParticles.interpolate_points([0.03; 0.0;;],
                                                                 ode.p.semi, structure,
                                                                 v_ode, u_ode,
                                                                 interpolation_searches(structure);
                                                                 cut_off_bnd=false)
            return fluid_result.neighbor_count[1], structure_result.neighbor_count[1]
        end

        @test interpolation_counts(nothing) == (3, 2)
        @test interpolation_counts(1:2) == (3, 2)
        @test interpolation_counts([1]) == (2, 2)
        @test interpolation_counts(Int[]) == (1, 2)
    end
end
