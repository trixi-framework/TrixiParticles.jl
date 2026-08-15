@testset verbose=true "Velocity Averaging" begin
    function velocity_averaging_test_system(; velocity_averaging=nothing,
                                            clamped_particles=Int[])
        coordinates = [0.0 1.0 2.0
                       0.0 0.0 0.0]
        velocity = [1.0 3.0 5.0
                    2.0 4.0 6.0]
        mass = ones(3)
        density = ones(3)
        initial_condition = InitialCondition(; coordinates, velocity, mass, density)

        return TotalLagrangianSPHSystem(initial_condition;
                                        smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                        smoothing_length=1.0, young_modulus=1.0,
                                        poisson_ratio=0.25, clamped_particles,
                                        self_interaction_nhs=nothing, velocity_averaging)
    end

    @testset "Constructor" begin
        velocity_averaging = VelocityAveraging(time_constant=1.0f-2)

        @test velocity_averaging isa VelocityAveraging{Float32}
        @test velocity_averaging.time_constant === 1.0f-2
    end

    @testset "No velocity averaging" begin
        system = velocity_averaging_test_system()
        semi = Semidiscretization(system, neighborhood_search=nothing)
        system = semi.systems[1]
        v_ode = vec([10.0 30.0 50.0
                     20.0 40.0 60.0])
        v = TrixiParticles.wrap_v(v_ode, system, semi)

        @test TrixiParticles.velocity_averaging(system) === nothing
        @test_nowarn TrixiParticles.initialize_averaged_velocity!(system, v_ode, semi, 0.5)
        @test_nowarn TrixiParticles.compute_averaged_velocity!(system, v_ode, semi, 1.0)
        @test TrixiParticles.velocity_for_viscosity(v, system, 2) ==
              TrixiParticles.current_velocity(v, system, 2)
    end

    @testset "Initialize averaged velocity" begin
        velocity_averaging = VelocityAveraging(time_constant=0.5)
        system = velocity_averaging_test_system(; velocity_averaging,
                                                clamped_particles=[3])
        semi = Semidiscretization(system, neighborhood_search=nothing)
        system = semi.systems[1]
        v_ode = vec([10.0 30.0
                     20.0 40.0])

        fill!(system.cache.averaged_velocity, NaN)
        TrixiParticles.initialize_averaged_velocity!(system, v_ode, semi, 0.5)

        @test system.cache.averaged_velocity == [10.0 30.0 0.0
                                                 20.0 40.0 0.0]
        @test system.cache.t_last_averaging[] == 0.5
    end

    @testset "Compute averaged velocity" begin
        velocity_averaging = VelocityAveraging(time_constant=0.5)
        system = velocity_averaging_test_system(; velocity_averaging,
                                                clamped_particles=[3])
        semi = Semidiscretization(system, neighborhood_search=nothing)
        system = semi.systems[1]
        initial_velocity = [10.0 30.0
                            20.0 40.0]
        v_ode_initial = vec(initial_velocity)
        new_velocity = [14.0 38.0
                        28.0 56.0]
        v_ode_new = vec(new_velocity)
        v = TrixiParticles.wrap_v(v_ode_new, system, semi)

        TrixiParticles.initialize_averaged_velocity!(system, v_ode_initial, semi, 0.5)
        TrixiParticles.compute_averaged_velocity!(system, v_ode_new, semi, 0.75)

        alpha = 1 - exp(-(0.75 - 0.5) / velocity_averaging.time_constant)
        expected_integrated = (1 - alpha) * initial_velocity + alpha * new_velocity
        @test system.cache.averaged_velocity[:, 1:2] ≈ expected_integrated
        @test system.cache.averaged_velocity[:, 3] == zeros(2)
        @test system.cache.t_last_averaging[] == 0.75

        # Test `velocity_for_viscosity`.
        @test TrixiParticles.velocity_for_viscosity(v, system, 1) ==
              TrixiParticles.extract_svector(expected_integrated, system, 1)
        @test TrixiParticles.velocity_for_viscosity(v, system, 1) !=
              TrixiParticles.current_velocity(v, system, 1)
        @test TrixiParticles.velocity_for_viscosity(v, system, 2) ==
              TrixiParticles.extract_svector(expected_integrated, system, 2)
        @test TrixiParticles.velocity_for_viscosity(v, system, 3) == zeros(2)
    end
end;
