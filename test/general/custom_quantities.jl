@testset verbose=true "Custom Quantities" begin
    particle_spacing = 0.1
    coordinates = [0.0 0.1 0.2; 0.0 0.0 0.1]
    velocities = [1.0 2.0 0.5; 0.5 1.0 1.5]
    densities = [1000.0, 1000.0, 1000.0]
    pressures = [101325.0, 101330.0, 101320.0]

    initial_condition = InitialCondition(; coordinates, velocity=velocities,
                                         density=densities,
                                         pressure=pressures, particle_spacing)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.2 * particle_spacing

    fluid_system = EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                               smoothing_length, sound_speed=1.0)
    fluid_system.cache.density .= initial_condition.density

    boundary_model = BoundaryModelDummyParticles(initial_condition.density,
                                                 initial_condition.mass,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length)

    boundary_system = WallBoundarySystem(initial_condition, boundary_model)

    semi = Semidiscretization(fluid_system, boundary_system)

    v_ode, u_ode = semidiscretize(semi, (0, 1)).u0.x
    dv_ode, du_ode = similar(v_ode)
    t = 0.0

    @testset "Kinetic Energy" begin
        @testset "Fluid System" begin
            ekin = kinetic_energy(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)

            expected_ekin = sum(velocities) do velocity
                return 0.5 * first(fluid_system.mass) * dot(velocity, velocity)
            end

            @test isapprox(expected_ekin, ekin)

            # Test the `PtrArray` path used without `ThreadedBroadcastArray`.
            ekin_ptr_array = kinetic_energy(fluid_system, dv_ode, du_ode,
                                            parent(v_ode), u_ode, semi, t)
            @test isapprox(expected_ekin, ekin_ptr_array)
        end

        @testset "Boundary System" begin
            ekin = kinetic_energy(boundary_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            @test ekin == 0
        end
    end

    @testset "Total Mass" begin
        @testset "Fluid System" begin
            mass = total_mass(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            expected_mass = first(fluid_system.mass) * nparticles(fluid_system)

            @test isapprox(mass, expected_mass)
        end

        @testset "Boundary System" begin
            mass = total_mass(boundary_system, dv_ode, du_ode, v_ode, u_ode, semi, t)

            @test isnan(mass)
        end
    end

    @testset "Pressure Quantities" begin
        @testset "Max Pressure" begin
            max_p = max_pressure(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            @test isapprox(max_p, 101330.0)

            # Boundary system should return NaN
            @test isnan(max_pressure(boundary_system, dv_ode, du_ode, v_ode, u_ode,
                                     semi, t))
        end

        @testset "Min Pressure" begin
            min_p = min_pressure(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            @test min_p ≈ 101320.0

            # Boundary system should return NaN
            @test isnan(min_pressure(boundary_system, dv_ode, du_ode, v_ode, u_ode,
                                     semi, t))
        end

        @testset "Average Pressure" begin
            avg_p = avg_pressure(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            expected_avg = (101325.0 + 101330.0 + 101320.0) / 3
            @test isapprox(avg_p, expected_avg)

            # Boundary system should return NaN
            @test isnan(avg_pressure(boundary_system, dv_ode, du_ode, v_ode, u_ode,
                                     semi, t))
        end
    end

    @testset "Density Quantities" begin
        @testset "max_density" begin
            max_d = max_density(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            @test isapprox(max_d, 1000.0)  # All particles have same density

            # Boundary system should return NaN
            @test isnan(max_density(boundary_system, dv_ode, du_ode, v_ode, u_ode, semi, t))
        end

        @testset "min_density" begin
            min_d = min_density(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            @test isapprox(min_d, 1000.0)

            # Boundary system should return NaN
            @test isnan(min_density(boundary_system, dv_ode, du_ode, v_ode, u_ode, semi, t))
        end

        @testset "avg_density" begin
            avg_d = avg_density(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            @test isapprox(avg_d, 1000.0)

            # Boundary system should return NaN
            @test isnan(avg_density(boundary_system, dv_ode, du_ode, v_ode, u_ode, semi, t))
        end
    end

    @testset "Structure kinetic energy" begin
        struct EnergyStructureMock{IC, M} <: TrixiParticles.AbstractStructureSystem{2}
            initial_condition::IC
            mass::M
        end

        Base.eltype(::EnergyStructureMock) = Float64
        TrixiParticles.compact_support(::EnergyStructureMock, neighbor) = 1.0
        function TrixiParticles.write_u0!(u0, system::EnergyStructureMock)
            u0 .= system.initial_condition.coordinates
            return u0
        end
        function TrixiParticles.write_v0!(v0, system::EnergyStructureMock)
            v0 .= system.initial_condition.velocity
            return v0
        end

        coordinates = [0.0 1.0 2.0
                       0.0 0.0 0.0]
        velocity = [1.0 2.0 3.0
                    4.0 5.0 6.0]
        mass = [1.0, 2.0, 3.0]
        ic = InitialCondition(; coordinates, velocity, mass, density=ones(3))
        system = EnergyStructureMock(ic, mass)
        semi = Semidiscretization(system; neighborhood_search=nothing)
        ode = semidiscretize(semi, (0.0, 1.0))
        v_ode, u_ode = ode.u0.x
        dv_ode, du_ode = similar(v_ode), similar(u_ode)

        expected = sum(axes(velocity, 2)) do particle
            return mass[particle] * dot(velocity[:, particle], velocity[:, particle]) / 2
        end

        @test kinetic_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t) == expected
    end

    @testset "Active particle reductions" begin
        coordinates = [0.0 1.0 2.0
                       0.0 0.0 0.0]
        velocity = [1.0 10.0 3.0
                    2.0 20.0 4.0]
        mass = [1.0, 2.0, 4.0]
        density = [10.0, 50.0, 30.0]
        pressure = [100.0, 500.0, 300.0]

        # The density needs to be constant when using a buffer, so we set it in `v_ode` below
        ic = InitialCondition(; coordinates, velocity, mass,
                              density=fill(density[1], 3), particle_spacing=1.0)
        fluid_system = WeaklyCompressibleSPHSystem(ic;
                                                   smoothing_kernel,
                                                   smoothing_length,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation=StateEquationCole(;
                                                                                    sound_speed=1.0,
                                                                                    reference_density=density[1],
                                                                                    exponent=7),
                                                   buffer_size=1)
        fluid_system.pressure[1:3] .= pressure

        # Deactivate particle 2
        fluid_system.buffer.active_particle[2] = false
        TrixiParticles.update_system_buffer!(fluid_system.buffer)

        semi_active = Semidiscretization(fluid_system)
        ode = semidiscretize(semi_active, (0.0, 1.0))
        v_ode, u_ode = ode.u0.x
        dv_ode, du_ode = similar(v_ode), similar(u_ode)

        # Set varying densities in `v_ode` (density is the last row for `ContinuityDensity`)
        reshape(v_ode, 3, :)[3, 1:3] .= density

        @test total_mass(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi_active, t) ==
              5.0
        @test max_pressure(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi_active, t) ==
              300.0
        @test min_pressure(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi_active, t) ==
              100.0
        @test avg_pressure(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi_active, t) ==
              200.0
        @test max_density(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi_active, t) ==
              30.0
        @test min_density(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi_active, t) ==
              10.0
        @test avg_density(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi_active, t) ==
              20.0
        @test kinetic_energy(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi_active, t) ==
               52.5
    end
end
