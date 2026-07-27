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

    material_mass = [1.0, 2.0, 3.0]
    structure_system_ = TotalLagrangianSPHSystem(initial_condition;
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 young_modulus=1.0,
                                                 poisson_ratio=0.25,
                                                 material_mass)
    structure_semi_ = Semidiscretization(structure_system_)
    structure_ode = semidiscretize(structure_semi_, (0, 1))
    structure_v_ode, structure_u_ode = structure_ode.u0.x
    structure_dv_ode, structure_du_ode = similar(structure_v_ode)
    structure_semi = structure_ode.p.semi
    structure_system = structure_semi.systems[1]

    t = 0.0

    @testset "Kinetic Energy" begin
        @testset "Fluid System" begin
            ekin = kinetic_energy(fluid_system, dv_ode, du_ode, v_ode, u_ode, semi, t)

            expected_ekin = sum(velocities) do velocity
                return 0.5 * first(fluid_system.mass) * dot(velocity, velocity)
            end

            @test isapprox(expected_ekin, ekin)
        end

        @testset "Boundary System" begin
            ekin = kinetic_energy(boundary_system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            @test ekin == 0
        end

        @testset "TLSPH System" begin
            ekin = kinetic_energy(structure_system,
                                  structure_dv_ode, structure_du_ode,
                                  structure_v_ode, structure_u_ode,
                                  structure_semi, t)
            expected_ekin = sum(eachindex(material_mass)) do particle
                velocity = velocities[:, particle]
                return material_mass[particle] * dot(velocity, velocity) / 2
            end

            @test ekin ≈ expected_ekin
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

        @testset "TLSPH System" begin
            mass = total_mass(structure_system,
                              structure_dv_ode, structure_du_ode,
                              structure_v_ode, structure_u_ode,
                              structure_semi, t)
            @test mass ≈ sum(material_mass)
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
end
