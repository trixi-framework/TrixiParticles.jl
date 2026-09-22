@testset "Marrone pressure extrapolation" begin
    function marrone_test_setup(D; viscosity=nothing, state_equation=nothing,
                                clip_negative_pressure=false, prescribed_motion=nothing)
        interpolation_point = zeros(D, 1)
        interpolation_point[1] = 1
        boundary_coordinates = -interpolation_point
        fluid_coordinates = stack([collect(x)
                                   for x in Iterators.product(ntuple(i -> i == 1 ?
                                                                          [0.7, 1.1, 1.3] :
                                                                          [-0.2, 0.05, 0.3],
                                                                     D)...)] |> vec)
        n_particles = size(fluid_coordinates, 2)
        velocity = [dimension + 0.3 * fluid_coordinates[1, particle] -
                    0.2 * fluid_coordinates[D, particle]
                    for dimension in 1:D, particle in 1:n_particles]

        fluid = InitialCondition(; coordinates=fluid_coordinates, velocity,
                                 density=fill(1000.0, n_particles),
                                 mass=collect(range(0.5, 2.0; length=n_particles)),
                                 particle_spacing=0.1)
        smoothing_kernel = SchoenbergCubicSplineKernel{D}()
        fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length=0.4,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation,
                                                   acceleration=ntuple(i -> i == 1 ? -2.0 :
                                                                            0.0, D))

        boundary = InitialCondition(; coordinates=boundary_coordinates, density=1000.0,
                                    particle_spacing=0.1,
                                    normals=(boundary_coordinates - interpolation_point) /
                                            2)
        model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                            MarronePressureExtrapolation(),
                                            smoothing_kernel, 0.4; viscosity,
                                            state_equation,
                                            clip_negative_pressure)
        wall = WallBoundarySystem(boundary, model; prescribed_motion)
        v_fluid = vcat(velocity, fluid.density')

        return (; fluid_system, wall, model, fluid_coordinates, boundary_coordinates,
                v_fluid, interpolation_point)
    end

    @testset "linear MLS in $D dimensions" for D in (2, 3)
        viscosity = ViscosityAdami(nu=1.0e-6)
        setup = marrone_test_setup(D; viscosity)
        (; fluid_system, wall, model, fluid_coordinates, boundary_coordinates,
         v_fluid) = setup
        fluid_system.pressure .= [6000 - 2000 * fluid_coordinates[1, particle] +
                                  3 * fluid_coordinates[D, particle]
                                  for particle in axes(fluid_coordinates, 2)]

        TrixiParticles.accumulate_marrone!(model, wall, fluid_system,
                                           boundary_coordinates, fluid_coordinates,
                                           v_fluid, DummySemidiscretization())
        TrixiParticles.finalize_marrone!(model, wall, zeros(1, 1), 1)

        @test model.pressure[1] ≈ 8000
        @test model.cache.density[1] == 1000 # No state equation (EDAC behavior)
        @test model.cache.wall_velocity[:, 1] ≈ -(collect(1:D) .+ 0.3)
    end

    @testset "deficient support" begin
        for D in (2, 3)
            N = D + 1
            basis = SVector{N}(ntuple(i -> i == 1 ? 1.0 : 0.2, N))
            moment = 2 * basis * basis'
            coefficients = TrixiParticles.marrone_mls_coefficients(moment)
            @test dot(coefficients, 2 * basis * 7) ≈ 7
            @test iszero(TrixiParticles.marrone_mls_coefficients(zero(moment)))
        end
    end

    @testset "full update, clipping, and interaction filtering" begin
        state_equation = StateEquationCole(sound_speed=20.0, reference_density=1000.0,
                                           exponent=1, clip_negative_pressure=false)
        for clip_negative_pressure in (false, true)
            setup = marrone_test_setup(2; state_equation, clip_negative_pressure)
            (; fluid_system, wall, model) = setup
            semi = Semidiscretization(fluid_system, wall)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            v = TrixiParticles.wrap_v(v_ode, wall, semi)
            u = TrixiParticles.wrap_u(u_ode, wall, semi)

            fluid_system.pressure .= -5000 # Hydrostatic contribution is +4000.
            for _ in 1:2
                TrixiParticles.update_pressure!(model, wall, v, u, v_ode, u_ode, semi)
                @test model.pressure[1] ≈ (clip_negative_pressure ? 0 : -1000)
                @test model.cache.density[1] ≈
                      (clip_negative_pressure ? 1000 : 997.5)
            end

            semi.interaction_matrix[2, 1] = false
            TrixiParticles.update_pressure!(model, wall, v, u, v_ode, u_ode, semi)
            @test model.pressure[1] == 0
            @test model.cache.density[1] ≈ 1000
        end
    end

    @testset "automatic mirror points and prescribed motion" begin
        setup = marrone_test_setup(2)
        @test setup.model.cache.interpolation_coordinates ≈ setup.interpolation_point

        rotation(x, t) = SVector(cos(t) * x[1] - sin(t) * x[2],
                                 sin(t) * x[1] + cos(t) * x[2])
        motion = PrescribedMotion(rotation, t -> true)
        moving_setup = marrone_test_setup(2; prescribed_motion=motion)
        TrixiParticles.update_positions!(moving_setup.wall, nothing, nothing, nothing,
                                         nothing, DummySemidiscretization(), pi / 2)
        @test moving_setup.wall.coordinates[:, 1] ≈ [0.0, -1.0] atol = 1.0e-14
        @test moving_setup.model.cache.interpolation_coordinates[:, 1] ≈ [0.0, 1.0] atol = 1.0e-14
    end

    @testset "validation" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        for normals in (nothing, zeros(2, 1), fill(NaN, 2, 1), fill(Inf, 2, 1))
            boundary = InitialCondition(; coordinates=[-0.1; 0.0;;], density=1000.0,
                                        particle_spacing=0.1, normals)
            model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                                MarronePressureExtrapolation(), kernel, 0.1)
            @test_throws ArgumentError WallBoundarySystem(boundary, model)
        end

        setup = marrone_test_setup(2)
        @test_throws ArgumentError Semidiscretization(setup.wall;
                                                      neighborhood_search=PrecomputedNeighborhoodSearch{2}())
        @test repr(setup.model) ==
              "BoundaryModelDummyParticles(MarronePressureExtrapolation, Nothing)"
    end
end
