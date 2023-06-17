
@testset verbose=true "Dummy Particles" begin @testset "Viscosity Adami: Wall Velocity" begin
    particle_spacing = 0.1

    # boundary particles in fluid compact support
    boundary_1 = RectangularShape(particle_spacing, (10, 1), (0.0, 0.2), 257.0)
    boundary_2 = RectangularShape(particle_spacing, (10, 1), (0.0, 0.1), 257.0)

    # boundary particles out of fluid compact support
    boundary_3 = RectangularShape(particle_spacing, (10, 1), (0, 0), 257.0)

    boundary = InitialCondition(boundary_1, boundary_2, boundary_3)

    particles_in_compact_support = length(boundary_1.mass) + length(boundary_2.mass)

    fluid = RectangularShape(particle_spacing, (16, 5), (-0.3, 0.3), 257.0)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.2 * particle_spacing
    viscosity = ViscosityAdami(1e-6)
    state_equation = StateEquationCole(10, 7, 257, 0.0)

    boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                                 state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length,
                                                 viscosity=viscosity)

    boundary_system = BoundarySPHSystem(boundary.coordinates, boundary_model)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(),
                                               state_equation,
                                               smoothing_kernel, smoothing_length)

    neighborhood_search = TrixiParticles.TrivialNeighborhoodSearch(TrixiParticles.eachparticle(fluid_system))

    v_fluid = [0 1 -1 0.7 0.3; -1 1 0 0.2 0.8]
    velocities = [[0; -1], [1; 1], [-1; 0], [0.7; 0.2], [0.3; 0.8]]

    @testset "Wall Velocity $v_fluid" for v_fluid in velocities
        TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                    boundary_system.boundary_model.viscosity)
        TrixiParticles.adami_pressure_extrapolation!(boundary_model, boundary_system,
                                                     fluid_system, boundary.coordinates,
                                                     fluid.coordinates,
                                                     v_fluid .*
                                                     ones(size(fluid.coordinates)),
                                                     neighborhood_search)

        v_wall = zeros(size(boundary.coordinates))
        v_wall[:, Base.OneTo(particles_in_compact_support)] .= -v_fluid

        @test isapprox(boundary_system.boundary_model.cache.wall_velocity, v_wall)
    end

    scale_v = [1, 0.5, 0.7, 1.8, 67.5]
    @testset "Wall Velocity Staggerd: Factor $scale" for scale in scale_v
        v_fluid = zeros(size(fluid.coordinates))
        for i in Base.OneTo(TrixiParticles.nparticles(fluid_system))
            if true == mod(i, 2)
                v_fluid[:, i] .= scale
            end
        end

        TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                    boundary_system.boundary_model.viscosity)
        TrixiParticles.adami_pressure_extrapolation!(boundary_model, boundary_system,
                                                     fluid_system, boundary.coordinates,
                                                     fluid.coordinates, v_fluid,
                                                     neighborhood_search)

        v_wall = zeros(size(boundary.coordinates))

        # first boundary row
        for i in Base.OneTo(length(boundary_1.mass))
            if true == mod(i, 2)
                v_wall[:, i] .= -0.42040669416720744 * scale
            else
                v_wall[:, i] .= -0.5795933058327924 * scale
            end
        end

        # second boundary row
        for i in (length(boundary_1.mass) + 1):particles_in_compact_support
            if true == mod(i, 2)
                v_wall[:, i] .= -0.12101100073462243 * scale
            else
                v_wall[:, i] .= -0.8789889992653775 * scale
            end
        end

        @test isapprox(boundary_system.boundary_model.cache.wall_velocity, v_wall)
    end
end end
