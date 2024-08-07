@testset verbose=true "Dummy Particles" begin
    @testset "show" begin
        boundary_model = BoundaryModelDummyParticles([1000.0], [1.0],
                                                     SummationDensity(),
                                                     SchoenbergCubicSplineKernel{2}(), 0.1)

        show_compact = "BoundaryModelDummyParticles(SummationDensity, Nothing)"
        @test repr(boundary_model) == show_compact
    end

    @testset "Viscosity Adami: Wall Velocity" begin
        particle_spacing = 0.1

        # Boundary particles in fluid compact support
        boundary_1 = RectangularShape(particle_spacing, (10, 1), (0.0, 0.2), density=257.0)
        boundary_2 = RectangularShape(particle_spacing, (10, 1), (0.0, 0.1), density=257.0)

        # Boundary particles out of fluid compact support
        boundary_3 = RectangularShape(particle_spacing, (10, 1), (0, 0), density=257.0)

        boundary = union(boundary_1, boundary_2, boundary_3)

        particles_in_compact_support = length(boundary_1.mass) + length(boundary_2.mass)

        fluid = RectangularShape(particle_spacing, (16, 5), (-0.3, 0.3), density=257.0,
                                 loop_order=:x_first)

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2 * particle_spacing
        viscosity = ViscosityAdami(nu=1e-6)
        state_equation = StateEquationCole(sound_speed=10, reference_density=257,
                                           exponent=7)

        boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                                     state_equation=state_equation,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length,
                                                     viscosity=viscosity)

        boundary_system = BoundarySPHSystem(boundary, boundary_model)

        fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(),
                                                   state_equation,
                                                   smoothing_kernel, smoothing_length)

        neighborhood_search = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius=1.0,
                                                                          eachpoint=TrixiParticles.eachparticle(fluid_system))

        velocities = [[0; -1], [1; 1], [-1; 0], [0.7; 0.2], [0.3; 0.8]]

        @testset "Wall Velocity $v_fluid" for v_fluid in velocities
            viscosity = boundary_system.boundary_model.viscosity
            volume = boundary_system.boundary_model.cache.volume

            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        boundary_system.boundary_model.viscosity)
            TrixiParticles.boundary_pressure_extrapolation!(boundary_model,
                                                            boundary_system,
                                                            fluid_system,
                                                            boundary.coordinates,
                                                            fluid.coordinates, v_fluid,
                                                            v_fluid .*
                                                            ones(size(fluid.coordinates)),
                                                            neighborhood_search)

            for particle in TrixiParticles.eachparticle(boundary_system)
                if volume[particle] > eps()
                    TrixiParticles.compute_wall_velocity!(viscosity, boundary_system,
                                                          boundary.coordinates, particle)
                end
            end

            v_wall = zeros(size(boundary.coordinates))
            v_wall[:, 1:particles_in_compact_support] .= -v_fluid

            @test isapprox(boundary_system.boundary_model.cache.wall_velocity, v_wall)
        end

        scale_v = [1, 0.5, 0.7, 1.8, 67.5]
        @testset "Wall Velocity Staggerd: Factor $scale" for scale in scale_v
            viscosity = boundary_system.boundary_model.viscosity
            volume = boundary_system.boundary_model.cache.volume

            # For a constant velocity profile (each fluid particle has the same velocity)
            # the wall velocity is `v_wall = -v_fluid` (see eq. 22 in Adami_2012).
            # Thus, generate a staggered velocity profile to test the smoothed velocity field
            # for a variable velocity profile.
            v_fluid = zeros(size(fluid.coordinates))
            for i in TrixiParticles.eachparticle(fluid_system)
                if mod(i, 2) == 1
                    v_fluid[:, i] .= scale
                end
            end

            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        boundary_system.boundary_model.viscosity)
            TrixiParticles.boundary_pressure_extrapolation!(boundary_model, boundary_system,
                                                            fluid_system,
                                                            boundary.coordinates,
                                                            fluid.coordinates, v_fluid,
                                                            v_fluid,
                                                            neighborhood_search)

            for particle in TrixiParticles.eachparticle(boundary_system)
                if volume[particle] > eps()
                    TrixiParticles.compute_wall_velocity!(viscosity, boundary_system,
                                                          boundary.coordinates, particle)
                end
            end

            v_wall = zeros(size(boundary.coordinates))

            # First boundary row
            for i in 1:length(boundary_1.mass)
                if mod(i, 2) == 1

                    # Particles with a diagonal distance to a fluid particle with `v_fluid > 0.0`
                    v_wall[:, i] .= -0.42040669416720744 * scale
                else

                    # Particles with a orthogonal distance to a fluid particle with `v_fluid > 0.0`
                    v_wall[:, i] .= -0.5795933058327924 * scale
                end
            end

            # Second boundary row
            for i in (length(boundary_1.mass) + 1):particles_in_compact_support
                if true == mod(i, 2)

                    # Particles with a diagonal distance to a fluid particle with `v_fluid > 0.0`
                    v_wall[:, i] .= -0.12101100073462243 * scale
                else

                    # Particles with a orthogonal distance to a fluid particle with `v_fluid > 0.0`
                    v_wall[:, i] .= -0.8789889992653775 * scale
                end
            end

            @test isapprox(boundary_system.boundary_model.cache.wall_velocity, v_wall)
        end
    end
end

include("rhs.jl")
