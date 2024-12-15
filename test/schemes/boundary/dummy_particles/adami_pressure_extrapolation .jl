@testset "Pressure Extrapolation Adami" begin
    @testset "Pressure Extrapolation Adami: Constant Pressure" begin
        particle_spacing = 0.1
        num_particles = 4
        n_blayers = 3
        width = particle_spacing * num_particles
        height = particle_spacing * num_particles

        density = 257
        tank1 = RectangularTank(particle_spacing, (width, height), (width, height),
                                density, n_layers=n_blayers,
                                faces=(true, true, true, false))

        # particles_in_compact_support = length(tank1.boundary.mass)

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2 * particle_spacing
        viscosity = ViscosityAdami(nu=1e-6)
        state_equation = StateEquationCole(sound_speed=10, reference_density=257,
                                           exponent=7)

        boundary_model = BoundaryModelDummyParticles(tank1.boundary.density,
                                                     tank1.boundary.mass,
                                                     state_equation=state_equation,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length,
                                                     viscosity=viscosity)

        boundary_system = BoundarySPHSystem(tank1.boundary, boundary_model)

        fluid_system = WeaklyCompressibleSPHSystem(tank1.fluid, SummationDensity(),
                                                   state_equation,
                                                   smoothing_kernel, smoothing_length)
        fluid_system.cache.density .= tank1.fluid.density
        v_fluid = zeros(2, TrixiParticles.nparticles(fluid_system))
        TrixiParticles.compute_pressure!(fluid_system, v_fluid)
        # TrixiParticles.@autoinfiltrate

        neighborhood_search = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius=TrixiParticles.compact_support(smoothing_kernel,
                                                                                                                       smoothing_length),
                                                                          eachpoint=TrixiParticles.eachparticle(fluid_system))

        viscosity = boundary_system.boundary_model.viscosity

        # @infiltrate
        TrixiParticles.set_zero!(boundary_model.pressure)
        TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                    viscosity)
        # @infiltrate
        TrixiParticles.adami_pressure_extrapolation!(boundary_model, boundary_system,
                                                     fluid_system,
                                                     tank1.boundary.coordinates,
                                                     tank1.fluid.coordinates,
                                                     v_fluid,
                                                     neighborhood_search)

        # # Plot the pressure for fluid and boundary points
        scatter(boundary_system.coordinates[1, :],
                boundary_system.coordinates[2, :],
                marker_z=boundary_system.boundary_model.pressure,
                marker_color=:thermal)
        scatter!(tank1.fluid.coordinates[1, :],
                 tank1.fluid.coordinates[2, :],
                 marker_z=fluid_system.pressure,
                 marker_color=:thermal)
        gui()

        @test all(boundary_system.boundary_model.pressure .== 0.0) &
              all(fluid_system.pressure .== 0.0)

        # TrixiParticles.@autoinfiltrate

        tank2 = RectangularTank(particle_spacing, (width, height), (width, height),
                                density, n_layers=n_blayers,
                                faces=(true, true, true, false))

        fluid_system2 = WeaklyCompressibleSPHSystem(tank2.fluid, SummationDensity(),
                                                    state_equation,
                                                    smoothing_kernel, smoothing_length)
        fluid_system2.cache.density .= tank2.fluid.density
        v_fluid2 = zeros(2, TrixiParticles.nparticles(fluid_system2))
        TrixiParticles.compute_pressure!(fluid_system2, v_fluid2)

        # TrixiParticles.@autoinfiltrate

        scatter(boundary_system.coordinates[1, :],
                boundary_system.coordinates[2, :],
                marker_z=boundary_system.boundary_model.pressure,
                marker_color=:thermal)
        scatter!(tank2.fluid.coordinates[1, :],
                 tank2.fluid.coordinates[2, :],
                 marker_z=fluid_system.pressure,
                 marker_color=:thermal)
        gui()

        # TrixiParticles.@autoinfiltrate
        @test true
    end

    @testset "Pressure Extrapolation Adami: Linear Pressure" begin
        particle_spacing = 0.1
        num_particles = 4
        n_blayers = 3
        width = particle_spacing * num_particles
        height = particle_spacing * num_particles

        density = 257

        state_equation = StateEquationCole(sound_speed=10, reference_density=257,
                                           exponent=7)

        tank1 = RectangularTank(particle_spacing, (width, height), (width, height),
                                density, acceleration=[0.0, -9.81],
                                state_equation=state_equation, n_layers=n_blayers,
                                faces=(true, true, true, false))

        # particles_in_compact_support = length(tank1.boundary.mass)

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2 * particle_spacing
        viscosity = ViscosityAdami(nu=1e-6)

        boundary_model = BoundaryModelDummyParticles(tank1.boundary.density,
                                                     tank1.boundary.mass,
                                                     state_equation=state_equation,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length,
                                                     viscosity=viscosity)

        boundary_system = BoundarySPHSystem(tank1.boundary, boundary_model)

        fluid_system1 = WeaklyCompressibleSPHSystem(tank1.fluid, SummationDensity(),
                                                    state_equation,
                                                    smoothing_kernel, smoothing_length)
        fluid_system1.cache.density .= tank1.fluid.density
        v_fluid1 = zeros(2, TrixiParticles.nparticles(fluid_system1))
        TrixiParticles.compute_pressure!(fluid_system1, v_fluid1)
        # TrixiParticles.@autoinfiltrate

        neighborhood_search = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius=TrixiParticles.compact_support(smoothing_kernel,
                                                                                                                       smoothing_length),
                                                                          eachpoint=TrixiParticles.eachparticle(fluid_system1))

        viscosity = boundary_system.boundary_model.viscosity

        # @infiltrate
        TrixiParticles.set_zero!(boundary_model.pressure)
        TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                    viscosity)
        # @infiltrate
        scatter(boundary_system.coordinates[1, :],
                boundary_system.coordinates[2, :],
                marker_z=boundary_system.boundary_model.pressure,
                marker_color=:thermal)
        scatter!(tank1.fluid.coordinates[1, :],
                 tank1.fluid.coordinates[2, :],
                 marker_z=fluid_system1.pressure,
                 marker_color=:thermal)
        gui()
        # TrixiParticles.@autoinfiltrate

        TrixiParticles.adami_pressure_extrapolation!(boundary_model, boundary_system,
                                                     fluid_system1,
                                                     tank1.boundary.coordinates,
                                                     tank1.fluid.coordinates,
                                                     v_fluid1,
                                                     neighborhood_search)

        # # Plot the pressure for fluid and boundary points
        # p1 = scatter(boundary_system.coordinates[1, :],
        #         boundary_system.coordinates[2, :],
        #         marker_z=boundary_system.boundary_model.pressure,
        #         marker_color=:thermal)
        p1 = scatter(tank1.fluid.coordinates[1, :],
                     tank1.fluid.coordinates[2, :],
                     marker_z=fluid_system1.pressure,
                     marker_color=:thermal)
        # gui()
        # TrixiParticles.@autoinfiltrate
        @test true

        width2 = particle_spacing * (num_particles + 2 * n_blayers)
        height2 = particle_spacing * (num_particles + n_blayers)

        tank2 = RectangularTank(particle_spacing, (width2, height2), (width2, height2),
                                density, acceleration=[0.0, -9.81],
                                state_equation=state_equation, n_layers=0,
                                faces=(true, true, true, false))

        # particles_in_compact_support = length(tank1.boundary.mass)
        fluid_system2 = WeaklyCompressibleSPHSystem(tank2.fluid, SummationDensity(),
                                                    state_equation,
                                                    smoothing_kernel, smoothing_length)

        fluid_system2.cache.density .= tank2.fluid.density
        v_fluid2 = zeros(2, TrixiParticles.nparticles(fluid_system2))
        TrixiParticles.compute_pressure!(fluid_system2, v_fluid2)
        # TrixiParticles.@autoinfiltrate
        p2 = scatter(tank2.fluid.coordinates[1, :],
                     tank2.fluid.coordinates[2, :],
                     marker_z=fluid_system2.pressure,
                     marker_color=:thermal)

        plot(p1, p2, layout=(1, 2), grid=true)
        gui()
        TrixiParticles.@autoinfiltrate

        press1 = reshape(fluid_system1.pressure, (num_particles, num_particles))
        press2 = reshape(fluid_system2.pressure,
                         (num_particles + 2 * n_blayers, num_particles + n_blayers))[(1 + n_blayers):(n_blayers + num_particles),
                                                                                     (1 + n_blayers):end]
    end
end
