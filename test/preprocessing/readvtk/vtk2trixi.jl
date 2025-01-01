@testset verbose=true "vtk2trixi" begin
    @testset verbose=true "Functionality Check - 2D" begin

        # 'InitialCondition'-Files
        saved_ic = rectangular_patch(0.1, (10, 20))

        file = trixi2vtk(saved_ic; output_directory="test/preprocessing/readvtk",
                         filename="initial_condition")

        loaded_ic = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                       "initial_condition.vtu"))

        @test isapprox(saved_ic.coordinates, loaded_ic.coordinates, rtol=1e-5)
        @test isapprox(saved_ic.velocity, loaded_ic.velocity, rtol=1e-5)
        @test isapprox(saved_ic.density, loaded_ic.density, rtol=1e-5)
        @test isapprox(saved_ic.pressure, loaded_ic.pressure, rtol=1e-5)
        #@test isapprox(saved_ic.mass, loaded_ic.mass, rtol=1e-5) #TODO: wait until mass is written out with 'write2vtk'

        # Simulations-Files
        particle_spacing = 0.05
        smoothing_length = 0.15
        boundary_layers = 3
        open_boundary_layers = 6

        density = 1000.0
        pressure = 1.0
        domain_size = (1.0, 0.4)

        flow_direction = [1.0, 0.0]
        reynolds_number = 100
        prescribed_velocity = 2.0
        sound_speed = 20.0

        boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                         domain_size[2])

        state_equation = StateEquationCole(; sound_speed=sound_speed,
                                           reference_density=density,
                                           exponent=7, background_pressure=pressure)

        sim_ic = RectangularTank(particle_spacing, domain_size, boundary_size, density,
                                 pressure=pressure, n_layers=boundary_layers,
                                 faces=(false, false, true, true))

        sim_ic.boundary.coordinates[1, :] .-= particle_spacing * open_boundary_layers

        # ==== Fluid System
        smoothing_kernel = WendlandC2Kernel{2}()
        fluid_density_calculator = SummationDensity()

        kinematic_viscosity = 0.008
        n_buffer_particles = 0

        viscosity = ViscosityAdami(nu=kinematic_viscosity)

        fluid_system = EntropicallyDampedSPHSystem(sim_ic.fluid, smoothing_kernel,
                                                   smoothing_length,
                                                   sound_speed, viscosity=viscosity,
                                                   density_calculator=fluid_density_calculator,
                                                   buffer_size=n_buffer_particles)

        fluid_system.cache.density .= sim_ic.fluid.density
        # Write out 'fluid_system' Simulation-File                                          
        trixi2vtk(sim_ic.fluid.velocity, sim_ic.fluid.coordinates, 0.0, fluid_system,
                  nothing;
                  output_directory="test/preprocessing/readvtk", iter=1)

        # Load 'fluid_system' Simulation-File
        loaded_fs = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                       "fluid_1.vtu"))

        @test isapprox(sim_ic.fluid.coordinates, loaded_fs.coordinates, rtol=1e-5)
        @test isapprox(sim_ic.fluid.velocity, loaded_fs.velocity, rtol=1e-5)
        @test isapprox(sim_ic.fluid.density, loaded_fs.density, rtol=1e-5)
        @test isapprox(sim_ic.fluid.pressure, loaded_fs.pressure, rtol=1e-5)
        #@test isapprox(sim_ic.fluid.mass, loaded_fs.mass, rtol=1e-5) #TODO: wait until mass is written out with 'write2vtk'

        # ==== Boundary System
        boundary_model = BoundaryModelDummyParticles(sim_ic.boundary.density,
                                                     sim_ic.boundary.mass,
                                                     AdamiPressureExtrapolation(),
                                                     state_equation=state_equation,
                                                     smoothing_kernel, smoothing_length)

        boundary_system = BoundarySPHSystem(sim_ic.boundary, boundary_model)

        trixi2vtk(sim_ic.boundary.velocity, sim_ic.boundary.coordinates, 0.0,
                  boundary_system,
                  nothing;
                  output_directory="test/preprocessing/readvtk", iter=1)

        # Load 'boundary_system' Simulation-File
        loaded_bs = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                       "boundary_1.vtu"))

        @test isapprox(sim_ic.boundary.coordinates, loaded_bs.coordinates, rtol=1e-5)
        @test isapprox(sim_ic.boundary.velocity, loaded_bs.velocity, rtol=1e-5)
        @test isapprox(sim_ic.boundary.density, loaded_bs.density, rtol=1e-5)
        @test isapprox(sim_ic.boundary.pressure, loaded_bs.pressure, rtol=1e-5)
        #@test isapprox(sim_ic.boundary.mass, loaded_bs.mass, rtol=1e-5) #TODO: wait until mass is written out with 'write2vtk'

        # ==== Open Boundary System
        plane_in = ([0.0, 0.0], [0.0, domain_size[2]])
        inflow = InFlow(; plane=plane_in, flow_direction,
                        open_boundary_layers, density=density, particle_spacing)

        function velocity_function2d(pos, t)
            return SVector(prescribed_velocity, 0.0)
        end
        open_boundary = OpenBoundarySPHSystem(inflow; fluid_system,
                                              boundary_model=BoundaryModelLastiwka(),
                                              buffer_size=n_buffer_particles,
                                              reference_density=density,
                                              reference_pressure=pressure,
                                              reference_velocity=velocity_function2d)
    end
end