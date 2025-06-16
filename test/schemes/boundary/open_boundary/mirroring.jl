@testset verbose=true "Mirroring" begin
    validation_dir = pkgdir(TrixiParticles, "test", "schemes", "boundary", "open_boundary",
                            "data")

    @testset verbose=true "2D" begin
        files = [
            "open_boundary_extrapolated_2d.csv",
            "open_boundary_extrapolated_skew_2d.csv"
        ]

        particle_spacing = 0.05
        domain_length = 1.0

        plane_boundary = [
            ([0.0, 0.0], [0.0, domain_length]),
            ([0.0, 0.0], [-domain_length, domain_length])
        ]
        plane_boundary_normal = [[1.0, 0.0], [1.0, 1.0]]

        function pressure_function(pos)
            t = 0
            U = 1.0
            b = -8pi^2 / 10
            x = pos[1]
            y = pos[2]

            return -U^2 * exp(2 * b * t) * (cos(4pi * x) + cos(4pi * y)) / 4
        end

        function velocity_function(pos)
            t = 0
            U = 1.0
            b = -8pi^2 / 10
            x = pos[1]
            y = pos[2]

            vel = U * exp(b * t) *
                  [-cos(2pi * x) * sin(2pi * y), sin(2pi * x) * cos(2pi * y)]

            return SVector{2}(vel)
        end

        n_particles_xy = round(Int, domain_length / particle_spacing)

        domain_fluid = RectangularShape(particle_spacing, (2, 2) .* n_particles_xy,
                                        (-domain_length, 0.0), density=1000.0,
                                        pressure=pressure_function,
                                        velocity=velocity_function)

        smoothing_length = 1.5 * particle_spacing
        smoothing_kernel = WendlandC2Kernel{ndims(domain_fluid)}()
        fluid_system = EntropicallyDampedSPHSystem(domain_fluid, smoothing_kernel,
                                                   smoothing_length, 1.0)

        fluid_system.cache.density .= domain_fluid.density

        @testset verbose=true "plane normal $i" for i in eachindex(files)
            inflow = BoundaryZone(; plane=plane_boundary[i], boundary_type=InFlow(),
                                  plane_normal=plane_boundary_normal[i],
                                  open_boundary_layers=10, density=1000.0, particle_spacing)

            open_boundary = OpenBoundarySPHSystem(inflow; fluid_system,
                                                  boundary_model=BoundaryModelTafuni(),
                                                  buffer_size=0)

            semi = Semidiscretization(fluid_system, open_boundary)
            TrixiParticles.initialize_neighborhood_searches!(semi)

            v_open_boundary = zero(inflow.initial_condition.velocity)
            v_fluid = vcat(domain_fluid.velocity, domain_fluid.pressure')

            TrixiParticles.set_zero!(open_boundary.pressure)

            TrixiParticles.extrapolate_values!(open_boundary, v_open_boundary, v_fluid,
                                               inflow.initial_condition.coordinates,
                                               domain_fluid.coordinates, semi, 0.0;
                                               prescribed_pressure=false,
                                               prescribed_velocity=false)
            # Checked visually in ParaView:
            # trixi2vtk(fluid_system.initial_condition, filename="fluid",
            #           v=domain_fluid.velocity, p=domain_fluid.pressure)

            # trixi2vtk(open_boundary.initial_condition, filename="open_boundary",
            #           v=v_open_boundary, p=open_boundary.pressure)

            data = TrixiParticles.CSV.read(joinpath(validation_dir, files[i]),
                                           TrixiParticles.DataFrame)

            expected_velocity = vcat((data.var"v:0")',
                                     (data.var"v:1")')
            expected_pressure = data.var"p"

            @test isapprox(v_open_boundary, expected_velocity, atol=1e-3)
            @test isapprox(open_boundary.pressure, expected_pressure, atol=1e-3)
        end
    end

    @testset verbose=true "3D" begin
        files = [
            "open_boundary_extrapolated_3d.csv",
            "open_boundary_extrapolated_skew_3d.csv"
        ]

        particle_spacing = 0.05
        domain_length = 1.0

        plane_boundary = [
            ([0.0, 0.0, 0.0], [domain_length, 0.0, 0.0], [0.0, domain_length, 0.0]),
            ([0.0, 0.0, 0.0], [domain_length, 0.0, 0.0],
             [0.0, domain_length, domain_length])
        ]
        plane_boundary_normal = [[0.0, 0.0, 1.0], [0.0, -1.0, 1.0]]

        function pressure_function(pos)
            t = 0
            U = 1.0
            b = -8pi^2 / 10
            x = pos[1]
            y = pos[2]

            return -U^2 * exp(2 * b * t) * (cos(4pi * x) + cos(4pi * y)) / 4
        end

        function velocity_function(pos)
            t = 0
            U = 1.0
            b = -8pi^2 / 10
            x = pos[1]
            y = pos[2]

            vel = U * exp(b * t) *
                  [-cos(2pi * x) * sin(2pi * y), 0.0, sin(2pi * x) * cos(2pi * y)]

            return SVector{3}(vel)
        end

        n_particles_xyz = round(Int, domain_length / particle_spacing)

        domain_fluid = RectangularShape(particle_spacing,
                                        (2, 2, 2) .* n_particles_xyz,
                                        (-domain_length, -domain_length, 0.0),
                                        density=1000.0, pressure=pressure_function,
                                        velocity=velocity_function)

        smoothing_length = 1.5 * particle_spacing
        smoothing_kernel = WendlandC2Kernel{ndims(domain_fluid)}()
        fluid_system = EntropicallyDampedSPHSystem(domain_fluid, smoothing_kernel,
                                                   smoothing_length, 1.0)

        fluid_system.cache.density .= domain_fluid.density

        @testset verbose=true "plane normal $i" for i in eachindex(files)
            inflow = BoundaryZone(; plane=plane_boundary[i], boundary_type=InFlow(),
                                  plane_normal=plane_boundary_normal[i],
                                  open_boundary_layers=10, density=1000.0, particle_spacing)

            open_boundary = OpenBoundarySPHSystem(inflow; fluid_system,
                                                  boundary_model=BoundaryModelTafuni(),
                                                  buffer_size=0)

            semi = Semidiscretization(fluid_system, open_boundary)
            TrixiParticles.initialize_neighborhood_searches!(semi)

            v_open_boundary = zero(inflow.initial_condition.velocity)
            v_fluid = vcat(domain_fluid.velocity, domain_fluid.pressure')

            TrixiParticles.set_zero!(open_boundary.pressure)

            TrixiParticles.extrapolate_values!(open_boundary, v_open_boundary, v_fluid,
                                               inflow.initial_condition.coordinates,
                                               domain_fluid.coordinates, semi, 0.0;
                                               prescribed_pressure=false,
                                               prescribed_velocity=false)
            # Checked visually in ParaView:
            # trixi2vtk(fluid_system.initial_condition, filename="fluid",
            #           v=domain_fluid.velocity, p=domain_fluid.pressure)

            # trixi2vtk(open_boundary.initial_condition, filename="open_boundary",
            #           v=v_open_boundary, p=open_boundary.pressure)

            data = TrixiParticles.CSV.read(joinpath(validation_dir, files[i]),
                                           TrixiParticles.DataFrame)

            expected_velocity = vcat((data.var"v:0")',
                                     (data.var"v:1")',
                                     (data.var"v:2")')
            expected_pressure = data.var"p"

            @test isapprox(v_open_boundary, expected_velocity, atol=1e-2)
            @test isapprox(open_boundary.pressure, expected_pressure, atol=1e-2)
        end
    end
end
