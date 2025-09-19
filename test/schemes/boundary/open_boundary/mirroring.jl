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

        boundary_faces = [
            ([0.0, 0.0], [0.0, domain_length]),
            ([0.0, 0.0], [-domain_length, domain_length])
        ]
        boundary_face_normal = [[1.0, 0.0], [1.0, 1.0]]

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

        @testset verbose=true "face normal $i" for i in eachindex(files)
            inflow = BoundaryZone(; boundary_face=boundary_faces[i], boundary_type=InFlow(),
                                  face_normal=boundary_face_normal[i],
                                  average_inflow_velocity=false,
                                  open_boundary_layers=10, density=1000.0, particle_spacing)

            open_boundary = OpenBoundarySystem(inflow; fluid_system,
                                               boundary_model=BoundaryModelMirroringTafuni(),
                                               buffer_size=0)

            semi = Semidiscretization(fluid_system, open_boundary)
            TrixiParticles.initialize_neighborhood_searches!(semi)
            TrixiParticles.initialize!(open_boundary, semi)

            v_open_boundary = zero(inflow.initial_condition.velocity)
            v_fluid = vcat(domain_fluid.velocity, domain_fluid.pressure')

            TrixiParticles.set_zero!(open_boundary.cache.pressure)

            TrixiParticles.extrapolate_values!(open_boundary, FirstOrderMirroring(),
                                               v_open_boundary, v_fluid,
                                               inflow.initial_condition.coordinates,
                                               domain_fluid.coordinates, semi)
            # Checked visually in ParaView:
            # trixi2vtk(fluid_system.initial_condition, filename="fluid",
            #           v=domain_fluid.velocity, p=domain_fluid.pressure)

            # trixi2vtk(open_boundary.initial_condition, filename="open_boundary",
            #           v=v_open_boundary, p=open_boundary.cache.pressure)

            data = TrixiParticles.CSV.read(joinpath(validation_dir, files[i]),
                                           TrixiParticles.DataFrame)

            expected_velocity = vcat((data.var"v:0")',
                                     (data.var"v:1")')
            expected_pressure = data.var"p"

            @test isapprox(v_open_boundary, expected_velocity, atol=1e-3)
            @test isapprox(open_boundary.cache.pressure, expected_pressure, atol=1e-3)
        end
    end

    @testset verbose=true "3D" begin
        files = [
            "open_boundary_extrapolated_3d.csv",
            "open_boundary_extrapolated_skew_3d.csv"
        ]

        particle_spacing = 0.05
        domain_length = 1.0

        boundary_faces = [
            ([0.0, 0.0, 0.0], [domain_length, 0.0, 0.0], [0.0, domain_length, 0.0]),
            ([0.0, 0.0, 0.0], [domain_length, 0.0, 0.0],
             [0.0, domain_length, domain_length])
        ]
        boundary_face_normal = [[0.0, 0.0, 1.0], [0.0, -1.0, 1.0]]

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

        @testset verbose=true "face normal $i" for i in eachindex(files)
            inflow = BoundaryZone(; boundary_face=boundary_faces[i], boundary_type=InFlow(),
                                  face_normal=boundary_face_normal[i],
                                  average_inflow_velocity=false,
                                  open_boundary_layers=10, density=1000.0, particle_spacing)

            open_boundary = OpenBoundarySystem(inflow; fluid_system,
                                               boundary_model=BoundaryModelMirroringTafuni(),
                                               buffer_size=0)

            semi = Semidiscretization(fluid_system, open_boundary)
            TrixiParticles.initialize_neighborhood_searches!(semi)
            TrixiParticles.initialize!(open_boundary, semi)

            v_open_boundary = zero(inflow.initial_condition.velocity)
            v_fluid = vcat(domain_fluid.velocity, domain_fluid.pressure')

            TrixiParticles.set_zero!(open_boundary.cache.pressure)

            TrixiParticles.extrapolate_values!(open_boundary, FirstOrderMirroring(),
                                               v_open_boundary, v_fluid,
                                               inflow.initial_condition.coordinates,
                                               domain_fluid.coordinates, semi)
            # Checked visually in ParaView:
            # trixi2vtk(fluid_system.initial_condition, filename="fluid",
            #           v=domain_fluid.velocity, p=domain_fluid.pressure)

            # trixi2vtk(open_boundary.initial_condition, filename="open_boundary",
            #           v=v_open_boundary, p=open_boundary.cache.pressure)

            data = TrixiParticles.CSV.read(joinpath(validation_dir, files[i]),
                                           TrixiParticles.DataFrame)

            expected_velocity = vcat((data.var"v:0")',
                                     (data.var"v:1")',
                                     (data.var"v:2")')
            expected_pressure = data.var"p"

            @test isapprox(v_open_boundary, expected_velocity, atol=1e-2)
            @test isapprox(open_boundary.cache.pressure, expected_pressure, atol=1e-2)
        end
    end

    @testset verbose=true "Average Inflow Velocity $i-D" for i in (2, 3)
        particle_spacing = 0.05
        domain_length = 1.0
        open_boundary_layers = 40

        n_particles_xy = round(Int, domain_length / particle_spacing)

        if i == 2
            domain_fluid = RectangularShape(particle_spacing, (2, 1) .* n_particles_xy,
                                            (0.0, 0.0), density=1000.0,
                                            velocity=x -> SVector{2}(x[1], 0.0))

        else
            domain_fluid = RectangularShape(particle_spacing, (2, 1, 1) .* n_particles_xy,
                                            (0.0, 0.0, 0.0), density=1000.0,
                                            velocity=x -> SVector{3}(x[1], 0.0, 0.0))
        end

        smoothing_length = 1.5 * particle_spacing
        smoothing_kernel = WendlandC2Kernel{ndims(domain_fluid)}()
        fluid_system = EntropicallyDampedSPHSystem(domain_fluid, smoothing_kernel,
                                                   smoothing_length, 1.0)
        fluid_system.cache.density .= 1000.0

        if i == 2
            face_in = ([0.0, 0.0], [0.0, domain_length])
        else
            face_in = ([0.0, 0.0, 0.0], [0.0, domain_length, 0.0],
                       [0.0, 0.0, domain_length])
        end

        inflow = BoundaryZone(; boundary_face=face_in, boundary_type=InFlow(),
                              face_normal=(i == 2 ? [1.0, 0.0] : [1.0, 0.0, 0.0]),
                              open_boundary_layers=open_boundary_layers, density=1000.0,
                              particle_spacing, average_inflow_velocity=true)
        open_boundary_in = OpenBoundarySystem(inflow; fluid_system,
                                              boundary_model=BoundaryModelMirroringTafuni(),
                                              buffer_size=0)

        semi = Semidiscretization(fluid_system, open_boundary_in)
        TrixiParticles.initialize_neighborhood_searches!(semi)
        TrixiParticles.initialize!(open_boundary_in, semi)

        v_open_boundary = zero(inflow.initial_condition.velocity)
        u_open_boundary = inflow.initial_condition.coordinates
        v_fluid = vcat(domain_fluid.velocity, domain_fluid.pressure')

        TrixiParticles.set_zero!(open_boundary_in.cache.pressure)

        TrixiParticles.extrapolate_values!(open_boundary_in, FirstOrderMirroring(),
                                           v_open_boundary, v_fluid,
                                           inflow.initial_condition.coordinates,
                                           domain_fluid.coordinates, semi)

        TrixiParticles.average_velocity!(v_open_boundary, u_open_boundary, open_boundary_in,
                                         first(open_boundary_in.boundary_zones), semi)

        # Since the velocity profile increases linearly in positive x-direction,
        # we can use the first velocity entry as a representative value.
        v_x_fluid_first = v_fluid[1, 1]

        @test all(isapprox(-v_x_fluid_first), v_open_boundary[1, :])
    end

    @testset verbose=true "Mirroring Methods" begin
        function mirror(pressure_function, mirror_method;
                        particle_spacing=0.05, domain_size=(2.0, 1.0))
            # Initialize a fluid block with pressure according to `pressure_function`
            # and a adjacent inflow and outflow open boundaries to test the pressure extrapolation.
            domain_fluid = RectangularShape(particle_spacing,
                                            round.(Int, domain_size ./ particle_spacing),
                                            (0.0, 0.0), density=1000.0,
                                            pressure=pressure_function)

            smoothing_length = 1.2 * particle_spacing
            smoothing_kernel = WendlandC2Kernel{2}()
            fluid_system = EntropicallyDampedSPHSystem(domain_fluid, smoothing_kernel,
                                                       smoothing_length, 1.0)

            fluid_system.cache.density .= domain_fluid.density

            face_out = ([domain_size[1], 0.0], [domain_size[1], domain_size[2]])

            outflow = BoundaryZone(; boundary_face=face_out, boundary_type=OutFlow(),
                                   face_normal=[-1.0, 0.0],
                                   open_boundary_layers=10, density=1000.0,
                                   particle_spacing)
            open_boundary_out = OpenBoundarySystem(outflow; fluid_system,
                                                   boundary_model=BoundaryModelMirroringTafuni(),
                                                   buffer_size=0)

            # Temporary semidiscretization just to extrapolate the pressure into the outflow system
            semi = Semidiscretization(fluid_system, open_boundary_out)
            TrixiParticles.initialize_neighborhood_searches!(semi)
            TrixiParticles.initialize!(open_boundary_out, semi)

            v_open_boundary = zero(outflow.initial_condition.velocity)
            v_fluid = vcat(domain_fluid.velocity, domain_fluid.pressure')

            TrixiParticles.set_zero!(open_boundary_out.cache.pressure)

            TrixiParticles.extrapolate_values!(open_boundary_out, mirror_method,
                                               v_open_boundary, v_fluid,
                                               outflow.initial_condition.coordinates,
                                               domain_fluid.coordinates, semi)

            face_in = ([0.0, 0.0], [0.0, domain_size[2]])

            inflow = BoundaryZone(; boundary_face=face_in, boundary_type=InFlow(),
                                  face_normal=[1.0, 0.0],
                                  open_boundary_layers=10, density=1000.0, particle_spacing)
            open_boundary_in = OpenBoundarySystem(inflow; fluid_system,
                                                  boundary_model=BoundaryModelMirroringTafuni(),
                                                  buffer_size=0)

            # Temporary semidiscretization just to extrapolate the pressure into the outflow system
            semi = Semidiscretization(fluid_system, open_boundary_in)
            TrixiParticles.initialize_neighborhood_searches!(semi)
            TrixiParticles.initialize!(open_boundary_in, semi)

            v_open_boundary = zero(inflow.initial_condition.velocity)

            TrixiParticles.set_zero!(open_boundary_in.cache.pressure)

            TrixiParticles.extrapolate_values!(open_boundary_in, mirror_method,
                                               v_open_boundary, v_fluid,
                                               inflow.initial_condition.coordinates,
                                               domain_fluid.coordinates, semi)

            return fluid_system, open_boundary_in, open_boundary_out, v_fluid
        end

        function interpolate_pressure(mirror_method, pressure_func; particle_spacing=0.05)
            # First call the function above to initialize fluid with pressure according to the function
            # and then extrapolate pressure to the inflow and outflow boundary systems.
            # Then, in this function, we apply an SPH interpolation on this extrapolated pressure field
            # to get a continuous representation of the extrapolated pressure field to validate.
            fluid_system, open_boundary_in, open_boundary_out,
            v_fluid = mirror(pressure_func, mirror_method)

            p_fluid = [TrixiParticles.current_pressure(v_fluid, fluid_system, particle)
                       for particle in TrixiParticles.eachparticle(fluid_system)]

            fluid_system.initial_condition.pressure .= p_fluid
            open_boundary_in.initial_condition.pressure .= open_boundary_in.cache.pressure
            open_boundary_out.initial_condition.pressure .= open_boundary_out.cache.pressure

            entire_domain = union(fluid_system.initial_condition,
                                  open_boundary_in.initial_condition,
                                  open_boundary_out.initial_condition)

            smoothing_length = 1.2 * particle_spacing
            smoothing_kernel = WendlandC2Kernel{2}()

            # Use a temporary fluid system just to interpolate the pressure
            interpolation_system = WeaklyCompressibleSPHSystem(entire_domain,
                                                               ContinuityDensity(),
                                                               nothing, smoothing_kernel,
                                                               smoothing_length)
            interpolation_system.pressure .= entire_domain.pressure

            semi = Semidiscretization(interpolation_system)
            ode = semidiscretize(semi, (0, 0))
            v_ode, u_ode = ode.u0.x

            result = interpolate_line([-0.5, 0.5], [2.5, 0.5], 50, semi,
                                      interpolation_system, v_ode, u_ode)

            return result.pressure
        end

        pressure_func(pos) = cos(2pi * pos[1])

        # The pressures are interpolated to obtain a unified vector of length 50,
        # rather than handling three separate systems with numerous particles each.
        # Additionally, it facilitates plotting for test validation purposes.
        pressures = interpolate_pressure.([
                                              SimpleMirroring(),
                                              FirstOrderMirroring(),
                                              ZerothOrderMirroring()
                                          ], pressure_func)

        pressures_expected = [
            [
                -0.961368753262176,
                -0.889650754605612,
                -0.6905728633912162,
                -0.3906982503900935,
                -0.03185103264546474,
                0.333177250635236,
                0.6476757904268203,
                0.8708718636029472,
                0.9770644543539139,
                0.930127404224564,
                0.7453239657417114,
                0.45327391778794734,
                0.0957991316946515,
                -0.278044550109137,
                -0.6122210347366616,
                -0.8549953877169134,
                -0.972402370920352,
                -0.9485947076098263,
                -0.7852083273404225,
                -0.5067072947529646,
                -0.15639554738985995,
                0.216475912074678,
                0.5602009648581898,
                0.8223333154520063,
                0.9626909343468545,
                0.9626909343468544,
                0.8223333154520058,
                0.5602009648581916,
                0.2164759120746806,
                -0.15639554738985995,
                -0.5067072947529652,
                -0.7852083273404221,
                -0.9485947076098261,
                -0.9724023709203522,
                -0.8549953877169136,
                -0.6122210347366618,
                -0.2780445501091364,
                0.0957991316946491,
                0.45327391778794657,
                0.7453239657417114,
                0.9301274042245636,
                0.9770644543539139,
                0.8708718636029484,
                0.647675790426819,
                0.33317725063523634,
                -0.03185103264546614,
                -0.3906982503900926,
                -0.6905728633912179,
                -0.8896507546056125,
                -0.9613687532621761
            ],
            [
                0.06915008702843263,
                1.0737102752866752,
                2.4308935813709276,
                3.074251266025277,
                3.047730973768249,
                2.5546812207882246,
                1.8827468502566729,
                1.330793458342126,
                1.0685482295301676,
                0.9337092788362713,
                0.7453239657417114,
                0.45327391778794734,
                0.0957991316946515,
                -0.278044550109137,
                -0.6122210347366616,
                -0.8549953877169134,
                -0.972402370920352,
                -0.9485947076098263,
                -0.7852083273404225,
                -0.5067072947529646,
                -0.15639554738985995,
                0.216475912074678,
                0.5602009648581898,
                0.8223333154520063,
                0.9626909343468545,
                0.9626909343468544,
                0.8223333154520058,
                0.5602009648581916,
                0.2164759120746806,
                -0.15639554738985995,
                -0.5067072947529652,
                -0.7852083273404221,
                -0.9485947076098261,
                -0.9724023709203522,
                -0.8549953877169136,
                -0.6122210347366618,
                -0.2780445501091364,
                0.0957991316946491,
                0.45327391778794657,
                0.7453239657417114,
                0.9337092788362712,
                1.0685482295301671,
                1.3307934583421233,
                1.8827468502566747,
                2.55468122078822,
                3.047730973768248,
                3.0742512660252768,
                2.4308935813709227,
                1.0737102752866743,
                0.06915008702844015
            ],
            [
                -0.961368753262176,
                -0.8896507546056119,
                -0.690572863391216,
                -0.3906982503900935,
                -0.03185103264546471,
                0.333177250635236,
                0.6476223100256845,
                0.8652532179882182,
                0.9638320291788572,
                0.929273656897224,
                0.7453239657417114,
                0.45327391778794734,
                0.0957991316946515,
                -0.278044550109137,
                -0.6122210347366616,
                -0.8549953877169134,
                -0.972402370920352,
                -0.9485947076098263,
                -0.7852083273404225,
                -0.5067072947529646,
                -0.15639554738985995,
                0.216475912074678,
                0.5602009648581898,
                0.8223333154520063,
                0.9626909343468545,
                0.9626909343468544,
                0.8223333154520058,
                0.5602009648581916,
                0.2164759120746806,
                -0.15639554738985995,
                -0.5067072947529652,
                -0.7852083273404221,
                -0.9485947076098261,
                -0.9724023709203522,
                -0.8549953877169136,
                -0.6122210347366618,
                -0.2780445501091364,
                0.0957991316946491,
                0.45327391778794657,
                0.7453239657417114,
                0.9292736568972241,
                0.9638320291788572,
                0.865253217988219,
                0.6476223100256829,
                0.33317725063523645,
                -0.03185103264546612,
                -0.3906982503900927,
                -0.6905728633912182,
                -0.8896507546056123,
                -0.961368753262176
            ]
        ]

        @testset verbose=true "$method" for (i, method) in enumerate(("simple mirroring",
                                                       "first order mirroring",
                                                       "zeroth order mirroring"))
            @test isapprox(pressures[i], pressures_expected[i])
        end
    end

    @testset verbose=true "integrated variables $(n_dims)D" for n_dims in (2, 3)
        particle_spacing = 0.1
        initial_condition = rectangular_patch(particle_spacing, ntuple(_ -> 2, n_dims))

        boundary_face = n_dims == 2 ? ([0.0, 0.0], [0.0, 1.0]) :
                        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0])
        face_normal = n_dims == 2 ? [1.0, 0.0] : [1.0, 0.0, 0.0]
        inflow = BoundaryZone(; boundary_face, boundary_type=InFlow(), face_normal,
                              open_boundary_layers=10, density=1.0, particle_spacing)

        system_wcsph = WeaklyCompressibleSPHSystem(initial_condition, ContinuityDensity(),
                                                   nothing,
                                                   SchoenbergCubicSplineKernel{n_dims}(), 1)

        open_boundary_wcsph = OpenBoundarySystem(inflow; fluid_system=system_wcsph,
                                                 buffer_size=0,
                                                 boundary_model=BoundaryModelMirroringTafuni())

        @test TrixiParticles.v_nvariables(open_boundary_wcsph) == n_dims

        system_edac_1 = EntropicallyDampedSPHSystem(initial_condition,
                                                    SchoenbergCubicSplineKernel{n_dims}(),
                                                    1.0,
                                                    1.0)

        open_boundary_edac_1 = OpenBoundarySystem(inflow; fluid_system=system_edac_1,
                                                  buffer_size=0,
                                                  boundary_model=BoundaryModelMirroringTafuni())

        @test TrixiParticles.v_nvariables(open_boundary_edac_1) == n_dims
    end
end
