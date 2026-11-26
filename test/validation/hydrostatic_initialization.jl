@trixi_testset "Reference-consistent hydrostatic summation density" begin
    for (shepard, reverse_order) in ((false, false), (true, false), (true, true))
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fsi",
                                                  "hydrostatic_water_column_2d.jl");
                                         use_edac=false, n_particles_plate_y=3,
                                         fluid_density_calculator=SummationDensity(),
                                         density_correction=shepard ?
                                                            ShepardKernelCorrection() :
                                                            nothing,
                                         fluid_gradient_correction=MixedKernelGradientCorrection(),
                                         boundary_gradient_correction=MixedKernelGradientCorrection(),
                                         density_diffusion=nothing,
                                         reverse_system_order=reverse_order,
                                         update_strategy=SerialUpdate(),
                                         parallelization_backend=SerialBackend(),
                                         sol=nothing) [
            r"\[ Info: To create the self-interaction neighborhood search.*\n"
        ]
        semi = ode.p.semi
        fluid = only(filter(system -> system isa WeaklyCompressibleSPHSystem, semi.systems))
        solid = only(filter(system -> system isa TotalLagrangianSPHSystem, semi.systems))
        @test summation_initialization.policy == "operator-offset-hydrostatic"
        @test summation_initialization.relative_density_error < 1e-12
        @test summation_initialization.maximum_pressure_error < 1e-6
        @test fluid.mass == tank.fluid.mass == fluid.initial_condition.mass
        @test solid.mass == solid.initial_condition.mass
        @test all(>(0), fluid.mass)
        @test all(iszero, ode.u0.x[1])
        @test fluid.initial_condition.coordinates == tank.fluid.coordinates
        @test TrixiParticles.density_reference_initialized(fluid)

        v, u = ode.u0.x
        density_before = copy(fluid.cache.density)
        pressure_before = copy(fluid.pressure)
        offsets = copy(fluid.cache.initial_density_offset)
        reference = copy(fluid.cache.initial_filtered_density_offset)
        dv = zero(v)
        TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        @test fluid.cache.density == density_before
        @test fluid.pressure == pressure_before
        @test maximum(abs, TrixiParticles.wrap_v(dv, fluid, semi)) < 10

        for system in semi.systems
            system isa WeaklyCompressibleSPHSystem && continue
            system.boundary_model.cache.density .= 700
            system.boundary_model.pressure .= -1234
        end
        TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        @test fluid.cache.density == density_before
        @test fluid.pressure == pressure_before

        # The reference corrects initialization, rather than fixing density in time.
        original_coordinates = copy(u)
        fluid_u = TrixiParticles.wrap_u(u, fluid, semi)
        fluid_u[1, div(nparticles(fluid), 2)] += 0.001
        TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        @test maximum(abs.(fluid.cache.density-density_before)) > 0.01
        copyto!(u, original_coordinates)
        TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        @test fluid.cache.density == density_before
        @test fluid.pressure == pressure_before
        @test fluid.cache.initial_density_offset == offsets
        @test fluid.cache.initial_filtered_density_offset == reference

        # An in-memory restart retains the original material reference, including
        # the Shepard reference, rather than recalibrating at the displaced state.
        fluid_u[1, div(nparticles(fluid), 2)] += 0.001
        TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        restart_density = copy(fluid.cache.density)
        TrixiParticles.restart_with!(fluid, TrixiParticles.wrap_v(v, fluid, semi),
                                     TrixiParticles.wrap_u(u, fluid, semi))
        restarted = semidiscretize(semi, ode.tspan; reset_threads=false)
        restarted_fluid = only(filter(system -> system isa WeaklyCompressibleSPHSystem,
                                      restarted.p.semi.systems))
        @test restarted_fluid.cache.initial_density_offset == offsets
        @test restarted_fluid.cache.initial_filtered_density_offset == reference
        TrixiParticles.kick!(zero(restarted.u0.x[1]), restarted.u0.x...,
                             (; semi=restarted.p.semi, split_integration_data=nothing), 0.0)
        @test restarted_fluid.cache.density ≈ restart_density rtol=1e-13

        if shepard && reverse_order
            directory = mktempdir()
            trixi2vtk(restarted.u0, restarted.p.semi, 0.0; iter=0,
                      output_directory=directory)
            file = only(filter(path -> startswith(basename(path), "fluid") &&
                                       endswith(path, ".vtu"),
                               readdir(directory; join=true)))
            restarted_fluid.cache.initial_density_offset .= 0
            restarted_fluid.cache.initial_filtered_density_offset .= 0
            restarted_fluid.cache.initial_density_reference_ready[] = false
            TrixiParticles.restore_previous_state!(restarted_fluid, file)
            @test TrixiParticles.density_reference_initialized(restarted_fluid)
            @test restarted_fluid.cache.initial_density_offset == offsets
            @test restarted_fluid.cache.initial_filtered_density_offset == reference
        end
    end
    @test_throws ArgumentError WeaklyCompressibleSPHSystem(fluid_system.initial_condition;
                                                           smoothing_kernel,
                                                           smoothing_length=smoothing_length_fluid,
                                                           state_equation,
                                                           density_calculator=ContinuityDensity(),
                                                           initial_density_correction=true)
    @test_throws ArgumentError WeaklyCompressibleSPHSystem(fluid_system.initial_condition;
                                                           smoothing_kernel,
                                                           smoothing_length=smoothing_length_fluid,
                                                           state_equation,
                                                           density_calculator=SummationDensity(),
                                                           buffer_size=1,
                                                           initial_density_correction=true)

    # Measuring the density operator must not apply the EOS to an inadmissible
    # temporary, uncorrected boundary state before its reference offset is known.
    spacing = 0.025
    kernel = WendlandC2Kernel{2}()
    h = sqrt(2) * spacing
    equation = StateEquationCole(; sound_speed=50.0, reference_density=1000.0, exponent=7)
    initial = RectangularShape(spacing, (1, 1), (0.025, 0.0); density=1000.0)
    geometry = RectangularShape(spacing, (1, 1), (0.0, 0.06); density=1000.0)
    reference_fluid = WeaklyCompressibleSPHSystem(initial; smoothing_kernel=kernel,
                                                  smoothing_length=h,
                                                  density_calculator=SummationDensity(),
                                                  state_equation=equation,
                                                  density_correction=ShepardKernelCorrection(),
                                                  initial_density_correction=true,
                                                  acceleration=(0.0, -9.81))
    model = BoundaryModelDummyParticles(geometry.density, geometry.mass,
                                        AdamiPressureExtrapolation(), kernel, h;
                                        state_equation=equation)
    reference_ode = semidiscretize(Semidiscretization(reference_fluid,
                                                      WallBoundarySystem(geometry, model);
                                                      neighborhood_search=nothing,
                                                      parallelization_backend=SerialBackend()),
                                   (0.0, 1.0); reset_threads=false)
    initialized = reference_ode.p.semi.systems[1]
    @test only(initialized.cache.density) == 1000.0
    @test only(initialized.pressure) == 0.0
    @test all(isfinite, initialized.cache.initial_filtered_density_offset)
end
