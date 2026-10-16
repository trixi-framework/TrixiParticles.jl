@trixi_testset "Ghost reconstruction is independent of fluid correction caches" begin
    spacing = 0.1
    pressure(pos) = 2.0 + 3pos[1] - 2pos[2]
    particles = RectangularShape(spacing, (12, 12), (0.0, 0.0);
                                 density=1.0, pressure)
    for correction in (KernelCorrection(), GradientCorrection(), MixedKernelGradientCorrection())
        fluid = EntropicallyDampedSPHSystem(particles;
                                            smoothing_kernel=WendlandC6Kernel{2}(),
                                            smoothing_length=2spacing, sound_speed=10.0,
                                            density_calculator=ContinuityDensity(),
                                            correction, pressure_acceleration=nothing,
                                            buffer_size=0)
        zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.2]),
                            face_normal=(1.0, 0.0), boundary_type=OutFlow(),
                            density=1.0, particle_spacing=spacing,
                            average_inflow_velocity=false, open_boundary_layers=4)
        boundary = OpenBoundarySystem(zone; fluid_system=fluid, buffer_size=0)
        search = GridNeighborhoodSearch{2}(; update_strategy=SerialUpdate())
        semi = Semidiscretization(fluid, boundary; neighborhood_search=search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1); reset_threads=false)
        v, u = ode.u0.x
        fluid, boundary = ode.p.semi.systems
        TP = TrixiParticles
        TP.update_systems_and_nhs(v, u, ode.p.semi, 0.0)
        GC.@preserve v u begin
            vf, uf = TP.wrap_v(v, fluid, ode.p.semi), TP.wrap_u(u, fluid, ode.p.semi)
            vb, ub = TP.wrap_v(v, boundary, ode.p.semi), TP.wrap_u(u, boundary, ode.p.semi)
            expected = [pressure(TP.current_coords(ub, boundary, a))
                        for a in TP.each_integrated_particle(boundary)]
            TP.extrapolate_values!(boundary, FirstOrderMirroring(), vb, vf, ub, uf,
                                   ode.p.semi)
            @test boundary.cache.pressure ≈ expected atol=3e-12
            for name in (:kernel_correction_coefficient, :dw_gamma, :correction_matrix)
                hasproperty(fluid.cache, name) || continue
                fill!(getproperty(fluid.cache, name), NaN)
            end
            fill!(boundary.cache.pressure, 0.0)
            TP.extrapolate_values!(boundary, FirstOrderMirroring(), vb, vf, ub, uf,
                                   ode.p.semi)
            @test all(isfinite, boundary.cache.pressure)
            @test boundary.cache.pressure ≈ expected atol=3e-12
        end
    end
end
