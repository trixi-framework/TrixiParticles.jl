@testset verbose=true "Calculate Flow Rate" begin
    particle_spacing = 0.01

    # Define a parabolic velocity profile
    velocity_function(pos) = [4 * (pos[2] - 0.5) * (1.5 - pos[2]), 0.0]

    # Create fluid domain with the specified velocity profile
    n_particles_y = round(Int, 2 / particle_spacing)
    fluid = RectangularShape(particle_spacing, (4, n_particles_y), (0.0, 0.0),
                             density=1000.0, velocity=velocity_function)

    smoothing_length = 1.3 * particle_spacing
    smoothing_kernel = WendlandC2Kernel{ndims(fluid)}()
    fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                               1.0)
    fluid_system.cache.density .= fluid.density

    # Use a smaller cross-sectional area to test user-defined area functionality
    # and to perform interpolation within an embedded domain.
    # (In a simulation with solid boundaries, wall velocities would also be included.)
    sample_points = RectangularShape(particle_spacing, (1, round(Int, n_particles_y / 2)),
                                     (0.0, 0.5), density=1.0).coordinates

    zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 2.0]),
                        face_normal=(-1.0, 0.0), open_boundary_layers=10, density=1000.0,
                        particle_spacing, sample_points=sample_points,
                        reference_velocity=(pos, t) -> velocity_function(pos))

    open_boundary = OpenBoundarySystem(zone; fluid_system,
                                       boundary_model=BoundaryModelMirroringTafuni(),
                                       calculate_flow_rate=true, buffer_size=0)

    semi = Semidiscretization(fluid_system, open_boundary)
    TrixiParticles.initialize_neighborhood_searches!(semi)
    TrixiParticles.initialize!(open_boundary, semi)

    # Set up ODE for initial conditions
    ode = semidiscretize(semi, (0, 1))
    v_ode, u_ode = ode.u0.x

    boundary_zone = first(open_boundary.boundary_zones)

    @testset verbose=true "Velocity Interpolation" begin
        TrixiParticles.interpolate_velocity!(open_boundary, boundary_zone,
                                             v_ode, u_ode, semi)

        coords = reinterpret(reshape, SVector{2, Float64},
                             boundary_zone.cache.sample_points)
        vels_analytic = first.(velocity_function.(coords))
        vels_interpolated = first.(reinterpret(reshape, SVector{2, Float64},
                                               boundary_zone.cache.sample_velocity))

        @test isapprox(vels_interpolated, vels_analytic, rtol=1e-3)
    end

    @testset verbose=true "Flow Rate Calculation" begin
        TrixiParticles.calculate_flow_rate!(open_boundary, v_ode, u_ode, semi)

        Q_analytic = zone.cache.cross_sectional_area * (2 / 3)
        Q_calculated = first(open_boundary.cache.boundary_zones_flow_rate)[]

        @test isapprox(Q_analytic, Q_calculated; rtol=1e-3)
    end
end
