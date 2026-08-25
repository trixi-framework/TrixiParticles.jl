@trixi_testset "IO Data" begin
    # Exported system data combines state-derived rigid kinematics with zero-valued force and
    # contact diagnostics before any interactions have run.
    coordinates = [-1.0 1.0
                   0.0 0.0]
    velocity = [0.0 0.0
                -1.0 1.0]
    mass = [1.0, 1.0]
    density = [1000.0, 1000.0]

    initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                         particle_spacing=1.0)
    rigid_system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0))

    semi = Semidiscretization(rigid_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    dv_ode = zeros(eltype(v_ode), size(v_ode))
    du_ode = zeros(eltype(u_ode), size(u_ode))

    v = TrixiParticles.wrap_v(v_ode, rigid_system, semi)
    u = TrixiParticles.wrap_u(u_ode, rigid_system, semi)
    TrixiParticles.update_final!(rigid_system, v, u, v_ode, u_ode, semi, 0.0)

    data = TrixiParticles.system_data(rigid_system, dv_ode, du_ode,
                                      v_ode, u_ode, semi)
    fields = TrixiParticles.available_data(rigid_system)

    @test data.center_of_mass == [0.0, 0.0]
    @test data.center_of_mass_velocity == [0.0, 0.0]
    @test data.angular_velocity == 1.0
    @test data.resultant_force == [0.0, 0.0]
    @test data.resultant_torque == 0.0
    @test data.angular_acceleration_force == 0.0
    @test data.gyroscopic_acceleration == 0.0
    @test data.contact_count == 0
    @test data.max_contact_penetration == 0.0
    @test data.relative_coordinates == rigid_system.relative_coordinates
    @test :contact_count in fields
    @test :max_contact_penetration in fields
    @test !(:local_coordinates in fields)
end

@trixi_testset "Restart" begin
    # Restart replaces the ODE initial arrays immediately, while derived kinematics and force
    # caches remain unchanged until the normal update lifecycle runs.
    coordinates = [0.0 1.0 2.0
                   0.0 0.0 0.0]
    velocity = [0.0 0.0 0.0
                0.0 0.0 0.0]
    mass = [1.0, 1.0, 1.0]
    density = [1000.0, 1000.0, 1000.0]

    initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                         particle_spacing=1.0)
    rigid_system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0))

    u_new = [2.0 4.0 6.0
             3.0 3.0 3.0]
    v_new = [1.0 2.0 3.0
             4.0 5.0 6.0]

    TrixiParticles.update_final!(rigid_system, rigid_system.initial_velocity,
                                 rigid_system.initial_condition.coordinates,
                                 nothing, nothing, nothing, 0.0)
    stale_relative_coordinates = copy(rigid_system.relative_coordinates)
    stale_center_of_mass = rigid_system.center_of_mass[]
    stale_center_of_mass_velocity = rigid_system.center_of_mass_velocity[]
    stale_angular_velocity = rigid_system.angular_velocity[]
    stale_force = SVector(7.0, -11.0)
    stale_torque = 5.0
    stale_angular_acceleration_force = 2.0
    rigid_system.resultant_force[] = stale_force
    rigid_system.resultant_torque[] = stale_torque
    rigid_system.angular_acceleration_force[] = stale_angular_acceleration_force

    restarted_system = TrixiParticles.restart_with!(rigid_system, v_new, u_new)

    @test restarted_system === rigid_system
    @test rigid_system.initial_condition.coordinates == u_new
    @test rigid_system.initial_condition.velocity == v_new
    @test rigid_system.initial_velocity == v_new
    @test rigid_system.relative_coordinates == stale_relative_coordinates
    @test rigid_system.center_of_mass[] == stale_center_of_mass
    @test rigid_system.center_of_mass_velocity[] == stale_center_of_mass_velocity
    @test rigid_system.angular_velocity[] == stale_angular_velocity
    @test rigid_system.resultant_force[] == stale_force
    @test rigid_system.resultant_torque[] == stale_torque
    @test rigid_system.angular_acceleration_force[] == stale_angular_acceleration_force

    # Timestep estimation must use the restarted state even before cache refresh; afterwards,
    # the refreshed center-of-mass and rotation values must describe that same state.
    expected_center_of_mass = [4.0, 3.0]
    expected_relative_coordinates = u_new .- expected_center_of_mass
    semi = Semidiscretization(rigid_system, neighborhood_search=nothing)
    dt_restarted = TrixiParticles.calculate_dt(v_new, u_new, 0.25, rigid_system, semi)

    TrixiParticles.update_final!(rigid_system, v_new, u_new, nothing, nothing, semi,
                                 0.0)
    dt_updated = TrixiParticles.calculate_dt(v_new, u_new, 0.25, rigid_system, semi)

    @test rigid_system.center_of_mass[] == expected_center_of_mass
    @test rigid_system.relative_coordinates == expected_relative_coordinates
    @test rigid_system.center_of_mass_velocity[] == [2.0, 5.0]
    @test rigid_system.angular_velocity[] == 0.5
    @test isapprox(dt_restarted, dt_updated)
end

@trixi_testset "Velocity Components with ContinuityDensity" begin
    # Hydrodynamic density adds an ODE component, but exported velocity and acceleration must
    # still contain exactly the physical spatial dimensions.
    coordinates = [0.0 0.1
                   0.0 0.0]
    mass = [1.0, 1.0]
    density = [1000.0, 1000.0]
    initial_condition = InitialCondition(; coordinates, mass, density)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 0.12
    boundary_model = BoundaryModelDummyParticles(density, mass,
                                                 ContinuityDensity(),
                                                 smoothing_kernel,
                                                 smoothing_length)

    rigid_system = RigidBodySystem(initial_condition; boundary_model)
    semi = Semidiscretization(rigid_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    dv_ode = zeros(eltype(v_ode), size(v_ode))
    du_ode = zeros(eltype(u_ode), size(u_ode))

    data = TrixiParticles.system_data(rigid_system, dv_ode, du_ode,
                                      v_ode, u_ode, semi)

    @test size(data.velocity, 1) == ndims(rigid_system)
    @test size(data.acceleration, 1) == ndims(rigid_system)
end

@trixi_testset "Configuration" begin
    # Rigid bodies are boundaries, not fluids: reject unsupported fluid-rigid and surface
    # tension combinations during semidiscretization instead of failing in the RHS.
    coordinates = [1.0 2.0
                   1.0 2.0]
    mass = [1.0, 1.0]
    density = [1000.0, 1000.0]

    rigid_ic = InitialCondition(; coordinates, mass, density)
    rigid_system = RigidBodySystem(rigid_ic)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 0.12
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=7.0)
    fluid_system = WeaklyCompressibleSPHSystem(rigid_ic; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=SummationDensity(),
                                               state_equation)

    @test_throws ArgumentError Semidiscretization(fluid_system, rigid_system)

    rigid_boundary_model = BoundaryModelDummyParticles(density, mass,
                                                       SummationDensity(),
                                                       smoothing_kernel,
                                                       smoothing_length)
    rigid_system_with_dummy = RigidBodySystem(rigid_ic;
                                              boundary_model=rigid_boundary_model)
    fluid_with_surface_tension = WeaklyCompressibleSPHSystem(rigid_ic;
                                                             smoothing_kernel,
                                                             smoothing_length,
                                                             density_calculator=SummationDensity(),
                                                             state_equation,
                                                             surface_tension=SurfaceTensionMorris(surface_tension_coefficient=0.072),
                                                             reference_particle_spacing=0.1)

    @test_throws ArgumentError Semidiscretization(fluid_with_surface_tension,
                                                  rigid_system_with_dummy)
end
