@trixi_testset "Constructor" begin
    # Construction copies immutable input data but leaves state-derived rigid kinematics at
    # zero until the first state update.
    coordinates = [1.0 2.0 3.0
                   1.0 2.0 3.0]
    mass = [1.25, 1.5, 1.75]
    material_densities = [990.0, 995.0, 1000.0]

    initial_condition = InitialCondition(; coordinates, mass,
                                         density=material_densities)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 0.12
    boundary_model = BoundaryModelDummyParticles(material_densities, mass,
                                                 SummationDensity(),
                                                 smoothing_kernel,
                                                 smoothing_length)

    system = RigidBodySystem(initial_condition; boundary_model,
                             acceleration=(0.0, -9.81), particle_spacing=0.1)

    @test ndims(system) == 2
    @test system.initial_condition == initial_condition
    @test all(iszero, system.relative_coordinates)
    @test system.mass == mass
    @test system.material_density == material_densities
    @test system.initial_velocity == initial_condition.velocity
    @test system.acceleration == [0.0, -9.81]
    @test iszero(system.center_of_mass[])
    @test iszero(system.center_of_mass_velocity[])
    @test iszero(system.angular_velocity[])
    @test system.particle_spacing == 0.1
    @test system.boundary_model == boundary_model
    @test system.adhesion_coefficient == 0.0
    @test TrixiParticles.v_nvariables(system) == 2

    semi = Semidiscretization(system, neighborhood_search=nothing)
    system = semi.systems[1]
    ode = semidiscretize(semi, (0.0, 0.0); reset_threads=false)

    # A stationary body without contact has no finite stability restriction of its own.
    dt = TrixiParticles.calculate_dt(ode.u0.x[1], ode.u0.x[2], 0.25, system, semi)
    @test isinf(dt)
end

@trixi_testset "Show" begin
    # A system without contact should omit contact caches and keep both display formats
    # focused on its active boundary model.
    coordinates = [1.0 2.0
                   1.0 2.0]
    mass = [1.25, 1.5]
    material_densities = [990.0, 1000.0]

    initial_condition = InitialCondition(; coordinates, mass,
                                         density=material_densities)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 0.12
    boundary_model = BoundaryModelDummyParticles(material_densities, mass,
                                                 SummationDensity(),
                                                 smoothing_kernel,
                                                 smoothing_length)

    system = RigidBodySystem(initial_condition; boundary_model,
                             acceleration=(0.0, -9.81))
    @test !haskey(system.cache, :contact_manifold_count)

    show_compact = "RigidBodySystem{2}([0.0, -9.81], BoundaryModelDummyParticles(SummationDensity, Nothing)) with 2 particles"
    @test repr(system) == show_compact

    show_box = """
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ RigidBodySystem{2}                                                                               │
    │ ══════════════════                                                                               │
    │ #particles: ………………………………………………… 2                                                                │
    │ acceleration: …………………………………………… [0.0, -9.81]                                                     │
    │ boundary model: ……………………………………… BoundaryModelDummyParticles(SummationDensity, Nothing)           │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
    @test repr("text/plain", system) == show_box
end

@trixi_testset "Hydrodynamic Density" begin
    # Fluid interactions use density, mass, and smoothing length from the boundary model,
    # while structural mechanics retains the rigid material density.
    coordinates = [1.0 2.0
                   1.0 2.0]
    mass = [1.25, 1.5]
    material_densities = [990.0, 1000.0]
    hydrodynamic_densities = [1001.0, 1002.0]
    hydrodynamic_masses = [2.5, 3.0]

    initial_condition = InitialCondition(; coordinates, mass,
                                         density=material_densities)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 0.12
    boundary_model = BoundaryModelDummyParticles(hydrodynamic_densities,
                                                 hydrodynamic_masses,
                                                 SummationDensity(),
                                                 smoothing_kernel,
                                                 smoothing_length)

    system = RigidBodySystem(initial_condition; boundary_model)
    v = zeros(TrixiParticles.v_nvariables(system),
              TrixiParticles.n_integrated_particles(system))

    @test TrixiParticles.current_density(v, system) == hydrodynamic_densities
    @test TrixiParticles.hydrodynamic_mass(system, 1) == hydrodynamic_masses[1]
    @test TrixiParticles.smoothing_length(system, 1) == smoothing_length
    @test system.material_density == material_densities

    monaghan_model = BoundaryModelMonaghanKajtar(10.0, 1.0, smoothing_length,
                                                 hydrodynamic_masses)
    system_monaghan = RigidBodySystem(initial_condition; boundary_model=monaghan_model)
    @test TrixiParticles.hydrodynamic_mass(system_monaghan, 1) == hydrodynamic_masses[1]
end

@trixi_testset "Source Terms without Boundary Model" begin
    # Source terms must act on a standalone rigid system without requiring hydrodynamic state.
    coordinates = [1.0 2.0
                   1.0 2.0]
    mass = [1.25, 1.5]
    material_densities = [990.0, 1000.0]
    initial_condition = InitialCondition(; coordinates, mass,
                                         density=material_densities)

    source_terms = (coords, velocity, density, pressure,
                    t) -> SVector(density, pressure)
    system = RigidBodySystem(initial_condition; source_terms)
    semi = Semidiscretization(system, neighborhood_search=nothing)
    system = semi.systems[1]
    ode = semidiscretize(semi, (0.0, 0.0); reset_threads=false)

    v_ode = ode.u0.x[1]
    u_ode = ode.u0.x[2]
    dv_ode = similar(v_ode)
    fill!(dv_ode, 0.0)

    TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi, 0.0)

    dv = TrixiParticles.wrap_v(dv_ode, system, semi)
    @test dv[1, :] == material_densities
    @test dv[2, :] == zeros(2)
end

@trixi_testset "Initial Angular Velocity" begin
    # `apply_angular_velocity` encodes rigid rotation in particle velocities; initialization
    # writes those velocities before runtime kinematic caches are populated.
    coordinates_2d = [0.0 1.0
                      0.0 0.0]
    mass_2d = [1.0, 1.0]
    density_2d = [1000.0, 1000.0]
    ic_2d = apply_angular_velocity(InitialCondition(; coordinates=coordinates_2d,
                                                    mass=mass_2d,
                                                    density=density_2d),
                                   2.0)

    system_2d = RigidBodySystem(ic_2d; particle_spacing=0.1)
    u0_2d = zeros(2, 2)
    v0_2d = zeros(2, 2)
    TrixiParticles.write_u0!(u0_2d, system_2d)
    TrixiParticles.write_v0!(v0_2d, system_2d)

    @test iszero(system_2d.angular_velocity[])
    @test v0_2d == [0.0 0.0
                    -1.0 1.0]
    semi_2d = Semidiscretization(system_2d, neighborhood_search=nothing)
    system_2d = semi_2d.systems[1]
    ode_2d = semidiscretize(semi_2d, (0.0, 0.0); reset_threads=false)
    dt_2d = TrixiParticles.calculate_dt(v0_2d, u0_2d, 0.25, system_2d, semi_2d)
    @test isapprox(dt_2d, 0.25 * 0.1 / 1.0)
    dt_2d_larger_cfl = TrixiParticles.calculate_dt(v0_2d, u0_2d, 0.5,
                                                   system_2d, semi_2d)
    @test isapprox(dt_2d_larger_cfl, 0.5 * 0.1 / 1.0)
    dt_2d_semi = TrixiParticles.calculate_dt(ode_2d.u0.x[1], ode_2d.u0.x[2], 0.25,
                                             ode_2d.p.semi)
    @test isapprox(dt_2d_semi, dt_2d)

    TrixiParticles.update_final!(system_2d, v0_2d, u0_2d, nothing, nothing, nothing,
                                 0.0)
    @test system_2d.angular_velocity[] == 2.0

    # The same initialization and reconstruction path must preserve a vector-valued 3D spin.
    coordinates_3d = [0.0 1.0
                      0.0 0.0
                      0.0 0.0]
    mass_3d = [1.0, 1.0]
    density_3d = [1000.0, 1000.0]
    ic_3d = apply_angular_velocity(InitialCondition(; coordinates=coordinates_3d,
                                                    mass=mass_3d,
                                                    density=density_3d),
                                   (0.0, 0.0, 2.0))

    system_3d = RigidBodySystem(ic_3d)
    u0_3d = zeros(3, 2)
    v0_3d = zeros(3, 2)
    TrixiParticles.write_u0!(u0_3d, system_3d)
    TrixiParticles.write_v0!(v0_3d, system_3d)

    @test iszero(system_3d.angular_velocity[])
    @test v0_3d == [0.0 0.0
                    -1.0 1.0
                    0.0 0.0]
    TrixiParticles.update_final!(system_3d, v0_3d, u0_3d, nothing, nothing, nothing,
                                 0.0)
    @test system_3d.angular_velocity[] == [0.0, 0.0, 2.0]
end

@trixi_testset "Time Step Estimate 3D Gyroscopic" begin
    # An asymmetric 3D body is limited by both rotational velocity and the gyroscopic
    # acceleration generated by its nonuniform principal inertia.
    coordinates = [1.0 -1.0 0.0 0.0 0.0 0.0
                   0.0 0.0 2.0 -2.0 0.0 0.0
                   0.0 0.0 0.0 0.0 3.0 -3.0]
    mass = fill(1.0, 6)
    density = fill(1000.0, 6)
    initial_condition = apply_angular_velocity(InitialCondition(; coordinates, mass,
                                                                density,
                                                                particle_spacing=10.0),
                                               (1.0, 2.0, 3.0))
    system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0, 0.0))
    semi = Semidiscretization(system, neighborhood_search=nothing)
    system = semi.systems[1]
    ode = semidiscretize(semi, (0.0, 0.0); reset_threads=false)

    angular_velocity = SVector(1.0, 2.0, 3.0)
    gyroscopic_acceleration = SVector(-30 / 13, 12 / 5, -6 / 5)
    acceleration_scale = 3.0 * (norm(angular_velocity)^2 +
                                norm(gyroscopic_acceleration))
    dt_acceleration = 0.25 * sqrt(10.0 / acceleration_scale)
    dt_velocity = 0.25 * 10.0 / (3.0 * norm(angular_velocity))

    dt = TrixiParticles.calculate_dt(ode.u0.x[1], ode.u0.x[2], 0.25, system, semi)
    @test isapprox(dt, min(dt_acceleration, dt_velocity))
end

@trixi_testset "Time Step Estimate from Initial Velocity" begin
    # Timestep estimation must derive rotation directly from the ODE state before
    # `update_final!` has populated the cached angular velocity.
    coordinates = [-1.0 1.0
                   0.0 0.0]
    velocity = [0.0 0.0
                -1.0 1.0]
    mass = [1.0, 1.0]
    density = [1000.0, 1000.0]
    initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                         particle_spacing=0.1)
    system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0))
    semi = Semidiscretization(system, neighborhood_search=nothing)
    system = semi.systems[1]
    ode = semidiscretize(semi, (0.0, 0.0); reset_threads=false)

    @test iszero(system.angular_velocity[])

    dt = TrixiParticles.calculate_dt(ode.u0.x[1], ode.u0.x[2], 0.25, system, semi)
    @test isapprox(dt, 0.25 * 0.1 / 1.0)

    v = TrixiParticles.wrap_v(ode.u0.x[1], system, semi)
    u = TrixiParticles.wrap_u(ode.u0.x[2], system, semi)
    TrixiParticles.update_final!(system, v, u, ode.u0.x[1], ode.u0.x[2], semi, 0.0)
    @test system.angular_velocity[] == 1.0
end

@trixi_testset "Time Step Invariance under Uniform Acceleration" begin
    # Uniform translation/acceleration does not deform a rigid body and therefore must not
    # tighten its internal kinematic timestep estimate.
    coordinates = [-1.0 1.0
                   0.0 0.0]
    velocity = [1.0 1.0
                0.0 0.0]
    mass = [1.0, 1.0]
    density = [1000.0, 1000.0]
    initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                         particle_spacing=0.1)

    system_ref = RigidBodySystem(initial_condition; acceleration=(0.0, -9.81))
    semi_ref = Semidiscretization(system_ref, neighborhood_search=nothing)
    system_ref = semi_ref.systems[1]
    ode_ref = semidiscretize(semi_ref, (0.0, 0.0); reset_threads=false)

    system_shifted = RigidBodySystem(initial_condition; acceleration=(0.0, -1000.0))
    semi_shifted = Semidiscretization(system_shifted, neighborhood_search=nothing)
    system_shifted = semi_shifted.systems[1]
    ode_shifted = semidiscretize(semi_shifted, (0.0, 0.0); reset_threads=false)

    dt_ref = TrixiParticles.calculate_dt(ode_ref.u0.x[1], ode_ref.u0.x[2], 0.25,
                                         system_ref, semi_ref)
    dt_shifted = TrixiParticles.calculate_dt(ode_shifted.u0.x[1], ode_shifted.u0.x[2],
                                             0.25, system_shifted, semi_shifted)

    @test isapprox(dt_ref, 0.25 * 0.1 / 1.0)
    @test dt_shifted == dt_ref
end

@trixi_testset "Rotational Kinematics" begin
    # Opposite particle velocities represent unit angular velocity; force reduction then
    # converts the corresponding centripetal acceleration back to particle accelerations.
    coordinates = [-1.0 1.0
                   0.0 0.0]
    velocity = [0.0 0.0
                -1.0 1.0]
    mass = [1.0, 1.0]
    density = [1000.0, 1000.0]

    initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                         particle_spacing=1.0)
    rigid_system = RigidBodySystem(initial_condition;
                                   acceleration=(0.0, 0.0))

    v = copy(velocity)
    u = copy(coordinates)
    TrixiParticles.update_final!(rigid_system, v, u, nothing, nothing, nothing, 0.0)

    @test rigid_system.angular_velocity[] == 1.0
    @test rigid_system.inertia[] == 2.0

    dv = zeros(size(v))
    semi = DummySemidiscretization()
    TrixiParticles.interact!(dv, v, u, v, u, rigid_system, rigid_system, semi)
    @test all(iszero, dv)

    TrixiParticles.finalize_interaction!(rigid_system, dv, v, u,
                                         nothing, nothing, nothing, semi)

    @test dv == [1.0 -1.0
                 0.0 0.0]
end
