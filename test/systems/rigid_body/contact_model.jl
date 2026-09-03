@trixi_testset "Contact Model and Rigid-Wall Contact" begin
    # A single particle approaching a wall gives analytically simple penetration and force
    # values for checking contact-model construction and runtime interaction paths.
    rigid_coordinates = reshape([0.0, 0.05], 2, 1)
    rigid_velocity = reshape([0.0, -1.0], 2, 1)
    rigid_mass = [1.0]
    rigid_density = [1000.0]
    rigid_ic = InitialCondition(; coordinates=rigid_coordinates,
                                velocity=rigid_velocity,
                                mass=rigid_mass,
                                density=rigid_density,
                                particle_spacing=0.1)

    boundary_coordinates = reshape([0.0, 0.0], 2, 1)
    boundary_mass = [1.0]
    boundary_density = [1000.0]
    boundary_ic = InitialCondition(; coordinates=boundary_coordinates,
                                   mass=boundary_mass,
                                   density=boundary_density,
                                   particle_spacing=0.1)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 0.15
    boundary_model = BoundaryModelDummyParticles(boundary_density, boundary_mass,
                                                 SummationDensity(),
                                                 smoothing_kernel,
                                                 smoothing_length)
    boundary_system = WallBoundarySystem(boundary_ic, boundary_model)

    contact_model = RigidContactModel(; normal_stiffness=2.0e4,
                                      normal_damping=20.0,
                                      contact_distance=0.1)

    # Copying a normal-only model fills all inactive friction parameters with zero.
    runtime_model = TrixiParticles.copy_contact_model(contact_model, 0.1, Float64)
    @test runtime_model.normal_stiffness ≈ 2.0e4
    @test runtime_model.normal_damping ≈ 20.0
    @test runtime_model.static_friction_coefficient ≈ 0.0
    @test runtime_model.kinetic_friction_coefficient ≈ 0.0
    @test runtime_model.tangential_stiffness ≈ 0.0
    @test runtime_model.tangential_damping ≈ 0.0
    @test runtime_model.contact_distance ≈ 0.1
    @test runtime_model.stick_velocity_tolerance ≈ 1.0e-6
    @test runtime_model.penetration_slop ≈ 0.0

    # Runtime copies adopt the system's scalar type and replace a zero contact distance with
    # the system particle spacing.
    advanced_contact_model = RigidContactModel(; normal_stiffness=5.0,
                                               normal_damping=1.5,
                                               static_friction_coefficient=0.6,
                                               kinetic_friction_coefficient=0.4,
                                               tangential_stiffness=9.0,
                                               tangential_damping=2.5,
                                               contact_distance=0.0,
                                               stick_velocity_tolerance=1.0e-5,
                                               penetration_slop=0.01)
    advanced_runtime_model = TrixiParticles.copy_contact_model(advanced_contact_model,
                                                               0.125, Float32)
    @test advanced_runtime_model.normal_stiffness ≈ Float32(5.0)
    @test advanced_runtime_model.normal_damping ≈ Float32(1.5)
    @test advanced_runtime_model.static_friction_coefficient ≈ Float32(0.6)
    @test advanced_runtime_model.kinetic_friction_coefficient ≈ Float32(0.4)
    @test advanced_runtime_model.tangential_stiffness ≈ Float32(9.0)
    @test advanced_runtime_model.tangential_damping ≈ Float32(2.5)
    @test advanced_runtime_model.contact_distance ≈ Float32(0.125)
    @test advanced_runtime_model.stick_velocity_tolerance ≈ Float32(1.0e-5)
    @test advanced_runtime_model.penetration_slop ≈ Float32(0.01)

    # The same spacing fallback also applies when no contact distance is supplied.
    spacing_scaled_model = RigidContactModel(; normal_stiffness=5.0)
    spacing_scaled_runtime = TrixiParticles.copy_contact_model(spacing_scaled_model,
                                                               0.125,
                                                               Float64)
    @test spacing_scaled_runtime.contact_distance ≈ 0.125

    # Reject invalid normal/friction values and incomplete friction configurations rather
    # than silently constructing a model with no physical tangential response.
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=0.0)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 normal_damping=-1.0)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 contact_distance=-1.0)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 static_friction_coefficient=-0.1)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 static_friction_coefficient=0.3,
                                                 kinetic_friction_coefficient=0.4)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 tangential_stiffness=-1.0)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 tangential_damping=-1.0)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 stick_velocity_tolerance=-1.0)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 penetration_slop=-1.0)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 static_friction_coefficient=0.6,
                                                 kinetic_friction_coefficient=0.4)
    @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                 static_friction_coefficient=0.0,
                                                 kinetic_friction_coefficient=0.0,
                                                 tangential_stiffness=1.0)

    # Exercise both the zero-slip restoring direction and regularized kinetic branch.
    force_model = RigidContactModel(; normal_stiffness=100.0,
                                    static_friction_coefficient=0.6,
                                    kinetic_friction_coefficient=0.4,
                                    tangential_stiffness=100.0,
                                    stick_velocity_tolerance=1.0e-6)
    zero_slip_force = TrixiParticles.tangential_contact_force(force_model,
                                                              SVector(1.0, 0.0),
                                                              SVector(0.0, 0.0), 10.0)
    @test zero_slip_force ≈ SVector(-4.0, 0.0)

    sliding_velocity = SVector(1.0e-5, 0.0)
    sliding_force = TrixiParticles.tangential_contact_force(force_model,
                                                            SVector(1.0, 0.0),
                                                            sliding_velocity, 1.0)
    @test dot(sliding_force, sliding_velocity) <= 0

    force_model_f32 = RigidContactModel(; normal_stiffness=100.0f0,
                                        static_friction_coefficient=0.6f0,
                                        kinetic_friction_coefficient=0.4f0,
                                        tangential_stiffness=100.0f0,
                                        stick_velocity_tolerance=1.0f-6)
    sliding_force_f32 = TrixiParticles.tangential_contact_force(force_model_f32,
                                                                SVector(1.0f0, 0.0f0),
                                                                SVector(1.0f-5, 0.0f0),
                                                                1.0f0)
    @test norm(sliding_force_f32) ≈ norm(sliding_force) rtol = 5.0f-6

    # Pair reduction must be independent of the order in which the two bodies are visited.
    pair_parameters_12 = TrixiParticles.rigid_contact_pair_parameters(force_model,
                                                                      advanced_contact_model)
    pair_parameters_21 = TrixiParticles.rigid_contact_pair_parameters(advanced_contact_model,
                                                                      force_model)
    @test pair_parameters_12 == pair_parameters_21

    # Only frictional contact needs persistent history and therefore an update callback.
    rigid_system = RigidBodySystem(rigid_ic;
                                   acceleration=(0.0, 0.0),
                                   contact_model=contact_model)
    rigid_system_advanced = RigidBodySystem(rigid_ic;
                                            acceleration=(0.0, 0.0),
                                            contact_model=advanced_contact_model)
    rigid_system_with_boundary = RigidBodySystem(rigid_ic;
                                                 acceleration=(0.0, 0.0),
                                                 boundary_model=boundary_model,
                                                 contact_model=contact_model)
    rigid_system_custom_manifolds = RigidBodySystem(rigid_ic;
                                                    acceleration=(0.0, 0.0),
                                                    contact_model=contact_model,
                                                    max_manifolds=3)
    rigid_system_without_contact = RigidBodySystem(rigid_ic;
                                                   acceleration=(0.0, 0.0),
                                                   boundary_model=boundary_model)
    @test haskey(rigid_system.cache, :contact_manifold_count)
    @test rigid_system.contact_model.normal_stiffness ≈ contact_model.normal_stiffness
    @test rigid_system.contact_model.normal_damping ≈ contact_model.normal_damping
    @test rigid_system.contact_model.contact_distance ≈ contact_model.contact_distance
    @test !TrixiParticles.requires_update_callback(rigid_system)
    @test isnothing(rigid_system.cache.contact_tangential_displacement)
    @test TrixiParticles.requires_update_callback(rigid_system_advanced)
    @test rigid_system_advanced.cache.contact_tangential_displacement isa Dict
    # Contact configuration is serialized with the system and determines rigid-wall search
    # support; the reverse wall-rigid direction does not initiate contact.
    rigid_system_data = Dict{String, Any}()
    TrixiParticles.add_system_data!(rigid_system_data, rigid_system)
    @test rigid_system_data["contact_model"]["model"] ==
          TrixiParticles.type2string(rigid_system.contact_model)
    @test rigid_system_data["contact_model"]["normal_stiffness"] ≈
          contact_model.normal_stiffness
    @test rigid_system_data["contact_model"]["normal_damping"] ≈
          contact_model.normal_damping
    @test rigid_system_data["contact_model"]["contact_distance"] ≈
          contact_model.contact_distance
    @test size(rigid_system_custom_manifolds.cache.contact_manifold_weight_sum, 1) == 3
    @test TrixiParticles.compact_support(rigid_system, boundary_system) ≈
          contact_model.contact_distance
    @test TrixiParticles.compact_support(rigid_system_with_boundary,
                                         boundary_system) ≈
          contact_model.contact_distance
    @test iszero(TrixiParticles.compact_support(boundary_system, rigid_system))
    @test iszero(TrixiParticles.compact_support(rigid_system_without_contact,
                                                boundary_system))
    @test_throws ArgumentError RigidBodySystem(rigid_ic; contact_model, max_manifolds=0)

    # Runtime metadata must expose every active friction parameter for reproducibility.
    system_meta_data = Dict{String, Any}()
    TrixiParticles.add_system_data!(system_meta_data, rigid_system)
    @test system_meta_data["contact_model"]["normal_stiffness"] ≈ 2.0e4
    @test system_meta_data["contact_model"]["normal_damping"] ≈ 20.0
    @test system_meta_data["contact_model"]["contact_distance"] ≈ 0.1

    system_meta_data = Dict{String, Any}()
    TrixiParticles.add_system_data!(system_meta_data, rigid_system_advanced)
    @test system_meta_data["contact_model"]["normal_stiffness"] ≈ 5.0
    @test system_meta_data["contact_model"]["normal_damping"] ≈ 1.5
    @test system_meta_data["contact_model"]["static_friction_coefficient"] ≈ 0.6
    @test system_meta_data["contact_model"]["kinetic_friction_coefficient"] ≈ 0.4
    @test system_meta_data["contact_model"]["tangential_stiffness"] ≈ 9.0
    @test system_meta_data["contact_model"]["tangential_damping"] ≈ 2.5
    @test system_meta_data["contact_model"]["contact_distance"] ≈ 0.1
    @test system_meta_data["contact_model"]["stick_velocity_tolerance"] ≈ 1.0e-5
    @test system_meta_data["contact_model"]["penetration_slop"] ≈ 0.01

    semi = Semidiscretization(rigid_system, boundary_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    dv_ode = zero(v_ode)
    wall_contact_dt = sqrt(rigid_mass[1] / contact_model.normal_stiffness)

    # The stable contact step is the minimum active spring and damping timescale.
    @test TrixiParticles.contact_time_step(rigid_system) ≈ wall_contact_dt
    @test TrixiParticles.contact_time_step(rigid_system, boundary_system) ≈
          wall_contact_dt
    advanced_contact_dt = min(sqrt(rigid_mass[1] /
                                   advanced_runtime_model.normal_stiffness),
                              rigid_mass[1] /
                              advanced_runtime_model.normal_damping,
                              sqrt(rigid_mass[1] /
                                   advanced_runtime_model.tangential_stiffness),
                              rigid_mass[1] /
                              advanced_runtime_model.tangential_damping)
    @test TrixiParticles.contact_time_step(rigid_system_advanced) ≈
          advanced_contact_dt

    # Check direct and full-RHS wall contact, including contact support wider than the
    # boundary model's hydrodynamic support.
    kick_boundary_model = BoundaryModelDummyParticles(boundary_density, boundary_mass,
                                                      SummationDensity(),
                                                      smoothing_kernel,
                                                      smoothing_length)
    kick_rigid_system = RigidBodySystem(rigid_ic; acceleration=(0.0, 0.0),
                                        contact_model)
    kick_boundary_system = WallBoundarySystem(boundary_ic, kick_boundary_model)
    kick_semi = Semidiscretization(kick_rigid_system, kick_boundary_system)
    kick_ode = semidiscretize(kick_semi, (0.0, 0.01))
    kick_v_ode, kick_u_ode = kick_ode.u0.x
    kick_dv_ode = zero(kick_v_ode)

    TrixiParticles.kick!(kick_dv_ode, kick_v_ode, kick_u_ode, kick_ode.p, 0.0)
    kick_dv = TrixiParticles.wrap_v(kick_dv_ode, kick_rigid_system, kick_semi)

    @test kick_dv[2, 1] > 0
    @test kick_rigid_system.resultant_force[][2] > 0

    TrixiParticles.reset_interaction_caches!(semi)
    TrixiParticles.interact!(dv_ode, v_ode, u_ode, rigid_system, boundary_system, semi)
    dv = TrixiParticles.wrap_v(dv_ode, rigid_system, semi)
    v_rigid = TrixiParticles.wrap_v(v_ode, rigid_system, semi)
    u_rigid = TrixiParticles.wrap_u(u_ode, rigid_system, semi)
    TrixiParticles.finalize_interaction!(rigid_system, dv, v_rigid, u_rigid,
                                         dv_ode, v_ode, u_ode, semi)

    @test dv[2, 1] > 0
    @test rigid_system.cache.contact_count[] == 1
    @test rigid_system.cache.max_contact_penetration[] ≈ 0.05
    direct_force = copy(rigid_system.force_per_particle)
    direct_resultant_force = rigid_system.resultant_force[]

    # Repeating the same direct interaction after a cache reset must reproduce, rather than
    # accumulate, force and diagnostic values.
    TrixiParticles.set_zero!(dv_ode)
    TrixiParticles.update_final!(rigid_system, v_rigid, u_rigid, v_ode, u_ode, semi,
                                 0.0)
    TrixiParticles.reset_interaction_caches!(semi)
    TrixiParticles.interact!(dv_ode, v_ode, u_ode, rigid_system, boundary_system, semi)
    TrixiParticles.finalize_interaction!(rigid_system, dv, v_rigid, u_rigid,
                                         dv_ode, v_ode, u_ode, semi)

    @test rigid_system.cache.contact_count[] == 1
    @test rigid_system.cache.max_contact_penetration[] ≈ 0.05
    @test rigid_system.force_per_particle == direct_force
    @test rigid_system.resultant_force[] ≈ direct_resultant_force

    # Finalizing without a preceding interaction must not retain stale contact resultants.
    TrixiParticles.set_zero!(dv_ode)
    TrixiParticles.update_final!(rigid_system, v_rigid, u_rigid, v_ode, u_ode, semi,
                                 0.0)
    TrixiParticles.reset_interaction_caches!(semi)
    TrixiParticles.finalize_interaction!(rigid_system, dv, v_rigid, u_rigid,
                                         dv_ode, v_ode, u_ode, semi)

    @test all(iszero, dv)
    @test iszero(rigid_system.resultant_force[])
    @test iszero(rigid_system.resultant_torque[])
    @test iszero(rigid_system.angular_acceleration_force[])

    # Contact distance, not the shorter hydrodynamic kernel support, controls whether the
    # rigid particle and wall become neighbors.
    far_rigid_ic = InitialCondition(; coordinates=reshape([0.0, 0.09], 2, 1),
                                    velocity=rigid_velocity,
                                    mass=rigid_mass,
                                    density=rigid_density,
                                    particle_spacing=0.1)
    short_support_boundary_model = BoundaryModelDummyParticles(boundary_density,
                                                               boundary_mass,
                                                               SummationDensity(),
                                                               smoothing_kernel,
                                                               0.04)
    short_support_boundary = WallBoundarySystem(boundary_ic,
                                                short_support_boundary_model)
    far_rigid_system = RigidBodySystem(far_rigid_ic; acceleration=(0.0, 0.0),
                                       contact_model)
    short_support_semi = Semidiscretization(far_rigid_system, short_support_boundary)
    short_support_ode = semidiscretize(short_support_semi, (0.0, 0.01))
    short_support_v_ode, short_support_u_ode = short_support_ode.u0.x
    short_support_dv_ode = zero(short_support_v_ode)

    TrixiParticles.reset_interaction_caches!(short_support_semi)
    TrixiParticles.interact!(short_support_dv_ode, short_support_v_ode,
                             short_support_u_ode, far_rigid_system,
                             short_support_boundary, short_support_semi)
    short_support_dv = TrixiParticles.wrap_v(short_support_dv_ode, far_rigid_system,
                                             short_support_semi)
    short_support_v = TrixiParticles.wrap_v(short_support_v_ode, far_rigid_system,
                                            short_support_semi)
    short_support_u = TrixiParticles.wrap_u(short_support_u_ode, far_rigid_system,
                                            short_support_semi)
    TrixiParticles.finalize_interaction!(far_rigid_system, short_support_dv,
                                         short_support_v, short_support_u,
                                         short_support_dv_ode, short_support_v_ode,
                                         short_support_u_ode, short_support_semi)

    @test short_support_dv[2, 1] > 0
end
