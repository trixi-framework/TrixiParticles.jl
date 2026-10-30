@trixi_testset "Rigid Contact History" begin
    using OrdinaryDiffEqLowStorageRK

    # Rigid-wall setup used to exercise callback scheduling and persistent manifold IDs.
    rigid_coordinates = reshape([0.0, 0.05], 2, 1)
    rigid_velocity = reshape([1.0, -1.0], 2, 1)
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

    history_model = RigidContactModel(; normal_stiffness=2.0e4,
                                      normal_damping=20.0,
                                      static_friction_coefficient=0.6,
                                      kinetic_friction_coefficient=0.4,
                                      tangential_stiffness=1.0e4,
                                      tangential_damping=5.0,
                                      contact_distance=0.1,
                                      stick_velocity_tolerance=1.0e-6)
    rigid_system = RigidBodySystem(rigid_ic;
                                   acceleration=(0.0, 0.0),
                                   contact_model=history_model)

    # Tangential displacement is path-dependent, so frictional systems allocate history and
    # require accepted-step updates.
    @test TrixiParticles.requires_update_callback(rigid_system)
    @test rigid_system.cache.contact_tangential_displacement isa Dict

    semi = Semidiscretization(rigid_system, boundary_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    dv_ode = zero(v_ode)
    # Evaluating the RHS without the required callback must fail before using stale history.
    update_error = try
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
        nothing
    catch err
        err
    end
    @test update_error isa ArgumentError
    @test occursin("`UpdateCallback` is required for `RigidBodySystem`",
                   sprint(showerror, update_error))

    # Sparse callbacks would integrate several accepted steps as one history increment.
    callback_error = try
        init(ode, RDPK3SpFSAL49(); adaptive=false, dt=1.0e-3,
             callback=UpdateCallback(interval=2))
        nothing
    catch err
        err
    end
    @test callback_error isa ArgumentError
    @test occursin("requires `UpdateCallback(interval=1)`",
                   sprint(showerror, callback_error))

    # Initialization registers a zero displacement; the first accepted step advances it once.
    integrator = init(ode, RDPK3SpFSAL49(); adaptive=false, dt=1.0e-3,
                      save_everystep=false, callback=UpdateCallback())
    initialized_map = rigid_system.cache.contact_tangential_displacement
    @test length(initialized_map) == 1
    @test all(iszero, values(initialized_map))

    step!(integrator)
    @test 0 < norm(first(values(initialized_map))) < 1.5e-3

    TrixiParticles.reset_contact_history!(rigid_system)

    # A direct accepted-step update creates a wall-contact key and integrates the slip over
    # exactly the supplied step size.
    TrixiParticles.update_rigid_contact_eachstep!(rigid_system, v_ode, u_ode, semi, 0.0,
                                                  1.0e-3)

    contact_map = rigid_system.cache.contact_tangential_displacement
    @test length(contact_map) == 1
    contact_key = first(keys(contact_map))
    tangential_displacement = contact_map[contact_key]
    @test contact_key.contact_kind == TrixiParticles.WallContact
    @test contact_key.local_particle == 1
    @test norm(tangential_displacement) > 0

    # A geometrically different manifold must receive a new ID rather than inheriting the
    # tangential displacement associated with the transient slot number.
    old_contact_id = contact_key.contact_slot
    descriptor = rigid_system.cache.wall_contact_descriptors[contact_key]
    TrixiParticles.set_zero!(rigid_system.cache.contact_manifold_count)
    TrixiParticles.set_zero!(rigid_system.cache.contact_manifold_weight_sum)
    TrixiParticles.set_zero!(rigid_system.cache.contact_manifold_normal_sum)
    TrixiParticles.set_zero!(rigid_system.cache.contact_manifold_wall_position_sum)
    TrixiParticles.set_zero!(rigid_system.cache.contact_manifold_history_id)
    rigid_system.cache.contact_manifold_count[1] = 1
    rigid_system.cache.contact_manifold_weight_sum[1, 1] = 1.0
    rigid_system.cache.contact_manifold_normal_sum[:, 1, 1] .= descriptor.normal
    rigid_system.cache.contact_manifold_wall_position_sum[:, 1,
                                                          1] .= descriptor.anchor .+
                                                                SVector(1.0, 0.0)
    boundary_index = TrixiParticles.system_indices(boundary_system, semi)
    TrixiParticles.match_wall_contact_manifolds!(rigid_system, boundary_index,
                                                 history_model;
                                                 update_descriptors=true)
    @test rigid_system.cache.contact_manifold_history_id[1, 1] != old_contact_id

    v_rigid = TrixiParticles.wrap_v(v_ode, rigid_system, semi)
    u_rigid = TrixiParticles.wrap_u(u_ode, rigid_system, semi)
    dv = TrixiParticles.wrap_v(dv_ode, rigid_system, semi)
    # Restart files do not serialize path-dependent contact state.
    TrixiParticles.restart_with!(rigid_system, v_rigid, u_rigid)
    @test isempty(rigid_system.cache.contact_tangential_displacement)
    @test isempty(rigid_system.cache.contact_sliding)
    @test isempty(rigid_system.cache.wall_contact_descriptors)
    @test rigid_system.cache.next_wall_contact_id[] == 1
    TrixiParticles.update_rigid_contact_eachstep!(rigid_system, v_ode, u_ode, semi, 0.0,
                                                  1.0e-3)

    # Recreated history opposes horizontal slip while the normal force opposes penetration;
    # both signs must survive reduction from particle forces to acceleration.
    TrixiParticles.update_final!(rigid_system, v_rigid, u_rigid, v_ode, u_ode, semi,
                                 0.0)
    TrixiParticles.reset_interaction_caches!(semi)
    TrixiParticles.interact!(dv_ode, v_ode, u_ode, rigid_system, boundary_system, semi)
    TrixiParticles.finalize_interaction!(rigid_system, dv, v_rigid, u_rigid,
                                         dv_ode, v_ode, u_ode, semi)

    @test rigid_system.force_per_particle[1, 1] < 0.0
    @test rigid_system.force_per_particle[2, 1] > 0.0
    @test dv[1, 1] < 0.0
    @test dv[2, 1] > 0.0

    # An RHS stage with a rotated wall normal must preserve normal loading and history.
    history_before_stage = copy(rigid_system.cache.contact_tangential_displacement)
    normal = normalize(SVector(1.0, 1.0))
    u_rigid[:, 1] .= 0.05 * normal
    v_rigid .= 0.0
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi)
    wall_normal_force = history_model.normal_stiffness * 0.05
    @test dot(rigid_system.resultant_force[], normal) ≈ wall_normal_force
    @test norm(rigid_system.resultant_force[] - wall_normal_force * normal) > 1.0
    @test rigid_system.cache.contact_tangential_displacement == history_before_stage

    # Once contact is lost, its path-dependent displacement must not affect future contacts.
    u_rigid[2, 1] = 0.2
    TrixiParticles.update_rigid_contact_eachstep!(rigid_system, v_ode, u_ode, semi, 0.0,
                                                  1.0e-3)
    @test isempty(rigid_system.cache.contact_tangential_displacement)
    @test isempty(rigid_system.cache.contact_sliding)

    # Both ordered rigid-rigid passes integrate opposite histories and must return an exact
    # action-reaction pair, including tangential force.
    rigid_coordinates_1 = reshape([0.0, 0.0], 2, 1)
    rigid_coordinates_2 = reshape([0.08, 0.0], 2, 1)
    rigid_velocity_1 = reshape([1.0, 0.5], 2, 1)
    rigid_velocity_2 = reshape([-0.5, -0.25], 2, 1)
    rigid_ic_1 = InitialCondition(; coordinates=rigid_coordinates_1,
                                  velocity=rigid_velocity_1,
                                  mass=[2.0],
                                  density=rigid_density,
                                  particle_spacing=0.1)
    rigid_ic_2 = InitialCondition(; coordinates=rigid_coordinates_2,
                                  velocity=rigid_velocity_2,
                                  mass=rigid_mass,
                                  density=rigid_density,
                                  particle_spacing=0.1)

    rigid_contact_model_1 = RigidContactModel(; normal_stiffness=20.0,
                                              normal_damping=4.0,
                                              static_friction_coefficient=0.6,
                                              kinetic_friction_coefficient=0.4,
                                              tangential_stiffness=10.0,
                                              tangential_damping=2.0,
                                              contact_distance=0.1)
    rigid_contact_model_2 = RigidContactModel(; normal_stiffness=30.0,
                                              normal_damping=8.0,
                                              static_friction_coefficient=0.5,
                                              kinetic_friction_coefficient=0.3,
                                              tangential_stiffness=8.0,
                                              tangential_damping=1.0,
                                              contact_distance=0.12)

    rigid_system_1 = RigidBodySystem(rigid_ic_1;
                                     acceleration=(0.0, 0.0),
                                     contact_model=rigid_contact_model_1)
    rigid_system_2 = RigidBodySystem(rigid_ic_2;
                                     acceleration=(0.0, 0.0),
                                     contact_model=rigid_contact_model_2)

    semi_rigid = Semidiscretization(rigid_system_1, rigid_system_2)
    ode_rigid = semidiscretize(semi_rigid, (0.0, 0.01))
    v_ode_rigid, u_ode_rigid = ode_rigid.u0.x
    dv_ode_rigid = zero(v_ode_rigid)
    TrixiParticles.update_rigid_contact_eachstep!(rigid_system_1, v_ode_rigid,
                                                  u_ode_rigid, semi_rigid, 0.0,
                                                  1.0e-3)
    TrixiParticles.update_rigid_contact_eachstep!(rigid_system_2, v_ode_rigid,
                                                  u_ode_rigid, semi_rigid, 0.0,
                                                  1.0e-3)

    rigid_key_1 = first(keys(rigid_system_1.cache.contact_tangential_displacement))
    rigid_key_2 = first(keys(rigid_system_2.cache.contact_tangential_displacement))
    @test rigid_key_1.contact_kind == TrixiParticles.RigidRigidContact
    @test rigid_key_2.contact_kind == TrixiParticles.RigidRigidContact

    v_rigid_1 = TrixiParticles.wrap_v(v_ode_rigid, rigid_system_1, semi_rigid)
    u_rigid_1 = TrixiParticles.wrap_u(u_ode_rigid, rigid_system_1, semi_rigid)
    v_rigid_2 = TrixiParticles.wrap_v(v_ode_rigid, rigid_system_2, semi_rigid)
    u_rigid_2 = TrixiParticles.wrap_u(u_ode_rigid, rigid_system_2, semi_rigid)
    TrixiParticles.update_final!(rigid_system_1, v_rigid_1, u_rigid_1,
                                 v_ode_rigid, u_ode_rigid, semi_rigid, 0.0)
    TrixiParticles.update_final!(rigid_system_2, v_rigid_2, u_rigid_2,
                                 v_ode_rigid, u_ode_rigid, semi_rigid, 0.0)
    TrixiParticles.interact!(dv_ode_rigid, v_ode_rigid, u_ode_rigid,
                             rigid_system_1, rigid_system_2, semi_rigid)
    TrixiParticles.interact!(dv_ode_rigid, v_ode_rigid, u_ode_rigid,
                             rigid_system_2, rigid_system_1, semi_rigid)

    # The uncapped pair force uses the symmetric pair parameters and opposite tangential
    # histories, so both ordered passes must agree on one analytical force.
    pair_contact_distance = max(rigid_contact_model_1.contact_distance,
                                rigid_contact_model_2.contact_distance)
    pair_normal_stiffness = (rigid_contact_model_1.normal_stiffness +
                             rigid_contact_model_2.normal_stiffness) / 2
    pair_normal_damping = (rigid_contact_model_1.normal_damping +
                           rigid_contact_model_2.normal_damping) / 2
    pair_penetration = pair_contact_distance - 0.08
    normal_velocity = -1.5
    expected_force_magnitude = pair_normal_stiffness * pair_penetration -
                               pair_normal_damping * normal_velocity
    pair_tangential_stiffness = (rigid_contact_model_1.tangential_stiffness +
                                 rigid_contact_model_2.tangential_stiffness) / 2
    pair_tangential_damping = (rigid_contact_model_1.tangential_damping +
                               rigid_contact_model_2.tangential_damping) / 2
    tangential_velocity = 0.75
    tangential_displacement = 1.0e-3 * tangential_velocity
    expected_tangential_force = -(pair_tangential_stiffness *
                                  tangential_displacement +
                                  pair_tangential_damping * tangential_velocity)

    @test rigid_system_1.force_per_particle[1, 1] ≈ -expected_force_magnitude
    @test rigid_system_1.force_per_particle[2, 1] ≈ expected_tangential_force
    @test rigid_system_2.force_per_particle[1, 1] ≈ expected_force_magnitude
    @test rigid_system_2.force_per_particle[2, 1] ≈ -expected_tangential_force
    @test rigid_system_1.force_per_particle[:, 1] ≈
          -rigid_system_2.force_per_particle[:, 1]
    @test rigid_system_1.cache.contact_count[] == 1
    @test rigid_system_2.cache.contact_count[] == 1
    @test rigid_system_1.cache.max_contact_penetration[] ≈ pair_penetration
    @test rigid_system_2.cache.max_contact_penetration[] ≈ pair_penetration

    # The rigid-pair RHS also projects a local copy when its contact normal changes.
    history_before_stage = copy(rigid_system_1.cache.contact_tangential_displacement)
    normal = -normalize(SVector(1.0, 1.0))
    u_rigid_2[:, 1] .= -0.08 * normal
    v_rigid_1 .= 0.0
    v_rigid_2 .= 0.0
    TrixiParticles.update_systems_and_nhs(v_ode_rigid, u_ode_rigid, semi_rigid, 0.0)
    TrixiParticles.system_interaction!(dv_ode_rigid, v_ode_rigid, u_ode_rigid, semi_rigid)
    pair_normal_force = pair_normal_stiffness * pair_penetration
    @test dot(rigid_system_1.resultant_force[], normal) ≈ pair_normal_force
    @test norm(rigid_system_1.resultant_force[] - pair_normal_force * normal) > 1.0e-3
    @test rigid_system_1.cache.contact_tangential_displacement == history_before_stage

    # Separating the bodies removes both ordered copies of their shared contact history.
    u_rigid_2[1, 1] = 0.5
    TrixiParticles.update_rigid_contact_eachstep!(rigid_system_1, v_ode_rigid,
                                                  u_ode_rigid, semi_rigid, 0.0,
                                                  1.0e-3)
    TrixiParticles.update_rigid_contact_eachstep!(rigid_system_2, v_ode_rigid,
                                                  u_ode_rigid, semi_rigid, 0.0,
                                                  1.0e-3)
    @test isempty(rigid_system_1.cache.contact_tangential_displacement)
    @test isempty(rigid_system_2.cache.contact_tangential_displacement)
    @test isempty(rigid_system_1.cache.contact_sliding)
    @test isempty(rigid_system_2.cache.contact_sliding)

    # Offset tangential forces must also produce the expected same-sense body torques.
    torque_coordinates_1 = [-0.05 0.05; 0.0 0.0]
    torque_coordinates_2 = [0.13 0.23; 0.0 0.0]
    torque_velocity_1 = [0.0 0.0; 1.0 1.0]
    torque_velocity_2 = zeros(2, 2)
    torque_ic_1 = InitialCondition(; coordinates=torque_coordinates_1,
                                   velocity=torque_velocity_1,
                                   mass=ones(2), density=fill(1000.0, 2),
                                   particle_spacing=0.1)
    torque_ic_2 = InitialCondition(; coordinates=torque_coordinates_2,
                                   velocity=torque_velocity_2,
                                   mass=ones(2), density=fill(1000.0, 2),
                                   particle_spacing=0.1)
    torque_system_1 = RigidBodySystem(torque_ic_1;
                                      acceleration=(0.0, 0.0),
                                      contact_model=rigid_contact_model_1)
    torque_system_2 = RigidBodySystem(torque_ic_2;
                                      acceleration=(0.0, 0.0),
                                      contact_model=rigid_contact_model_2)
    torque_semi = Semidiscretization(torque_system_1, torque_system_2)
    torque_ode = semidiscretize(torque_semi, (0.0, 0.01))
    torque_v_ode, torque_u_ode = torque_ode.u0.x
    torque_dv_ode = zero(torque_v_ode)
    TrixiParticles.update_rigid_contact_eachstep!(torque_system_1, torque_v_ode,
                                                  torque_u_ode, torque_semi, 0.0,
                                                  1.0e-3)
    TrixiParticles.update_rigid_contact_eachstep!(torque_system_2, torque_v_ode,
                                                  torque_u_ode, torque_semi, 0.0,
                                                  1.0e-3)
    TrixiParticles.update_systems_and_nhs(torque_v_ode, torque_u_ode, torque_semi,
                                          0.0)
    TrixiParticles.system_interaction!(torque_dv_ode, torque_v_ode, torque_u_ode,
                                       torque_semi)

    @test torque_system_1.resultant_force[] ≈ -torque_system_2.resultant_force[]
    @test torque_system_1.resultant_torque[] < 0
    @test torque_system_2.resultant_torque[] < 0

    # Include orbital angular momentum, not just the signs of the body torques.
    angular_momentum_rate = torque_system_1.resultant_torque[] +
                            torque_system_2.resultant_torque[] +
                            TrixiParticles.cross_product(torque_system_1.center_of_mass[],
                                                         torque_system_1.resultant_force[]) +
                            TrixiParticles.cross_product(torque_system_2.center_of_mass[],
                                                         torque_system_2.resultant_force[])
    @test isapprox(angular_momentum_rate, 0.0; atol=1.0e-12)

    @testset "Spring-only sliding" begin
        sliding_model = RigidContactModel(; normal_stiffness=100.0,
                                          static_friction_coefficient=0.6,
                                          kinetic_friction_coefficient=0.4,
                                          tangential_stiffness=100.0,
                                          contact_distance=0.1,
                                          stick_velocity_tolerance=0.0)
        sliding_system = RigidBodySystem(rigid_ic; contact_model=sliding_model,
                                         acceleration=(0.0, 0.0))
        sliding_semi = Semidiscretization(sliding_system, boundary_system)
        sliding_ode = semidiscretize(sliding_semi, (0.0, 1.0))
        v_sliding, u_sliding = sliding_ode.u0.x
        dv_sliding = zero(v_sliding)

        # A large slip increment breaks away; a small subsequent increment stays kinetic.
        for dt in (0.1, 1.0e-4)
            TrixiParticles.update_rigid_contact_eachstep!(sliding_system, v_sliding,
                                                          u_sliding, sliding_semi, 0.0, dt)
            TrixiParticles.system_interaction!(dv_sliding, v_sliding, u_sliding,
                                               sliding_semi)
            @test sliding_system.resultant_force[] ≈ SVector(-2.0, 5.0)
        end

        history_before_stop = copy(sliding_system.cache.contact_tangential_displacement)
        v = TrixiParticles.wrap_v(v_sliding, sliding_system, sliding_semi)
        v .= 0.0
        changed = TrixiParticles.update_rigid_contact_eachstep!(sliding_system, v_sliding,
                                                                u_sliding, sliding_semi,
                                                                0.0, 1.0e-4)
        # A mode-only change still needs FSAL invalidation; reversal unloads the spring.
        @test changed &&
              sliding_system.cache.contact_tangential_displacement == history_before_stop
        v[1, 1] = -1.0
        TrixiParticles.update_rigid_contact_eachstep!(sliding_system, v_sliding, u_sliding,
                                                      sliding_semi, 0.0, 1.0e-4)
        TrixiParticles.system_interaction!(dv_sliding, v_sliding, u_sliding, sliding_semi)
        @test sliding_system.resultant_force[] ≈ SVector(-1.99, 5.0)
    end
end
