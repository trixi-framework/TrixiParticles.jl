@trixi_testset "Rigid-Rigid Normal Contact" begin
    # Each ordered interaction updates only its local body. Running both orders must produce
    # the complete action-reaction pair from one symmetric set of contact parameters.
    rigid_coordinates_1 = reshape([0.0, 0.0], 2, 1)
    rigid_coordinates_2 = reshape([0.08, 0.0], 2, 1)
    rigid_velocity_1 = reshape([1.0, 0.0], 2, 1)
    rigid_velocity_2 = reshape([-0.5, 0.0], 2, 1)
    rigid_mass_1 = [2.0]
    rigid_mass_2 = [1.0]
    rigid_density_pair = [1000.0]

    rigid_ic_1 = InitialCondition(; coordinates=rigid_coordinates_1,
                                  velocity=rigid_velocity_1,
                                  mass=rigid_mass_1,
                                  density=rigid_density_pair,
                                  particle_spacing=0.1)
    rigid_ic_2 = InitialCondition(; coordinates=rigid_coordinates_2,
                                  velocity=rigid_velocity_2,
                                  mass=rigid_mass_2,
                                  density=rigid_density_pair,
                                  particle_spacing=0.1)

    contact_model_1 = RigidContactModel(; normal_stiffness=20.0,
                                        normal_damping=4.0,
                                        contact_distance=0.1)
    contact_model_2 = RigidContactModel(; normal_stiffness=30.0,
                                        normal_damping=8.0,
                                        contact_distance=0.12)

    rigid_system_1 = RigidBodySystem(rigid_ic_1;
                                     acceleration=(0.0, 0.0),
                                     contact_model=contact_model_1)
    rigid_system_2 = RigidBodySystem(rigid_ic_2;
                                     acceleration=(0.0, 0.0),
                                     contact_model=contact_model_2)
    rigid_system_without_contact = RigidBodySystem(rigid_ic_1;
                                                   acceleration=(0.0, 0.0))

    semi_rigid = Semidiscretization(rigid_system_1, rigid_system_2)
    ode_rigid = semidiscretize(semi_rigid, (0.0, 0.01))
    v_ode_rigid, u_ode_rigid = ode_rigid.u0.x
    dv_ode_rigid = zero(v_ode_rigid)

    v_rigid_1 = TrixiParticles.wrap_v(v_ode_rigid, rigid_system_1, semi_rigid)
    u_rigid_1 = TrixiParticles.wrap_u(u_ode_rigid, rigid_system_1, semi_rigid)
    v_rigid_2 = TrixiParticles.wrap_v(v_ode_rigid, rigid_system_2, semi_rigid)
    u_rigid_2 = TrixiParticles.wrap_u(u_ode_rigid, rigid_system_2, semi_rigid)
    TrixiParticles.update_final!(rigid_system_1, v_rigid_1, u_rigid_1,
                                 v_ode_rigid, u_ode_rigid, semi_rigid, 0.0)
    TrixiParticles.update_final!(rigid_system_2, v_rigid_2, u_rigid_2,
                                 v_ode_rigid, u_ode_rigid, semi_rigid, 0.0)

    # The forward ordered pass updates only its first (local) system; the reverse pass then
    # supplies the reaction on the other body without changing the first force again.
    TrixiParticles.reset_interaction_caches!(semi_rigid)
    TrixiParticles.interact!(dv_ode_rigid, v_ode_rigid, u_ode_rigid,
                             rigid_system_1, rigid_system_2, semi_rigid)
    force_after_forward_1 = copy(rigid_system_1.force_per_particle)
    force_after_forward_2 = copy(rigid_system_2.force_per_particle)
    @test !all(iszero, force_after_forward_1)
    @test all(iszero, force_after_forward_2)

    TrixiParticles.interact!(dv_ode_rigid, v_ode_rigid, u_ode_rigid,
                             rigid_system_2, rigid_system_1, semi_rigid)
    @test rigid_system_1.force_per_particle == force_after_forward_1
    @test !all(iszero, rigid_system_2.force_per_particle)

    # Both orders use maximum support, averaged normal coefficients, and reduced pair mass.
    # These choices define one force magnitude and one contact stability timescale.
    pair_contact_distance = max(contact_model_1.contact_distance,
                                contact_model_2.contact_distance)
    pair_normal_stiffness = (contact_model_1.normal_stiffness +
                             contact_model_2.normal_stiffness) / 2
    pair_normal_damping = (contact_model_1.normal_damping +
                           contact_model_2.normal_damping) / 2
    pair_penetration = pair_contact_distance - 0.08
    normal_velocity = -1.5
    reduced_mass = rigid_mass_1[1] * rigid_mass_2[1] /
                   (rigid_mass_1[1] + rigid_mass_2[1])
    pair_contact_dt = min(sqrt(reduced_mass / pair_normal_stiffness),
                          reduced_mass / pair_normal_damping)
    expected_force_magnitude = pair_normal_stiffness * pair_penetration -
                               pair_normal_damping * normal_velocity
    expected_force = SVector(-expected_force_magnitude, 0.0)

    @test vec(force_after_forward_1[:, 1]) ≈ collect(expected_force)
    @test vec(rigid_system_2.force_per_particle[:, 1]) ≈ collect(-expected_force)
    @test rigid_system_1.cache.contact_count[] == 1
    @test rigid_system_2.cache.contact_count[] == 1
    @test rigid_system_1.cache.max_contact_penetration[] ≈ pair_penetration
    @test rigid_system_2.cache.max_contact_penetration[] ≈ pair_penetration

    # A missing model disables pair search and pair timestep restrictions in either order.
    @test TrixiParticles.compact_support(rigid_system_1, rigid_system_2) ≈
          pair_contact_distance
    @test iszero(TrixiParticles.compact_support(rigid_system_without_contact,
                                                rigid_system_2))
    @test iszero(TrixiParticles.compact_support(rigid_system_2,
                                                rigid_system_without_contact))
    @test TrixiParticles.contact_time_step(rigid_system_1, rigid_system_2) ≈
          pair_contact_dt
    @test TrixiParticles.contact_time_step(rigid_system_without_contact,
                                           rigid_system_2) == Inf
    @test TrixiParticles.contact_time_step(rigid_system_2,
                                           rigid_system_without_contact) == Inf
    @test TrixiParticles.contact_time_step(rigid_system_1) ≈
          min(sqrt(rigid_mass_1[1] / contact_model_1.normal_stiffness),
              rigid_mass_1[1] / contact_model_1.normal_damping)
    @test TrixiParticles.contact_time_step(rigid_system_2) ≈
          min(sqrt(rigid_mass_2[1] / contact_model_2.normal_stiffness),
              rigid_mass_2[1] / contact_model_2.normal_damping)
    # A lone rigid body has no pair restriction; once paired, the semidiscretization applies
    # the CFL factor exactly once to the reduced-mass contact timescale.
    semi_single_rigid = Semidiscretization(rigid_system_1)
    ode_single_rigid = semidiscretize(semi_single_rigid, (0.0, 0.01))
    zero_velocity_single = zero(ode_single_rigid.u0.x[1])
    @test TrixiParticles.calculate_dt(zero_velocity_single, ode_single_rigid.u0.x[2],
                                      0.25, rigid_system_1, semi_single_rigid) == Inf
    zero_velocity_ode = zero(v_ode_rigid)
    @test TrixiParticles.calculate_dt(zero_velocity_ode, u_ode_rigid, 0.25,
                                      rigid_system_1, semi_rigid) ≈
          0.25 * pair_contact_dt
    @test TrixiParticles.calculate_dt(zero_velocity_ode, u_ode_rigid, 0.25,
                                      semi_rigid) ≈ 0.25 * pair_contact_dt

    # The high-level interaction path must rebuild the same forces and diagnostics after its
    # internal cache reset.
    dv_ode_reset = zero(v_ode_rigid)
    TrixiParticles.system_interaction!(dv_ode_reset, v_ode_rigid, u_ode_rigid,
                                       semi_rigid)
    @test rigid_system_1.cache.contact_count[] == 1
    @test rigid_system_2.cache.contact_count[] == 1
    @test rigid_system_1.cache.max_contact_penetration[] ≈ pair_penetration
    @test rigid_system_2.cache.max_contact_penetration[] ≈ pair_penetration
    @test vec(rigid_system_1.force_per_particle[:, 1]) ≈ collect(expected_force)
    @test vec(rigid_system_2.force_per_particle[:, 1]) ≈ collect(-expected_force)

    # Refreshing system state and neighborhood searches must not erase completed diagnostics.
    TrixiParticles.update_systems_and_nhs(v_ode_rigid, u_ode_rigid, semi_rigid, 0.0)
    @test rigid_system_1.cache.contact_count[] == 1
    @test rigid_system_2.cache.contact_count[] == 1
    @test rigid_system_1.cache.max_contact_penetration[] ≈ pair_penetration
    @test rigid_system_2.cache.max_contact_penetration[] ≈ pair_penetration
    @test vec(rigid_system_1.force_per_particle[:, 1]) ≈ collect(expected_force)
    @test vec(rigid_system_2.force_per_particle[:, 1]) ≈ collect(-expected_force)

    # A second RHS interaction starts cleanly and reproduces the first evaluation.
    TrixiParticles.set_zero!(dv_ode_reset)
    TrixiParticles.system_interaction!(dv_ode_reset, v_ode_rigid, u_ode_rigid,
                                       semi_rigid)
    @test rigid_system_1.cache.contact_count[] == 1
    @test rigid_system_2.cache.contact_count[] == 1
    @test rigid_system_1.cache.max_contact_penetration[] ≈ pair_penetration
    @test rigid_system_2.cache.max_contact_penetration[] ≈ pair_penetration
    @test vec(rigid_system_1.force_per_particle[:, 1]) ≈ collect(expected_force)
    @test vec(rigid_system_2.force_per_particle[:, 1]) ≈ collect(-expected_force)

    # Final reduction preserves action-reaction forces and converts each force to acceleration
    # with its body's own mass.
    dv_rigid_1 = TrixiParticles.wrap_v(dv_ode_rigid, rigid_system_1, semi_rigid)
    dv_rigid_2 = TrixiParticles.wrap_v(dv_ode_rigid, rigid_system_2, semi_rigid)
    TrixiParticles.finalize_interaction!(rigid_system_1, dv_rigid_1, v_rigid_1,
                                         u_rigid_1, dv_ode_rigid, v_ode_rigid,
                                         u_ode_rigid, semi_rigid)
    TrixiParticles.finalize_interaction!(rigid_system_2, dv_rigid_2, v_rigid_2,
                                         u_rigid_2, dv_ode_rigid, v_ode_rigid,
                                         u_ode_rigid, semi_rigid)

    @test rigid_system_1.resultant_force[] ≈ expected_force
    @test rigid_system_2.resultant_force[] ≈ -expected_force
    @test dv_rigid_1[1, 1] ≈ expected_force[1] / rigid_mass_1[1]
    @test dv_rigid_2[1, 1] ≈ -expected_force[1] / rigid_mass_2[1]
    @test dv_rigid_1[2, 1] ≈ 0.0
    @test dv_rigid_2[2, 1] ≈ 0.0

    # Contact diagnostics written to VTK must match the active runtime cache, not zeros or
    # stale values from an earlier RHS evaluation.
    mktempdir() do tmp_dir
        du_ode_rigid = zero(u_ode_rigid)
        dvdu_ode_rigid = (; x=(dv_ode_rigid, du_ode_rigid))
        vu_ode_rigid = (; x=(v_ode_rigid, u_ode_rigid))
        trixi2vtk(dvdu_ode_rigid, vu_ode_rigid, semi_rigid, 0.0;
                  output_directory=tmp_dir, iter=1)

        contact_filename = TrixiParticles.system_names(semi_rigid.systems)[1]
        vtk_contact = TrixiParticles.ReadVTK.VTKFile(joinpath(tmp_dir,
                                                              "$(contact_filename)_1.vtu"))
        point_data_contact = TrixiParticles.ReadVTK.get_point_data(vtk_contact)

        @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["contact_count"]))) ==
              rigid_system_1.cache.contact_count[]
        @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["contact_count"]))) >
              0
        @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["max_contact_penetration"]))) ≈
              rigid_system_1.cache.max_contact_penetration[]
        @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["max_contact_penetration"]))) >
              0
    end
end
