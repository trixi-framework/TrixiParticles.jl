@trixi_testset "Rigid Dynamic Invariants" begin
    using OrdinaryDiffEqLowStorageRK

    function rigid_state(state, system, semi)
        v_ode, u_ode = state.x
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        coordinates = TrixiParticles.current_coordinates(u, system)
        velocity = TrixiParticles.current_velocity(v, system)
        center_of_mass,
        center_of_mass_velocity = TrixiParticles.rigid_center_of_mass_kinematics(system,
                                                                                 coordinates,
                                                                                 velocity)
        rotation = TrixiParticles.rigid_rotational_kinematics(system, coordinates,
                                                              velocity, center_of_mass,
                                                              center_of_mass_velocity)

        return (; coordinates=copy(coordinates), velocity=copy(velocity), center_of_mass,
                center_of_mass_velocity, rotation)
    end

    function pairwise_distances(coordinates)
        return [norm(coordinates[:, i] - coordinates[:, j])
                for i in axes(coordinates, 2) for j in (i + 1):size(coordinates, 2)]
    end

    function kinetic_energy(state, system)
        translation = 0.5 * system.total_mass * sum(abs2, state.center_of_mass_velocity)
        rotation = 0.5 * dot(state.rotation.angular_velocity,
                       state.rotation.inertia * state.rotation.angular_velocity)
        return translation + rotation
    end

    # Free 2D rotation should preserve the rigid shape, center of mass, energy, and angular
    # momentum over a complete revolution.
    coordinates_2d = [-0.5 0.5 0.5 -0.5
                      -0.5 -0.5 0.5 0.5]
    ic_2d = InitialCondition(; coordinates=coordinates_2d, mass=ones(4),
                             density=ones(4), particle_spacing=1.0)
    ic_2d = apply_angular_velocity(ic_2d, 2.0)
    system_2d = RigidBodySystem(ic_2d; acceleration=(0.0, 0.0))
    semi_2d = Semidiscretization(system_2d, neighborhood_search=nothing)
    ode_2d = semidiscretize(semi_2d, (0.0, pi))
    sol_2d = solve(ode_2d, RDPK3SpFSAL49(); abstol=1.0e-10, reltol=1.0e-10,
                   dtmax=0.01, save_everystep=false)
    initial_2d = rigid_state(sol_2d.u[begin], system_2d, semi_2d)
    final_2d = rigid_state(sol_2d.u[end], system_2d, semi_2d)

    @test sol_2d.retcode == ReturnCode.Success
    @test pairwise_distances(final_2d.coordinates) ≈
          pairwise_distances(initial_2d.coordinates) rtol = 1.0e-8
    @test final_2d.center_of_mass ≈ initial_2d.center_of_mass atol = 1.0e-10
    @test kinetic_energy(final_2d, system_2d) ≈
          kinetic_energy(initial_2d, system_2d) rtol = 1.0e-8
    @test final_2d.rotation.inertia * final_2d.rotation.angular_velocity ≈
          initial_2d.rotation.inertia * initial_2d.rotation.angular_velocity rtol = 1.0e-8

    # An isotropic 3D particle layout has constant angular velocity during torque-free motion.
    # This exercises tensor inertia and gyroscopic kinematics without requiring reference data.
    coordinates_3d = [1.0 -1.0 0.0 0.0 0.0 0.0
                      0.0 0.0 1.0 -1.0 0.0 0.0
                      0.0 0.0 0.0 0.0 1.0 -1.0]
    ic_3d = InitialCondition(; coordinates=coordinates_3d, mass=ones(6),
                             density=ones(6), particle_spacing=1.0)
    ic_3d = apply_angular_velocity(ic_3d, (0.7, -1.1, 0.9))
    system_3d = RigidBodySystem(ic_3d; acceleration=(0.0, 0.0, 0.0))
    semi_3d = Semidiscretization(system_3d, neighborhood_search=nothing)
    ode_3d = semidiscretize(semi_3d, (0.0, 1.0))
    sol_3d = solve(ode_3d, RDPK3SpFSAL49(); abstol=1.0e-10, reltol=1.0e-10,
                   dtmax=0.005, save_everystep=false)
    initial_3d = rigid_state(sol_3d.u[begin], system_3d, semi_3d)
    final_3d = rigid_state(sol_3d.u[end], system_3d, semi_3d)

    @test sol_3d.retcode == ReturnCode.Success
    @test pairwise_distances(final_3d.coordinates) ≈
          pairwise_distances(initial_3d.coordinates) rtol = 1.0e-8
    @test final_3d.center_of_mass ≈ initial_3d.center_of_mass atol = 1.0e-10
    @test final_3d.rotation.angular_velocity ≈
          initial_3d.rotation.angular_velocity rtol = 1.0e-8
    @test kinetic_energy(final_3d, system_3d) ≈
          kinetic_energy(initial_3d, system_3d) rtol = 1.0e-8
    @test final_3d.rotation.inertia * final_3d.rotation.angular_velocity ≈
          initial_3d.rotation.inertia * initial_3d.rotation.angular_velocity rtol = 1.0e-8

    # A nonuniform source field may generate force and torque, but the reduced acceleration
    # must keep all inter-particle distances fixed.
    source_terms = (coords, velocity, density, pressure, t) -> SVector(coords[2], 0.0)
    source_ic = InitialCondition(; coordinates=coordinates_2d, mass=ones(4),
                                 density=ones(4), particle_spacing=1.0)
    source_system = RigidBodySystem(source_ic; acceleration=(0.0, 0.0), source_terms)
    source_semi = Semidiscretization(source_system, neighborhood_search=nothing)
    source_ode = semidiscretize(source_semi, (0.0, 0.2))
    source_sol = solve(source_ode, RDPK3SpFSAL49(); abstol=1.0e-10, reltol=1.0e-10,
                       dtmax=0.001, save_everystep=false)
    source_initial = rigid_state(source_sol.u[begin], source_system, source_semi)
    source_final = rigid_state(source_sol.u[end], source_system, source_semi)

    @test source_sol.retcode == ReturnCode.Success
    @test count_rhs_allocations(source_sol) == 0
    @test pairwise_distances(source_final.coordinates) ≈
          pairwise_distances(source_initial.coordinates) rtol = 1.0e-8

    source_terms_3d = (coords, velocity, density, pressure,
                       t) -> SVector(coords[2], -coords[1], coords[3])
    source_ic_3d = InitialCondition(; coordinates=coordinates_3d, mass=ones(6),
                                    density=ones(6), particle_spacing=1.0)
    source_system_3d = RigidBodySystem(source_ic_3d; acceleration=(0.0, 0.0, 0.0),
                                       source_terms=source_terms_3d)
    source_semi_3d = Semidiscretization(source_system_3d, neighborhood_search=nothing)
    source_ode_3d = semidiscretize(source_semi_3d, (0.0, 0.1))
    source_sol_3d = solve(source_ode_3d, RDPK3SpFSAL49(); abstol=1.0e-10,
                          reltol=1.0e-10, dtmax=0.001, save_everystep=false)
    source_initial_3d = rigid_state(source_sol_3d.u[begin], source_system_3d,
                                    source_semi_3d)
    source_final_3d = rigid_state(source_sol_3d.u[end], source_system_3d,
                                  source_semi_3d)

    @test source_sol_3d.retcode == ReturnCode.Success
    @test count_rhs_allocations(source_sol_3d) == 0
    @test pairwise_distances(source_final_3d.coordinates) ≈
          pairwise_distances(source_initial_3d.coordinates) rtol = 1.0e-8

    # With zero damping, an isolated head-on penalty collision conserves total momentum and
    # returns all spring energy to translational kinetic energy after separation.
    collision_model = RigidContactModel(; normal_stiffness=100.0,
                                        contact_distance=0.1)
    collision_ic_1 = InitialCondition(; coordinates=reshape([-0.15, 0.0], 2, 1),
                                      velocity=reshape([0.5, 0.0], 2, 1), mass=[1.0],
                                      density=[1.0], particle_spacing=0.05)
    collision_ic_2 = InitialCondition(; coordinates=reshape([0.15, 0.0], 2, 1),
                                      velocity=reshape([-0.5, 0.0], 2, 1), mass=[1.0],
                                      density=[1.0], particle_spacing=0.05)
    collision_system_1 = RigidBodySystem(collision_ic_1; contact_model=collision_model,
                                         acceleration=(0.0, 0.0))
    collision_system_2 = RigidBodySystem(collision_ic_2; contact_model=collision_model,
                                         acceleration=(0.0, 0.0))
    collision_semi = Semidiscretization(collision_system_1, collision_system_2)
    collision_ode = semidiscretize(collision_semi, (0.0, 0.6))
    collision_sol = solve(collision_ode, RDPK3SpFSAL49(); abstol=1.0e-10,
                          reltol=1.0e-10, dtmax=0.0005, save_everystep=false)
    collision_initial_1 = rigid_state(collision_sol.u[begin], collision_system_1,
                                      collision_semi)
    collision_initial_2 = rigid_state(collision_sol.u[begin], collision_system_2,
                                      collision_semi)
    collision_final_1 = rigid_state(collision_sol.u[end], collision_system_1,
                                    collision_semi)
    collision_final_2 = rigid_state(collision_sol.u[end], collision_system_2,
                                    collision_semi)
    initial_momentum = collision_system_1.total_mass *
                       collision_initial_1.center_of_mass_velocity +
                       collision_system_2.total_mass *
                       collision_initial_2.center_of_mass_velocity
    final_momentum = collision_system_1.total_mass *
                     collision_final_1.center_of_mass_velocity +
                     collision_system_2.total_mass *
                     collision_final_2.center_of_mass_velocity
    initial_energy = kinetic_energy(collision_initial_1, collision_system_1) +
                     kinetic_energy(collision_initial_2, collision_system_2)
    final_energy = kinetic_energy(collision_final_1, collision_system_1) +
                   kinetic_energy(collision_final_2, collision_system_2)

    @test collision_sol.retcode == ReturnCode.Success
    @test final_momentum ≈ initial_momentum atol = 1.0e-9
    @test final_energy ≈ initial_energy rtol = 1.0e-7
    @test norm(collision_final_1.center_of_mass - collision_final_2.center_of_mass) >
          collision_model.contact_distance
end

@trixi_testset "Geometry-Aware Rigid-Wall Contact" begin
    function flat_wall_force(wall_spacing)
        contact_distance = 0.1
        normal_distance = 0.08
        floor = RectangularTank(wall_spacing, (0.0, 0.0), (1.0, wall_spacing),
                                1000.0; n_layers=1, min_coordinates=(-0.5, 0.0),
                                faces=(false, false, true, false))
        boundary_model = BoundaryModelMonaghanKajtar(10.0, 1.0, wall_spacing,
                                                     floor.boundary.mass)
        boundary_system = WallBoundarySystem(floor.boundary, boundary_model)

        rigid_y = normal_distance - wall_spacing / 2
        rigid_ic = InitialCondition(; coordinates=reshape([0.013, rigid_y], 2, 1),
                                    velocity=reshape([1.0, 0.0], 2, 1), mass=[1.0],
                                    density=[1.0], particle_spacing=0.05)
        contact_model = RigidContactModel(; normal_stiffness=100.0,
                                          contact_distance)
        rigid_system = RigidBodySystem(rigid_ic; contact_model,
                                       acceleration=(0.0, 0.0))
        semi = Semidiscretization(rigid_system, boundary_system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

        return rigid_system.resultant_force[], rigid_system,
               TrixiParticles.compact_support(rigid_system, boundary_system)
    end

    # Projected wall-normal distance makes a flat contact independent of tangential grid
    # alignment and wall resolution.
    coarse_force, coarse_system, coarse_support = flat_wall_force(0.1)
    fine_force, _, fine_support = flat_wall_force(0.05)
    @test coarse_force[1] ≈ 0.0 atol = eps()
    @test coarse_force[2] ≈ 2.0 rtol = 100 * eps()
    @test fine_force ≈ coarse_force rtol = 100 * eps()
    @test coarse_support ≈ hypot(coarse_system.contact_model.contact_distance, 0.1)
    @test fine_support ≈ hypot(coarse_system.contact_model.contact_distance, 0.05)

    # Prescribed rotation moves the tip of each stored normal with its wall particle.
    normal_ic = InitialCondition(; coordinates=zeros(2, 1), mass=[1.0], density=[1.0],
                                 particle_spacing=0.1,
                                 normals=reshape([0.0, -0.05], 2, 1))
    normal_model = BoundaryModelMonaghanKajtar(10.0, 1.0, 0.1, normal_ic.mass)
    movement_function(x,
                      t) = SVector(cospi(t / 2) * x[1] - sinpi(t / 2) * x[2] + t,
                                   sinpi(t / 2) * x[1] + cospi(t / 2) * x[2] - 2t)
    motion = PrescribedMotion(movement_function, t -> true)
    moving_wall = WallBoundarySystem(normal_ic, normal_model; prescribed_motion=motion)
    TrixiParticles.apply_prescribed_motion!(moving_wall, motion,
                                            DummySemidiscretization(), 1.0)
    @test moving_wall.cache.boundary_normals[:, 1] ≈ [0.05, 0.0] atol = 10 * eps()

    # Custom walls without normals keep radial pair geometry for backward compatibility.
    fallback_ic = InitialCondition(; coordinates=zeros(2, 1), mass=[1.0], density=[1.0],
                                   particle_spacing=0.1)
    fallback_wall = WallBoundarySystem(fallback_ic, normal_model)
    pos_diff = SVector(0.06, 0.08)
    fallback_normal,
    fallback_distance = TrixiParticles.rigid_wall_contact_geometry(fallback_wall, 1,
                                                                   pos_diff, 0.1)
    @test fallback_normal ≈ pos_diff / 0.1
    @test fallback_distance ≈ 0.1

    zero_normal_ic = InitialCondition(; coordinates=zeros(2, 1), mass=[1.0],
                                      density=[1.0], particle_spacing=0.1,
                                      normals=zeros(2, 1))
    zero_normal_wall = WallBoundarySystem(zero_normal_ic, normal_model)
    zero_normal,
    zero_normal_distance = TrixiParticles.rigid_wall_contact_geometry(zero_normal_wall, 1,
                                                                      pos_diff, 0.1)
    @test zero_normal ≈ fallback_normal
    @test zero_normal_distance ≈ fallback_distance

    # Perpendicular geometry normals at a corner must remain separate contact manifolds.
    corner_ic = InitialCondition(; coordinates=[0.0 -0.05
                                                -0.05 0.0], mass=ones(2),
                                 density=ones(2), particle_spacing=0.05,
                                 normals=[0.0 -0.05
                                          -0.05 0.0])
    corner_model = BoundaryModelMonaghanKajtar(10.0, 1.0, 0.05, corner_ic.mass)
    corner_wall = WallBoundarySystem(corner_ic, corner_model)
    corner_rigid_ic = InitialCondition(; coordinates=reshape([0.04, 0.04], 2, 1),
                                       mass=[1.0], density=[1.0], particle_spacing=0.05)
    corner_contact = RigidContactModel(; normal_stiffness=100.0,
                                       contact_distance=0.1)
    corner_rigid = RigidBodySystem(corner_rigid_ic; contact_model=corner_contact,
                                   acceleration=(0.0, 0.0), max_manifolds=2)
    corner_semi = Semidiscretization(corner_rigid, corner_wall)
    corner_ode = semidiscretize(corner_semi, (0.0, 0.01))
    corner_v_ode, corner_u_ode = corner_ode.u0.x
    corner_dv_ode = zero(corner_v_ode)
    TrixiParticles.kick!(corner_dv_ode, corner_v_ode, corner_u_ode, corner_ode.p, 0.0)

    @test corner_rigid.cache.contact_count[] == 2
    @test all(corner_rigid.resultant_force[] .> 0)
end
