@testset verbose=true "RigidSPHSystem" begin
    @trixi_testset "Constructor" begin
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

        system = RigidSPHSystem(initial_condition;
                                boundary_model=boundary_model,
                                acceleration=(0.0, -9.81),
                                particle_spacing=0.1)

        @test system isa RigidSPHSystem
        @test ndims(system) == 2
        @test system.initial_condition == initial_condition
        center_of_mass = [9.5 / 4.5, 9.5 / 4.5]
        @test system.local_coordinates ≈ coordinates .- center_of_mass
        @test system.mass == mass
        @test system.material_density == material_densities
        @test system.initial_velocity == initial_condition.velocity
        @test system.acceleration == [0.0, -9.81]
        @test iszero(system.initial_angular_velocity)
        @test system.particle_spacing == 0.1
        @test system.boundary_model == boundary_model
        @test TrixiParticles.v_nvariables(system) == 2

        dt = TrixiParticles.calculate_dt(zeros(2, 3), zeros(2, 3), 0.25, system,
                                         nothing)
        @test dt ≈ 0.25 * 0.1 / 9.81
    end

    @trixi_testset "Show" begin
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

        system = RigidSPHSystem(initial_condition;
                                boundary_model=boundary_model,
                                acceleration=(0.0, -9.81))

        compact_repr = repr(system)
        @test occursin("RigidSPHSystem{2}", compact_repr)
        @test occursin("with 2 particles", compact_repr)

        full_repr = repr("text/plain", system)
        @test occursin("RigidSPHSystem{2}", full_repr)
        @test occursin("#particles", full_repr)
        @test occursin("initial angular velocity", full_repr)
        @test occursin("boundary model", full_repr)
    end

    @trixi_testset "Initial Angular Velocity" begin
        coordinates_2d = [0.0 1.0
                          0.0 0.0]
        mass_2d = [1.0, 1.0]
        density_2d = [1000.0, 1000.0]
        ic_2d = InitialCondition(; coordinates=coordinates_2d, mass=mass_2d,
                                 density=density_2d)

        system_2d = RigidSPHSystem(ic_2d; angular_velocity=2.0)
        v0_2d = zeros(2, 2)
        TrixiParticles.write_v0!(v0_2d, system_2d)

        @test system_2d.initial_angular_velocity ≈ 2.0
        @test v0_2d ≈ [0.0 0.0
                       -1.0 1.0]
        @test_throws ArgumentError RigidSPHSystem(ic_2d; angular_velocity=(0.0, 1.0))

        coordinates_3d = [0.0 1.0
                          0.0 0.0
                          0.0 0.0]
        mass_3d = [1.0, 1.0]
        density_3d = [1000.0, 1000.0]
        ic_3d = InitialCondition(; coordinates=coordinates_3d, mass=mass_3d,
                                 density=density_3d)

        system_3d = RigidSPHSystem(ic_3d; angular_velocity=(0.0, 0.0, 2.0))
        v0_3d = zeros(3, 2)
        TrixiParticles.write_v0!(v0_3d, system_3d)

        @test system_3d.initial_angular_velocity ≈ [0.0, 0.0, 2.0]
        @test v0_3d ≈ [0.0 0.0
                       -1.0 1.0
                       0.0 0.0]
        @test_throws ArgumentError RigidSPHSystem(ic_3d; angular_velocity=2.0)
    end

    @trixi_testset "Rotational Kinematics" begin
        coordinates = [-1.0 1.0
                       0.0 0.0]
        velocity = [0.0 0.0
                    -1.0 1.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)
        rigid_system = RigidSPHSystem(initial_condition;
                                      acceleration=(0.0, 0.0))

        v = copy(velocity)
        u = copy(coordinates)
        TrixiParticles.update_final!(rigid_system, v, u, nothing, nothing, nothing, 0.0)

        @test rigid_system.cache.angular_velocity[] ≈ 1.0
        @test rigid_system.cache.inertia[] ≈ 2.0

        dv = zeros(size(v))
        for particle in TrixiParticles.each_integrated_particle(rigid_system)
            TrixiParticles.add_acceleration!(dv, particle, rigid_system)
        end

        @test dv ≈ [1.0 -1.0
                    0.0 0.0]
    end

    @trixi_testset "IO Data" begin
        coordinates = [-1.0 1.0
                       0.0 0.0]
        velocity = [0.0 0.0
                    -1.0 1.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)
        rigid_system = RigidSPHSystem(initial_condition; acceleration=(0.0, 0.0))

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

        @test :local_coordinates in fields
        @test :relative_coordinates in fields
        @test :center_of_mass in fields
        @test :center_of_mass_velocity in fields
        @test :angular_velocity in fields
        @test :resultant_force in fields
        @test :resultant_torque in fields
        @test :angular_acceleration_force in fields
        @test :gyroscopic_acceleration in fields
        @test :boundary_contact_count in fields
        @test :max_boundary_penetration in fields

        @test data.center_of_mass ≈ [0.0, 0.0]
        @test data.center_of_mass_velocity ≈ [0.0, 0.0]
        @test data.angular_velocity ≈ 1.0
        @test data.resultant_force ≈ [0.0, 0.0]
        @test data.resultant_torque ≈ 0.0
        @test data.angular_acceleration_force ≈ 0.0
        @test data.gyroscopic_acceleration ≈ 0.0
        @test data.boundary_contact_count == 0
        @test data.max_boundary_penetration == 0.0
        @test data.local_coordinates ≈ rigid_system.local_coordinates
        @test data.relative_coordinates ≈ rigid_system.cache.relative_coordinates
    end

    @trixi_testset "Velocity Components with ContinuityDensity" begin
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

        rigid_system = RigidSPHSystem(initial_condition;
                                      boundary_model=boundary_model)
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

    @trixi_testset "Boundary Contact Model" begin
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

        boundary_contact_model = RigidBoundaryContactModel(; normal_stiffness=2.0e4,
                                                           normal_damping=20.0,
                                                           static_friction_coefficient=0.6,
                                                           kinetic_friction_coefficient=0.4,
                                                           tangential_stiffness=200.0,
                                                           tangential_damping=5.0,
                                                           contact_distance=0.1,
                                                           stick_velocity_tolerance=1e-8)

        perfect_elastic_model = PerfectElasticBoundaryContactModel(; normal_stiffness=2.0e4,
                                                                    contact_distance=0.1,
                                                                    stick_velocity_tolerance=1e-8,
                                                                    torque_free=true)
        @test perfect_elastic_model isa PerfectElasticBoundaryContactModel
        @test perfect_elastic_model.normal_stiffness ≈ 2.0e4
        @test perfect_elastic_model.contact_distance ≈ 0.1
        @test perfect_elastic_model.contact_distance_factor ≈ 1.0
        @test perfect_elastic_model.stick_velocity_tolerance ≈ 1e-8
        @test perfect_elastic_model.torque_free

        perfect_elastic_model_runtime = TrixiParticles.convert_boundary_contact_model(perfect_elastic_model,
                                                                                       0.1, Float64)
        @test perfect_elastic_model_runtime isa RigidBoundaryContactModel
        @test perfect_elastic_model_runtime.normal_stiffness ≈ 2.0e4
        @test iszero(perfect_elastic_model_runtime.normal_damping)
        @test iszero(perfect_elastic_model_runtime.static_friction_coefficient)
        @test iszero(perfect_elastic_model_runtime.kinetic_friction_coefficient)
        @test iszero(perfect_elastic_model_runtime.tangential_stiffness)
        @test iszero(perfect_elastic_model_runtime.tangential_damping)
        @test perfect_elastic_model_runtime.contact_distance ≈ 0.1
        @test iszero(perfect_elastic_model_runtime.penetration_slop)
        @test perfect_elastic_model_runtime.torque_free
        @test !perfect_elastic_model_runtime.resting_contact_projection

        wall_material = (; youngs_modulus=3.0e10, poisson_ratio=0.2)
        wood_material = (; density=650.0, youngs_modulus=1.0e10, poisson_ratio=0.35,
                         restitution=0.35, friction_coefficient=0.5)
        contact_spec_2d = LinearizedHertzMindlinBoundaryContactModel(; material=wood_material,
                                                                      wall_material,
                                                                      radius=0.16,
                                                                      center=(0.45,
                                                                              1.5),
                                                                      gravity=9.81,
                                                                      particle_spacing=0.01,
                                                                      ndims=2,
                                                                      torque_free=true,
                                                                      resting_contact_projection=true)
        @test contact_spec_2d isa LinearizedHertzMindlinBoundaryContactModel
        @test contact_spec_2d.contact_distance_factor ≈ 2.0
        @test iszero(contact_spec_2d.contact_distance)

        contact_model_2d = TrixiParticles.convert_boundary_contact_model(contact_spec_2d,
                                                                         0.01, Float64)
        @test contact_model_2d isa RigidBoundaryContactModel
        @test contact_model_2d.normal_stiffness > 0
        @test contact_model_2d.normal_damping > 0
        @test contact_model_2d.tangential_stiffness > 0
        @test contact_model_2d.tangential_damping > 0
        @test contact_model_2d.contact_distance ≈ 0.02
        @test contact_model_2d.resting_contact_projection
        @test contact_model_2d.torque_free

        # Keep parity with the previous example-local 2D calibration.
        drop_height = max(1.5 - 0.16, 0.01)
        impact_velocity_legacy = sqrt(2.0 * 9.81 * drop_height)
        shear_wood = wood_material.youngs_modulus / (2.0 * (1.0 + wood_material.poisson_ratio))
        shear_wall = wall_material.youngs_modulus / (2.0 * (1.0 + wall_material.poisson_ratio))
        effective_E_legacy = 1.0 / ((1.0 - wood_material.poisson_ratio^2) /
                             wood_material.youngs_modulus +
                             (1.0 - wall_material.poisson_ratio^2) /
                             wall_material.youngs_modulus)
        effective_G_legacy = 1.0 / ((2.0 - wood_material.poisson_ratio) / shear_wood +
                             (2.0 - wall_material.poisson_ratio) / shear_wall)
        mass_legacy = wood_material.density * pi * 0.16^2
        hertz_coeff_legacy = (4.0 / 3.0) * effective_E_legacy * sqrt(0.16)
        reference_penetration_legacy = ((5.0 / 4.0) * mass_legacy *
                                        impact_velocity_legacy^2 /
                                        hertz_coeff_legacy)^(2.0 / 5.0)
        reference_penetration_legacy = max(reference_penetration_legacy, 0.01 * 0.01)
        normal_stiffness_legacy = (3.0 / 2.0) * hertz_coeff_legacy *
                                  sqrt(reference_penetration_legacy)
        tangential_stiffness_legacy = 8.0 * effective_G_legacy *
                                      sqrt(0.16 * reference_penetration_legacy)
        damping_ratio_legacy = let restitution_clamped = clamp(wood_material.restitution,
                                                                0.0, 1.0)
            if restitution_clamped >= 1.0
                0.0
            elseif restitution_clamped <= eps(restitution_clamped)
                1.0
            else
                log_restitution = log(restitution_clamped)
                -log_restitution / sqrt(pi^2 + log_restitution^2)
            end
        end
        normal_damping_legacy = 2.0 * damping_ratio_legacy *
                                sqrt(normal_stiffness_legacy * mass_legacy)
        tangential_damping_legacy = 2.0 * damping_ratio_legacy *
                                    sqrt(tangential_stiffness_legacy * mass_legacy)
        stick_tol_legacy = max(1e-5, 0.01 * impact_velocity_legacy)

        @test contact_model_2d.normal_stiffness ≈ normal_stiffness_legacy rtol=1e-12
        @test contact_model_2d.tangential_stiffness ≈ tangential_stiffness_legacy rtol=1e-12
        @test contact_model_2d.normal_damping ≈ normal_damping_legacy rtol=1e-12
        @test contact_model_2d.tangential_damping ≈ tangential_damping_legacy rtol=1e-12
        @test contact_model_2d.stick_velocity_tolerance ≈ stick_tol_legacy rtol=1e-12

        contact_spec_3d = LinearizedHertzMindlinBoundaryContactModel(; material=wood_material,
                                                                      wall_material,
                                                                      radius=0.16,
                                                                      impact_velocity=5.0,
                                                                      particle_spacing=0.01,
                                                                      ndims=3)
        contact_model_3d = TrixiParticles.convert_boundary_contact_model(contact_spec_3d,
                                                                         0.01, Float64)
        @test contact_model_3d isa RigidBoundaryContactModel
        @test contact_model_3d.normal_stiffness > 0
        @test contact_model_3d.normal_damping > 0
        @test contact_model_3d.tangential_stiffness > 0
        @test contact_model_3d.tangential_damping > 0
        @test contact_model_3d.contact_distance ≈ 0.02

        frictionless_material = (; density=1200.0, youngs_modulus=2.0e9, poisson_ratio=0.35,
                                 restitution=1.0, friction_coefficient=0.0)
        frictionless_spec = LinearizedHertzMindlinBoundaryContactModel(; material=frictionless_material,
                                                                        wall_material,
                                                                        radius=0.16,
                                                                        impact_velocity=2.0,
                                                                        particle_spacing=0.01,
                                                                        ndims=2)
        frictionless_model = TrixiParticles.convert_boundary_contact_model(frictionless_spec,
                                                                           0.01, Float64)
        @test iszero(frictionless_model.static_friction_coefficient)
        @test iszero(frictionless_model.kinetic_friction_coefficient)
        @test iszero(frictionless_model.tangential_stiffness)
        @test iszero(frictionless_model.tangential_damping)

        restitution_low = (; density=650.0, youngs_modulus=1.0e10, poisson_ratio=0.35,
                           restitution=0.3, friction_coefficient=0.5)
        restitution_high = (; density=650.0, youngs_modulus=1.0e10, poisson_ratio=0.35,
                            restitution=0.8, friction_coefficient=0.5)
        model_restitution_low_spec = LinearizedHertzMindlinBoundaryContactModel(; material=restitution_low,
                                                                                 wall_material,
                                                                                 radius=0.16,
                                                                                 impact_velocity=2.0,
                                                                                 particle_spacing=0.01,
                                                                                 ndims=3)
        model_restitution_high_spec = LinearizedHertzMindlinBoundaryContactModel(; material=restitution_high,
                                                                                  wall_material,
                                                                                  radius=0.16,
                                                                                  impact_velocity=2.0,
                                                                                  particle_spacing=0.01,
                                                                                  ndims=3)
        model_restitution_low = TrixiParticles.convert_boundary_contact_model(model_restitution_low_spec,
                                                                              0.01, Float64)
        model_restitution_high = TrixiParticles.convert_boundary_contact_model(model_restitution_high_spec,
                                                                               0.01, Float64)
        @test model_restitution_low.normal_damping > model_restitution_high.normal_damping

        rigid_system = RigidSPHSystem(rigid_ic;
                                      acceleration=(0.0, 0.0),
                                      boundary_contact_model=boundary_contact_model)

        @test TrixiParticles.requires_update_callback(rigid_system)

        semi = Semidiscretization(rigid_system, boundary_system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)

        TrixiParticles.interact!(dv_ode, v_ode, u_ode, rigid_system, boundary_system, semi)
        dv = TrixiParticles.wrap_v(dv_ode, rigid_system, semi)

        @test dv[2, 1] > 0
        @test dv[1, 1] < 0
        @test rigid_system.cache.boundary_contact_count[] > 0
        @test rigid_system.cache.max_boundary_penetration[] > 0

        v_rigid = TrixiParticles.wrap_v(v_ode, rigid_system, semi)
        u_rigid = TrixiParticles.wrap_u(u_ode, rigid_system, semi)
        v_wall = TrixiParticles.wrap_v(v_ode, boundary_system, semi)
        u_wall = TrixiParticles.wrap_u(u_ode, boundary_system, semi)

        v_rigid[1, 1] = 0.0
        v_rigid[2, 1] = -0.2
        u_rigid_before = copy(u_rigid)
        v_rigid_before = copy(v_rigid)
        active_keys = Set{NTuple{3, Int}}()

        modified = TrixiParticles.apply_boundary_contact_correction!(rigid_system,
                                                                     boundary_system,
                                                                     v_rigid, u_rigid,
                                                                     v_wall, u_wall,
                                                                     semi, 1e-3, active_keys)

        @test !modified
        @test u_rigid ≈ u_rigid_before
        @test v_rigid ≈ v_rigid_before
        @test !isempty(active_keys)
        @test !isempty(rigid_system.cache.contact_tangential_displacement)

        # Tangential history limiter must use the same slop-adjusted penetration
        # that is used by the normal-force path.
        boundary_contact_model_slop = RigidBoundaryContactModel(; normal_stiffness=2.0e4,
                                                                normal_damping=0.0,
                                                                static_friction_coefficient=0.6,
                                                                kinetic_friction_coefficient=0.4,
                                                                tangential_stiffness=2.0e6,
                                                                tangential_damping=0.0,
                                                                contact_distance=0.1,
                                                                penetration_slop=0.04,
                                                                stick_velocity_tolerance=1e-8)
        rigid_system_slop = RigidSPHSystem(rigid_ic;
                                           acceleration=(0.0, 0.0),
                                           boundary_contact_model=boundary_contact_model_slop)
        semi_slop = Semidiscretization(rigid_system_slop, boundary_system)
        ode_slop = semidiscretize(semi_slop, (0.0, 0.01))
        v_ode_slop, u_ode_slop = ode_slop.u0.x
        v_rigid_slop = TrixiParticles.wrap_v(v_ode_slop, rigid_system_slop, semi_slop)
        u_rigid_slop = TrixiParticles.wrap_u(u_ode_slop, rigid_system_slop, semi_slop)
        v_wall_slop = TrixiParticles.wrap_v(v_ode_slop, boundary_system, semi_slop)
        u_wall_slop = TrixiParticles.wrap_u(u_ode_slop, boundary_system, semi_slop)
        active_keys_slop = Set{NTuple{3, Int}}()

        TrixiParticles.apply_boundary_contact_correction!(rigid_system_slop, boundary_system,
                                                          v_rigid_slop, u_rigid_slop,
                                                          v_wall_slop, u_wall_slop,
                                                          semi_slop, 1e-3,
                                                          active_keys_slop)

        @test length(active_keys_slop) == 1
        contact_key = first(active_keys_slop)
        stored_displacement = rigid_system_slop.cache.contact_tangential_displacement[contact_key]
        stored_displacement_norm = LinearAlgebra.norm(stored_displacement)
        distance = LinearAlgebra.norm(rigid_coordinates[:, 1] - boundary_coordinates[:, 1])
        penetration = boundary_contact_model_slop.contact_distance - distance
        penetration_effective = penetration - boundary_contact_model_slop.penetration_slop

        normal_slop = (rigid_coordinates[:, 1] - boundary_coordinates[:, 1]) / distance
        relative_velocity_slop = TrixiParticles.current_velocity(v_rigid_slop, rigid_system_slop,
                                                                 1) -
                                 TrixiParticles.current_velocity(v_wall_slop, boundary_system, 1)
        normal_velocity_slop = LinearAlgebra.dot(relative_velocity_slop, normal_slop)
        normal_force_friction_reference_slop = TrixiParticles.normal_friction_reference_force(boundary_contact_model_slop,
                                                                                               penetration_effective,
                                                                                               normal_velocity_slop,
                                                                                               eltype(rigid_system_slop))

        expected_displacement_cap = boundary_contact_model_slop.static_friction_coefficient *
                                    normal_force_friction_reference_slop /
                                    boundary_contact_model_slop.tangential_stiffness
        expected_raw_displacement_cap = boundary_contact_model_slop.static_friction_coefficient *
                                        boundary_contact_model_slop.normal_stiffness *
                                        penetration /
                                        boundary_contact_model_slop.tangential_stiffness

        @test stored_displacement_norm ≈ expected_displacement_cap atol=1e-12
        @test stored_displacement_norm < expected_raw_displacement_cap

        # The tangential-history cap must use the same normal-load definition as the
        # tangential force path, including damping at high approach speed.
        rigid_ic_damped = InitialCondition(; coordinates=rigid_coordinates,
                                           velocity=reshape([1.0, -2.0], 2, 1),
                                           mass=rigid_mass,
                                           density=rigid_density,
                                           particle_spacing=0.1)
        boundary_contact_model_damped = RigidBoundaryContactModel(; normal_stiffness=2.0e4,
                                                                  normal_damping=40.0,
                                                                  static_friction_coefficient=0.6,
                                                                  kinetic_friction_coefficient=0.4,
                                                                  tangential_stiffness=2.0e6,
                                                                  tangential_damping=0.0,
                                                                  contact_distance=0.1,
                                                                  penetration_slop=0.0,
                                                                  stick_velocity_tolerance=1e-8)
        rigid_system_damped = RigidSPHSystem(rigid_ic_damped;
                                             acceleration=(0.0, 0.0),
                                             boundary_contact_model=boundary_contact_model_damped)
        semi_damped = Semidiscretization(rigid_system_damped, boundary_system)
        ode_damped = semidiscretize(semi_damped, (0.0, 0.01))
        v_ode_damped, u_ode_damped = ode_damped.u0.x
        v_rigid_damped = TrixiParticles.wrap_v(v_ode_damped, rigid_system_damped, semi_damped)
        u_rigid_damped = TrixiParticles.wrap_u(u_ode_damped, rigid_system_damped, semi_damped)
        v_wall_damped = TrixiParticles.wrap_v(v_ode_damped, boundary_system, semi_damped)
        u_wall_damped = TrixiParticles.wrap_u(u_ode_damped, boundary_system, semi_damped)
        active_keys_damped = Set{NTuple{3, Int}}()

        TrixiParticles.apply_boundary_contact_correction!(rigid_system_damped, boundary_system,
                                                          v_rigid_damped, u_rigid_damped,
                                                          v_wall_damped, u_wall_damped,
                                                          semi_damped, 1e-3,
                                                          active_keys_damped)

        @test length(active_keys_damped) == 1
        damped_key = first(active_keys_damped)
        damped_displacement = rigid_system_damped.cache.contact_tangential_displacement[damped_key]
        damped_displacement_norm = LinearAlgebra.norm(damped_displacement)
        damped_distance = LinearAlgebra.norm(rigid_coordinates[:, 1] - boundary_coordinates[:, 1])
        damped_penetration = boundary_contact_model_damped.contact_distance - damped_distance
        damped_penetration_effective = damped_penetration -
                                       boundary_contact_model_damped.penetration_slop
        damped_normal = (rigid_coordinates[:, 1] - boundary_coordinates[:, 1]) / damped_distance
        relative_velocity_damped = TrixiParticles.current_velocity(v_rigid_damped,
                                                                   rigid_system_damped, 1) -
                                   TrixiParticles.current_velocity(v_wall_damped, boundary_system,
                                                                   1)
        normal_velocity_damped = LinearAlgebra.dot(relative_velocity_damped, damped_normal)
        normal_force_friction_reference_damped = TrixiParticles.normal_friction_reference_force(boundary_contact_model_damped,
                                                                                                 damped_penetration_effective,
                                                                                                 normal_velocity_damped,
                                                                                                 eltype(rigid_system_damped))
        expected_damped_cap = boundary_contact_model_damped.static_friction_coefficient *
                              normal_force_friction_reference_damped /
                              boundary_contact_model_damped.tangential_stiffness
        elastic_only_damped_cap = boundary_contact_model_damped.static_friction_coefficient *
                                  boundary_contact_model_damped.normal_stiffness *
                                  damped_penetration_effective /
                                  boundary_contact_model_damped.tangential_stiffness
        total_normal_force_damped = TrixiParticles.normal_contact_force(boundary_contact_model_damped,
                                                                        damped_penetration_effective,
                                                                        normal_velocity_damped,
                                                                        eltype(rigid_system_damped))
        total_force_damped_cap = boundary_contact_model_damped.static_friction_coefficient *
                                 total_normal_force_damped /
                                 boundary_contact_model_damped.tangential_stiffness

        @test damped_displacement_norm ≈ expected_damped_cap atol=1e-12
        @test damped_displacement_norm ≈ total_force_damped_cap atol=1e-12
        @test damped_displacement_norm > elastic_only_damped_cap
    end

    @trixi_testset "Projection Fallback" begin
        rigid_coordinates = reshape([0.0, 0.05], 2, 1)
        rigid_velocity = reshape([2.0e-3, -2.0e-3], 2, 1)
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

        boundary_contact_model = RigidBoundaryContactModel(; normal_stiffness=2.0e4,
                                                           normal_damping=20.0,
                                                           static_friction_coefficient=0.6,
                                                           kinetic_friction_coefficient=0.4,
                                                           tangential_stiffness=1.0e4,
                                                           tangential_damping=5.0,
                                                           contact_distance=0.1,
                                                           stick_velocity_tolerance=1e-2)
        rigid_system = RigidSPHSystem(rigid_ic;
                                      acceleration=(0.0, 0.0),
                                      boundary_contact_model=boundary_contact_model)
        semi = Semidiscretization(rigid_system, boundary_system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        v_rigid = TrixiParticles.wrap_v(v_ode, rigid_system, semi)
        u_rigid = TrixiParticles.wrap_u(u_ode, rigid_system, semi)
        v_wall = TrixiParticles.wrap_v(v_ode, boundary_system, semi)
        u_wall = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
        TrixiParticles.update_final!(rigid_system, v_rigid, u_rigid, v_ode, u_ode, semi, 0.0)

        active_keys = Set{NTuple{3, Int}}()
        TrixiParticles.apply_boundary_contact_correction!(rigid_system, boundary_system,
                                                          v_rigid, u_rigid,
                                                          v_wall, u_wall,
                                                          semi, 1e-3, active_keys)

        @test rigid_system.cache.boundary_contact_count[] > 0
        @test !isempty(active_keys)
        @test !isempty(rigid_system.cache.contact_tangential_displacement)

        dt_contact = TrixiParticles.contact_time_step(rigid_system)
        modified = TrixiParticles.project_resting_contact_velocity!(rigid_system,
                                                                    v_rigid, u_rigid,
                                                                    v_ode, u_ode,
                                                                    semi,
                                                                    (; dt=0.01 *
                                                                           dt_contact))

        @test modified

        system_coords = TrixiParticles.current_coordinates(u_rigid, rigid_system)
        wall_coords = TrixiParticles.current_coordinates(u_wall, boundary_system)
        normal = system_coords[:, 1] - wall_coords[:, 1]
        normal ./= LinearAlgebra.norm(normal)
        relative_velocity = TrixiParticles.current_velocity(v_rigid, rigid_system, 1) -
                            TrixiParticles.current_velocity(v_wall, boundary_system, 1)
        velocity_floor = max(0.1 * boundary_contact_model.stick_velocity_tolerance,
                             sqrt(eps(eltype(rigid_system))))
        @test LinearAlgebra.dot(relative_velocity, normal) >= -velocity_floor -
              100 * eps(velocity_floor)

        for key in active_keys
            @test !haskey(rigid_system.cache.contact_tangential_displacement, key)
        end

        rigid_system_no_projection = RigidSPHSystem(rigid_ic;
                                                    acceleration=(0.0, 0.0),
                                                    boundary_contact_model=boundary_contact_model)
        semi_no_projection = Semidiscretization(rigid_system_no_projection, boundary_system)
        ode_no_projection = semidiscretize(semi_no_projection, (0.0, 0.01))
        v_ode_no_projection, u_ode_no_projection = ode_no_projection.u0.x
        v_rigid_no_projection = TrixiParticles.wrap_v(v_ode_no_projection,
                                                      rigid_system_no_projection,
                                                      semi_no_projection)
        u_rigid_no_projection = TrixiParticles.wrap_u(u_ode_no_projection,
                                                      rigid_system_no_projection,
                                                      semi_no_projection)
        v_wall_no_projection = TrixiParticles.wrap_v(v_ode_no_projection, boundary_system,
                                                     semi_no_projection)
        u_wall_no_projection = TrixiParticles.wrap_u(u_ode_no_projection, boundary_system,
                                                     semi_no_projection)
        TrixiParticles.update_final!(rigid_system_no_projection,
                                     v_rigid_no_projection,
                                     u_rigid_no_projection,
                                     v_ode_no_projection,
                                     u_ode_no_projection,
                                     semi_no_projection, 0.0)

        active_keys_no_projection = Set{NTuple{3, Int}}()
        TrixiParticles.apply_boundary_contact_correction!(rigid_system_no_projection,
                                                          boundary_system,
                                                          v_rigid_no_projection,
                                                          u_rigid_no_projection,
                                                          v_wall_no_projection,
                                                          u_wall_no_projection,
                                                          semi_no_projection, 1e-3,
                                                          active_keys_no_projection)

        modified_no_projection = TrixiParticles.project_resting_contact_velocity!(rigid_system_no_projection,
                                                                                  v_rigid_no_projection,
                                                                                  u_rigid_no_projection,
                                                                                  v_ode_no_projection,
                                                                                  u_ode_no_projection,
                                                                                  semi_no_projection,
                                                                                  (; dt=0.2 *
                                                                                         dt_contact))

        @test !modified_no_projection

        function effective_penetration(system, u_state, wall_system, u_wall_state, contact_model)
            system_coords_local = TrixiParticles.current_coordinates(u_state, system)
            wall_coords_local = TrixiParticles.current_coordinates(u_wall_state, wall_system)
            distance_local = LinearAlgebra.norm(system_coords_local[:, 1] - wall_coords_local[:, 1])
            penetration_local = contact_model.contact_distance - distance_local

            return max(zero(eltype(system)),
                       penetration_local - contact_model.penetration_slop)
        end

        penetration_before_persistent = effective_penetration(rigid_system_no_projection,
                                                              u_rigid_no_projection,
                                                              boundary_system,
                                                              u_wall_no_projection,
                                                              boundary_contact_model)
        modified_persistent = TrixiParticles.project_resting_contact_velocity!(rigid_system_no_projection,
                                                                               v_rigid_no_projection,
                                                                               u_rigid_no_projection,
                                                                               v_ode_no_projection,
                                                                               u_ode_no_projection,
                                                                               semi_no_projection,
                                                                               (; dt=0.2 *
                                                                                      dt_contact))
        penetration_after_persistent = effective_penetration(rigid_system_no_projection,
                                                             u_rigid_no_projection,
                                                             boundary_system,
                                                             u_wall_no_projection,
                                                             boundary_contact_model)

        @test modified_persistent
        @test penetration_after_persistent <= penetration_before_persistent
        @test penetration_after_persistent <=
              1.01e-3 * boundary_contact_model.contact_distance

        # Off-center resting contact should project both translational and
        # angular rigid-body velocity components.
        rigid_coordinates_angular = [0.0 0.12
                                     0.05 0.2]
        rigid_velocity_angular = [0.0 0.0
                                  -3.0e-3 -3.0e-3]
        rigid_mass_angular = [1.0, 1.0]
        rigid_density_angular = [1000.0, 1000.0]
        rigid_ic_angular = InitialCondition(; coordinates=rigid_coordinates_angular,
                                            velocity=rigid_velocity_angular,
                                            mass=rigid_mass_angular,
                                            density=rigid_density_angular,
                                            particle_spacing=0.1)
        rigid_system_angular = RigidSPHSystem(rigid_ic_angular;
                                              acceleration=(0.0, 0.0),
                                              boundary_contact_model=boundary_contact_model)
        semi_angular = Semidiscretization(rigid_system_angular, boundary_system)
        ode_angular = semidiscretize(semi_angular, (0.0, 0.01))
        v_ode_angular, u_ode_angular = ode_angular.u0.x
        v_rigid_angular = TrixiParticles.wrap_v(v_ode_angular, rigid_system_angular, semi_angular)
        u_rigid_angular = TrixiParticles.wrap_u(u_ode_angular, rigid_system_angular, semi_angular)
        v_wall_angular = TrixiParticles.wrap_v(v_ode_angular, boundary_system, semi_angular)
        u_wall_angular = TrixiParticles.wrap_u(u_ode_angular, boundary_system, semi_angular)
        TrixiParticles.update_final!(rigid_system_angular, v_rigid_angular, u_rigid_angular,
                                     v_ode_angular, u_ode_angular, semi_angular, 0.0)

        active_keys_angular = Set{NTuple{3, Int}}()
        TrixiParticles.apply_boundary_contact_correction!(rigid_system_angular, boundary_system,
                                                          v_rigid_angular, u_rigid_angular,
                                                          v_wall_angular, u_wall_angular,
                                                          semi_angular, 1e-3,
                                                          active_keys_angular)
        @test rigid_system_angular.cache.boundary_contact_count[] > 0

        dt_contact_angular = TrixiParticles.contact_time_step(rigid_system_angular)
        angular_before = rigid_system_angular.cache.angular_velocity[]
        modified_angular = TrixiParticles.project_resting_contact_velocity!(rigid_system_angular,
                                                                            v_rigid_angular,
                                                                            u_rigid_angular,
                                                                            v_ode_angular,
                                                                            u_ode_angular,
                                                                            semi_angular,
                                                                            (; dt=0.01 *
                                                                                   dt_contact_angular))
        angular_after = rigid_system_angular.cache.angular_velocity[]
        @test modified_angular
        @test abs(angular_after) > abs(angular_before) + 100 * eps(abs(angular_before))

        system_coords_angular = TrixiParticles.current_coordinates(u_rigid_angular,
                                                                   rigid_system_angular)
        wall_coords_angular = TrixiParticles.current_coordinates(u_wall_angular, boundary_system)
        normal_angular = system_coords_angular[:, 1] - wall_coords_angular[:, 1]
        normal_angular ./= LinearAlgebra.norm(normal_angular)
        relative_velocity_angular = TrixiParticles.current_velocity(v_rigid_angular,
                                                                    rigid_system_angular, 1) -
                                    TrixiParticles.current_velocity(v_wall_angular,
                                                                    boundary_system, 1)
        @test LinearAlgebra.dot(relative_velocity_angular, normal_angular) >=
              -velocity_floor - 100 * eps(velocity_floor)
    end

    @trixi_testset "Rigid-Wall Collision Shape Preservation" begin
        using LinearAlgebra: norm
        using OrdinaryDiffEq
        using Statistics: std

        particle_spacing = 0.1
        sphere = SphereShape(particle_spacing, 0.2, (0.0, 0.35), 3000.0,
                             sphere_type=VoxelSphere())

        wall_x = collect(-0.7:particle_spacing:0.7)
        wall_coordinates = zeros(2, length(wall_x))
        wall_coordinates[1, :] .= wall_x
        wall_mass = fill(1000.0 * particle_spacing^2, length(wall_x))
        wall_density = fill(1000.0, length(wall_x))
        wall_ic = InitialCondition(; coordinates=wall_coordinates,
                                   mass=wall_mass,
                                   density=wall_density,
                                   particle_spacing=particle_spacing)

        state_equation = StateEquationCole(; sound_speed=20.0,
                                           reference_density=1000.0,
                                           exponent=1.0)
        boundary_model = BoundaryModelDummyParticles(wall_density, wall_mass,
                                                     BernoulliPressureExtrapolation(),
                                                     WendlandC2Kernel{2}(),
                                                     1.5 * particle_spacing;
                                                     state_equation=state_equation)
        wall_system = WallBoundarySystem(wall_ic, boundary_model)

        boundary_contact_model = RigidBoundaryContactModel(; normal_stiffness=2.0e4,
                                                           normal_damping=200.0,
                                                           static_friction_coefficient=0.3,
                                                           kinetic_friction_coefficient=0.2,
                                                           tangential_stiffness=1.0e3,
                                                           tangential_damping=20.0,
                                                           contact_distance=particle_spacing,
                                                           penetration_slop=0.2 *
                                                                            particle_spacing)

        rigid_system = RigidSPHSystem(sphere;
                                      acceleration=(0.0, -9.81),
                                      boundary_contact_model=boundary_contact_model,
                                      particle_spacing=particle_spacing)

        initial_relative_coordinates = copy(rigid_system.local_coordinates)

        semi = Semidiscretization(rigid_system, wall_system)
        ode = semidiscretize(semi, (0.0, 0.2))
        solution = solve(ode, RDPK3SpFSAL35();
                         abstol=1e-6, reltol=1e-3,
                         save_everystep=false,
                         callback=UpdateCallback())

        _, u_end = solution.u[end].x
        v_end, _ = solution.u[end].x
        u_rigid_end = TrixiParticles.wrap_u(u_end, rigid_system, semi)
        v_rigid_end = TrixiParticles.wrap_v(v_end, rigid_system, semi)
        final_coordinates = TrixiParticles.current_coordinates(u_rigid_end, rigid_system)
        final_velocity = TrixiParticles.current_velocity(v_rigid_end, rigid_system)

        center_of_mass = zeros(eltype(final_coordinates), 2)
        total_mass = sum(rigid_system.mass)
        for particle in eachindex(rigid_system.mass)
            center_of_mass .+= rigid_system.mass[particle] .* final_coordinates[:, particle]
        end
        center_of_mass ./= total_mass

        final_relative_coordinates = similar(final_coordinates)
        for particle in axes(final_coordinates, 2)
            final_relative_coordinates[:, particle] .= final_coordinates[:, particle] .-
                                                      center_of_mass
        end

        max_pairwise_distance_error = zero(eltype(final_coordinates))
        max_initial_distance = zero(eltype(final_coordinates))
        for i in axes(initial_relative_coordinates, 2)
            for j in (i + 1):size(initial_relative_coordinates, 2)
                initial_distance = norm(initial_relative_coordinates[:, i] -
                                        initial_relative_coordinates[:, j])
                final_distance = norm(final_relative_coordinates[:, i] -
                                      final_relative_coordinates[:, j])
                max_pairwise_distance_error = max(max_pairwise_distance_error,
                                                  abs(final_distance - initial_distance))
                max_initial_distance = max(max_initial_distance, initial_distance)
            end
        end

        # Ensure the rigid body reaches wall-contact regime in this setup.
        @test minimum(final_coordinates[2, :]) < boundary_contact_model.contact_distance

        relative_shape_error = max_pairwise_distance_error / max_initial_distance
        @test relative_shape_error < 1.0e-10

        # For this symmetric setup without initial rotation, all particles should
        # move with (nearly) identical vertical velocity after impact.
        @test std(vec(final_velocity[2, :])) < 1.0e-10
    end

    @trixi_testset "Perfect Elastic Rigid-Wall Energy Conservation" begin
        using LinearAlgebra: dot
        using OrdinaryDiffEq

        particle_spacing = 0.01
        sphere = SphereShape(particle_spacing, 0.1, (0.0, 0.25), 1200.0;
                             sphere_type=RoundSphere(), velocity=(0.0, -1.0))

        wall_x = collect(-0.4:particle_spacing:0.4)
        wall_coordinates = zeros(2, length(wall_x))
        wall_coordinates[1, :] .= wall_x
        wall_mass = fill(1000.0 * particle_spacing^2, length(wall_x))
        wall_density = fill(1000.0, length(wall_x))
        wall_ic = InitialCondition(; coordinates=wall_coordinates,
                                   mass=wall_mass,
                                   density=wall_density,
                                   particle_spacing=particle_spacing)

        state_equation = StateEquationCole(; sound_speed=20.0,
                                           reference_density=1000.0,
                                           exponent=1.0)
        wall_model = BoundaryModelDummyParticles(wall_density, wall_mass,
                                                 BernoulliPressureExtrapolation(),
                                                 WendlandC2Kernel{2}(),
                                                 1.5 * particle_spacing;
                                                 state_equation=state_equation)
        wall_system = WallBoundarySystem(wall_ic, wall_model)

        boundary_contact_model = PerfectElasticBoundaryContactModel(; normal_stiffness=2.0e5,
                                                                    contact_distance=2.0 *
                                                                                     particle_spacing,
                                                                    torque_free=true,
                                                                    stick_velocity_tolerance=1e-8)

        rigid_system = RigidSPHSystem(sphere;
                                      acceleration=(0.0, 0.0),
                                      boundary_contact_model=boundary_contact_model,
                                      particle_spacing=particle_spacing)

        semi = Semidiscretization(rigid_system, wall_system)
        ode = semidiscretize(semi, (0.0, 0.6))
        solution = solve(ode, RDPK3SpFSAL35();
                         abstol=1e-8, reltol=1e-6,
                         save_everystep=true,
                         callback=UpdateCallback())

        @test solution.retcode == ReturnCode.Success
        @test length(solution.t) > 2

        function total_kinetic_energy(v_state)
            v_rigid = TrixiParticles.wrap_v(v_state, rigid_system, semi)
            velocity = TrixiParticles.current_velocity(v_rigid, rigid_system)
            kinetic_energy = zero(eltype(rigid_system))

            for particle in eachindex(rigid_system.mass)
                particle_velocity = velocity[:, particle]
                kinetic_energy += 0.5 * rigid_system.mass[particle] *
                                  dot(particle_velocity, particle_velocity)
            end

            return kinetic_energy
        end

        function mean_vertical_velocity(v_state)
            v_rigid = TrixiParticles.wrap_v(v_state, rigid_system, semi)
            velocity = TrixiParticles.current_velocity(v_rigid, rigid_system)

            return sum(velocity[2, :]) / size(velocity, 2)
        end

        function minimum_wall_clearance(u_state)
            u_rigid = TrixiParticles.wrap_u(u_state, rigid_system, semi)
            coordinates = TrixiParticles.current_coordinates(u_rigid, rigid_system)
            wall_top = maximum(wall_coordinates[2, :])

            return minimum(coordinates[2, :]) - wall_top
        end

        function center_of_mass_x(u_state)
            u_rigid = TrixiParticles.wrap_u(u_state, rigid_system, semi)
            coordinates = TrixiParticles.current_coordinates(u_rigid, rigid_system)
            total_mass = sum(rigid_system.mass)

            return sum(rigid_system.mass .* coordinates[1, :]) / total_mass
        end

        initial_ke = total_kinetic_energy(solution.u[1].x[1])
        final_ke = total_kinetic_energy(solution.u[end].x[1])
        relative_ke_error = abs(final_ke - initial_ke) / initial_ke

        vertical_velocity_history = [mean_vertical_velocity(state.x[1]) for state in solution.u]
        center_of_mass_x_history = [center_of_mass_x(state.x[2]) for state in solution.u]
        max_abs_com_x_drift = maximum(abs(com_x - center_of_mass_x_history[1])
                                      for com_x in center_of_mass_x_history)
        final_clearance = minimum_wall_clearance(solution.u[end].x[2])

        # Ensure this is an actual bounce (sign change in vertical velocity)
        # and that final state is separated from the wall.
        @test minimum(vertical_velocity_history) < -0.9
        @test maximum(vertical_velocity_history) > 0.8
        @test vertical_velocity_history[end] > 0.8
        @test final_clearance > 1.2 * boundary_contact_model.contact_distance
        @test max_abs_com_x_drift < 1e-6

        @test relative_ke_error < 1e-3
    end

    @trixi_testset "Boundary Contact Resolution Invariance" begin
        using OrdinaryDiffEq

        function make_wall_system(wall_spacing, n_layers)
            wall_x = collect(-0.8:wall_spacing:0.8)
            n_per_layer = length(wall_x)
            n_wall_particles = n_per_layer * n_layers
            wall_coordinates = zeros(2, n_wall_particles)
            wall_mass = fill(1000.0 * wall_spacing^2, n_wall_particles)
            wall_density = fill(1000.0, n_wall_particles)

            index = 1
            for layer in 0:(n_layers - 1), x in wall_x
                wall_coordinates[1, index] = x
                wall_coordinates[2, index] = -layer * wall_spacing
                index += 1
            end

            wall_ic = InitialCondition(; coordinates=wall_coordinates,
                                       mass=wall_mass,
                                       density=wall_density,
                                       particle_spacing=wall_spacing)
            state_equation = StateEquationCole(; sound_speed=20.0,
                                               reference_density=1000.0,
                                               exponent=1.0)
            wall_model = BoundaryModelDummyParticles(wall_density, wall_mass,
                                                     BernoulliPressureExtrapolation(),
                                                     WendlandC2Kernel{2}(),
                                                     1.5 * wall_spacing;
                                                     state_equation=state_equation)

            return WallBoundarySystem(wall_ic, wall_model), maximum(wall_coordinates[2, :])
        end

        function run_resolution_case(wall_spacing, n_layers)
            structure_spacing = 0.1
            sphere = SphereShape(structure_spacing, 0.2, (0.0, 0.6), 3000.0,
                                 sphere_type=VoxelSphere())

            boundary_contact_model = RigidBoundaryContactModel(; normal_stiffness=4.0e4,
                                                               normal_damping=150.0,
                                                               static_friction_coefficient=0.0,
                                                               kinetic_friction_coefficient=0.0,
                                                               tangential_stiffness=0.0,
                                                               tangential_damping=0.0,
                                                               contact_distance=2.0 *
                                                                                structure_spacing,
                                                               penetration_slop=0.0,
                                                               stick_velocity_tolerance=1e-6)

            rigid_system = RigidSPHSystem(sphere;
                                          acceleration=(0.0, -9.81),
                                          boundary_contact_model=boundary_contact_model,
                                          particle_spacing=structure_spacing)
            wall_system, wall_top = make_wall_system(wall_spacing, n_layers)

            semi = Semidiscretization(rigid_system, wall_system)
            ode = semidiscretize(semi, (0.0, 0.6))
            solution = solve(ode, RDPK3SpFSAL35();
                             abstol=1e-7, reltol=5e-4,
                             save_everystep=true,
                             callback=UpdateCallback())

            center_of_mass_y = similar(solution.t)
            minimum_y = similar(solution.t)
            total_mass = sum(rigid_system.mass)

            for i in eachindex(solution.t)
                _, u_state = solution.u[i].x
                u_rigid = TrixiParticles.wrap_u(u_state, rigid_system, semi)
                coords = TrixiParticles.current_coordinates(u_rigid, rigid_system)

                center_y = zero(eltype(coords))
                for particle in eachindex(rigid_system.mass)
                    center_y += rigid_system.mass[particle] * coords[2, particle]
                end
                center_of_mass_y[i] = center_y / total_mass
                minimum_y[i] = minimum(coords[2, :])
            end

            contact_distance = boundary_contact_model.contact_distance
            impact_index = findfirst(y -> y <= wall_top + contact_distance, minimum_y)
            rebound_height = isnothing(impact_index) ? NaN :
                             maximum(center_of_mass_y[impact_index:end])
            max_penetration = maximum(max(zero(eltype(minimum_y)),
                                          wall_top + contact_distance - y)
                                      for y in minimum_y)
            min_dt = minimum(diff(solution.t))

            return (; solution, rebound_height, max_penetration, min_dt)
        end

        baseline = run_resolution_case(0.1, 1)
        finer_wall = run_resolution_case(0.08, 1)
        layered_wall = run_resolution_case(0.1, 3)

        for result in (baseline, finer_wall, layered_wall)
            @test result.solution.retcode == ReturnCode.Success
            @test isfinite(result.rebound_height)
            @test result.max_penetration > 0
            @test result.min_dt > 1.0e-10
        end

        rebound_tolerance_spacing = 0.03
        penetration_tolerance_spacing = 0.12
        rebound_tolerance_layers = 0.02
        penetration_tolerance_layers = 0.03

        rebound_variation_finer = abs(finer_wall.rebound_height - baseline.rebound_height) /
                                  baseline.rebound_height
        rebound_variation_layered = abs(layered_wall.rebound_height - baseline.rebound_height) /
                                    baseline.rebound_height
        penetration_variation_finer = abs(finer_wall.max_penetration -
                                          baseline.max_penetration) /
                                      baseline.max_penetration
        penetration_variation_layered = abs(layered_wall.max_penetration -
                                            baseline.max_penetration) /
                                        baseline.max_penetration

        @test rebound_variation_finer < rebound_tolerance_spacing
        @test rebound_variation_layered < rebound_tolerance_layers
        @test penetration_variation_finer < penetration_tolerance_spacing
        @test penetration_variation_layered < penetration_tolerance_layers
    end

    @trixi_testset "Configuration" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]

        rigid_ic = InitialCondition(; coordinates, mass, density)
        rigid_system = RigidSPHSystem(rigid_ic)

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 0.12
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=7.0)
        fluid_system = WeaklyCompressibleSPHSystem(rigid_ic, SummationDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length)

        @test_throws ArgumentError Semidiscretization(fluid_system, rigid_system)
    end
end
