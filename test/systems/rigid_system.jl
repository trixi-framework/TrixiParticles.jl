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
