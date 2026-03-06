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
        @test isapprox(system.local_coordinates, coordinates .- center_of_mass)
        @test system.mass == mass
        @test system.material_density == material_densities
        @test system.initial_velocity == initial_condition.velocity
        @test system.acceleration == [0.0, -9.81]
        @test iszero(system.initial_condition.angular_velocity)
        @test system.particle_spacing == 0.1
        @test system.boundary_model == boundary_model
        @test TrixiParticles.v_nvariables(system) == 2

        dt = TrixiParticles.calculate_dt(zeros(2, 3), zeros(2, 3), 0.25, system,
                                         nothing)
        @test isinf(dt)
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

        show_compact = "RigidSPHSystem{2}([0.0, -9.81], BoundaryModelDummyParticles(SummationDensity, Nothing)) with 2 particles"
        @test repr(system) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ RigidSPHSystem{2}                                                                                │
        │ ═════════════════                                                                                │
        │ #particles: ………………………………………………… 2                                                                │
        │ acceleration: …………………………………………… [0.0, -9.81]                                                     │
        │ initial angular velocity: …………… 0.0                                                              │
        │ boundary model: ……………………………………… BoundaryModelDummyParticles(SummationDensity, Nothing)           │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end

    @trixi_testset "Hydrodynamic Density" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        material_densities = [990.0, 1000.0]
        hydrodynamic_densities = [1001.0, 1002.0]

        initial_condition = InitialCondition(; coordinates, mass,
                                             density=material_densities)

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 0.12
        boundary_model = BoundaryModelDummyParticles(hydrodynamic_densities, mass,
                                                     SummationDensity(),
                                                     smoothing_kernel,
                                                     smoothing_length)

        system = RigidSPHSystem(initial_condition; boundary_model=boundary_model)
        v = zeros(TrixiParticles.v_nvariables(system),
                  TrixiParticles.n_integrated_particles(system))

        @test TrixiParticles.current_density(v, system) == hydrodynamic_densities
        @test TrixiParticles.hydrodynamic_density(v, system) == hydrodynamic_densities
        @test system.material_density == material_densities

        system_no_model = RigidSPHSystem(initial_condition)
        v_no_model = zeros(TrixiParticles.v_nvariables(system_no_model),
                           TrixiParticles.n_integrated_particles(system_no_model))
        @test TrixiParticles.current_density(v_no_model, system_no_model) == material_densities
    end

    @trixi_testset "Initial Angular Velocity" begin
        coordinates_2d = [0.0 1.0
                          0.0 0.0]
        mass_2d = [1.0, 1.0]
        density_2d = [1000.0, 1000.0]
        ic_2d = InitialCondition(; coordinates=coordinates_2d, mass=mass_2d,
                                 density=density_2d, angular_velocity=2.0)

        system_2d = RigidSPHSystem(ic_2d; particle_spacing=0.1)
        v0_2d = zeros(2, 2)
        TrixiParticles.write_v0!(v0_2d, system_2d)

        @test system_2d.initial_condition.angular_velocity == 2.0
        @test v0_2d == [0.0 0.0
                        -1.0 1.0]
        dt_2d = TrixiParticles.calculate_dt(v0_2d, zeros(size(v0_2d)), 0.25, system_2d,
                                            nothing)
        @test isapprox(dt_2d, 0.25 * 0.1 / 1.0)
        dt_2d_larger_cfl = TrixiParticles.calculate_dt(v0_2d, zeros(size(v0_2d)), 0.5,
                                                       system_2d, nothing)
        @test isapprox(dt_2d_larger_cfl, 0.5 * 0.1 / 1.0)
        @test_throws ArgumentError InitialCondition(; coordinates=coordinates_2d,
                                                    mass=mass_2d,
                                                    density=density_2d,
                                                    angular_velocity=(0.0, 1.0))

        coordinates_3d = [0.0 1.0
                          0.0 0.0
                          0.0 0.0]
        mass_3d = [1.0, 1.0]
        density_3d = [1000.0, 1000.0]
        ic_3d = InitialCondition(; coordinates=coordinates_3d, mass=mass_3d,
                                 density=density_3d,
                                 angular_velocity=(0.0, 0.0, 2.0))

        system_3d = RigidSPHSystem(ic_3d)
        v0_3d = zeros(3, 2)
        TrixiParticles.write_v0!(v0_3d, system_3d)

        @test system_3d.initial_condition.angular_velocity == [0.0, 0.0, 2.0]
        @test v0_3d == [0.0 0.0
                        -1.0 1.0
                        0.0 0.0]

        ic_3d_default = InitialCondition(; coordinates=coordinates_3d, mass=mass_3d,
                                         density=density_3d)
        @test iszero(ic_3d_default.angular_velocity)
        @test_throws ArgumentError InitialCondition(; coordinates=coordinates_3d,
                                                    mass=mass_3d,
                                                    density=density_3d,
                                                    angular_velocity=2.0)
    end

    @trixi_testset "Time Step Estimate 3D Gyroscopic" begin
        coordinates = [-0.5 0.5
                       0.0 0.0
                       0.0 0.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]
        initial_condition = InitialCondition(; coordinates, mass, density, particle_spacing=0.1)
        system = RigidSPHSystem(initial_condition; acceleration=(0.0, 0.0, 0.0))

        system.angular_velocity[] = SVector(0.0, 0.0, 0.0)
        system.angular_acceleration_force[] = SVector(0.0, 0.0, 0.0)
        system.gyroscopic_acceleration[] = SVector(0.0, 0.0, 2.0)
        system.center_of_mass_velocity[] = SVector(0.0, 0.0, 0.0)

        dt = TrixiParticles.calculate_dt(zeros(3, 2), zeros(3, 2), 0.25, system, nothing)
        @test isapprox(dt, 0.25 * sqrt(0.1 / 1.0))
    end

    @trixi_testset "Time Step Estimate from Initial Velocity" begin
        coordinates = [-1.0 1.0
                       0.0 0.0]
        velocity = [0.0 0.0
                    -1.0 1.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]
        initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                             particle_spacing=0.1)
        system = RigidSPHSystem(initial_condition; acceleration=(0.0, 0.0))

        # No angular_velocity kwarg was set, so this must be reconstructed from the
        # initial velocity field.
        @test iszero(initial_condition.angular_velocity)
        @test isapprox(system.angular_velocity[], 1.0)

        dt = TrixiParticles.calculate_dt(zeros(2, 2), zeros(2, 2), 0.25, system, nothing)
        @test isapprox(dt, 0.25 * 0.1 / 1.0)
    end

    @trixi_testset "Time Step Invariance under Uniform Acceleration" begin
        coordinates = [-1.0 1.0
                       0.0 0.0]
        velocity = [1.0 1.0
                    0.0 0.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]
        initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                             particle_spacing=0.1)

        system_ref = RigidSPHSystem(initial_condition; acceleration=(0.0, -9.81))
        system_shifted = RigidSPHSystem(initial_condition; acceleration=(0.0, -1000.0))

        dt_ref = TrixiParticles.calculate_dt(zeros(2, 2), zeros(2, 2), 0.25, system_ref,
                                             nothing)
        dt_shifted = TrixiParticles.calculate_dt(zeros(2, 2), zeros(2, 2), 0.25,
                                                 system_shifted, nothing)

        @test isapprox(dt_ref, 0.25 * 0.1 / 1.0)
        @test dt_shifted == dt_ref
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

        @test isapprox(rigid_system.angular_velocity[], 1.0)
        @test isapprox(rigid_system.inertia[], 2.0)

        dv = zeros(size(v))
        for particle in TrixiParticles.each_integrated_particle(rigid_system)
            TrixiParticles.add_acceleration!(dv, particle, rigid_system)
        end

        @test isapprox(dv, [1.0 -1.0
                            0.0 0.0])
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

        @test isapprox(data.center_of_mass, [0.0, 0.0])
        @test isapprox(data.center_of_mass_velocity, [0.0, 0.0])
        @test isapprox(data.angular_velocity, 1.0)
        @test isapprox(data.resultant_force, [0.0, 0.0])
        @test data.resultant_torque == 0.0
        @test data.angular_acceleration_force == 0.0
        @test data.gyroscopic_acceleration == 0.0
        @test data.local_coordinates == rigid_system.local_coordinates
        @test data.relative_coordinates == rigid_system.relative_coordinates
    end

    @trixi_testset "Restart" begin
        coordinates = [0.0 1.0 2.0
                       0.0 0.0 0.0]
        velocity = [0.0 0.0 0.0
                    0.0 0.0 0.0]
        mass = [1.0, 1.0, 1.0]
        density = [1000.0, 1000.0, 1000.0]

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)
        rigid_system = RigidSPHSystem(initial_condition; acceleration=(0.0, 0.0))

        u_new = [2.0 4.0 6.0
                 3.0 3.0 3.0]
        v_new = [1.0 2.0 3.0
                 4.0 5.0 6.0]

        restarted_system = TrixiParticles.restart_with!(rigid_system, v_new, u_new)

        @test restarted_system === rigid_system
        @test rigid_system.initial_condition.coordinates == u_new
        @test rigid_system.initial_condition.velocity == v_new
        @test rigid_system.initial_velocity == v_new

        expected_center_of_mass = [4.0, 3.0]
        expected_relative_coordinates = u_new .- expected_center_of_mass

        @test isapprox(rigid_system.center_of_mass[], expected_center_of_mass)
        @test isapprox(rigid_system.local_coordinates, expected_relative_coordinates)
        @test isapprox(rigid_system.relative_coordinates, expected_relative_coordinates)
        @test isapprox(rigid_system.center_of_mass_velocity[], [2.0, 5.0])
        @test isapprox(rigid_system.angular_velocity[], 0.5)
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
