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

        @test data.center_of_mass ≈ [0.0, 0.0]
        @test data.center_of_mass_velocity ≈ [0.0, 0.0]
        @test data.angular_velocity ≈ 1.0
        @test data.resultant_force ≈ [0.0, 0.0]
        @test data.resultant_torque ≈ 0.0
        @test data.angular_acceleration_force ≈ 0.0
        @test data.gyroscopic_acceleration ≈ 0.0
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
