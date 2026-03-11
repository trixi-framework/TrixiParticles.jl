@testset verbose=true "RigidBodySystem" begin
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

        system = RigidBodySystem(initial_condition;
                                 boundary_model=boundary_model,
                                 acceleration=(0.0, -9.81),
                                 particle_spacing=0.1)

        @test system isa RigidBodySystem
        @test ndims(system) == 2
        @test system.initial_condition == initial_condition
        center_of_mass = [9.5 / 4.5, 9.5 / 4.5]
        @test system.relative_coordinates == coordinates .- center_of_mass
        @test system.mass == mass
        @test system.material_density == material_densities
        @test system.initial_velocity == initial_condition.velocity
        @test system.acceleration == [0.0, -9.81]
        @test iszero(system.angular_velocity[])
        @test system.particle_spacing == 0.1
        @test system.boundary_model == boundary_model
        @test system.adhesion_coefficient == 0.0
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

        system = RigidBodySystem(initial_condition;
                                 boundary_model=boundary_model,
                                 acceleration=(0.0, -9.81))

        show_compact = "RigidBodySystem{2}([0.0, -9.81], BoundaryModelDummyParticles(SummationDensity, Nothing)) with 2 particles"
        @test repr(system) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ RigidBodySystem{2}                                                                               │
        │ ══════════════════                                                                               │
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

        system = RigidBodySystem(initial_condition; boundary_model=boundary_model)
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
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        material_densities = [990.0, 1000.0]
        initial_condition = InitialCondition(; coordinates, mass,
                                             density=material_densities)

        source_terms = (coords, velocity, density, pressure,
                        t) -> SVector(density, pressure)
        system = RigidBodySystem(initial_condition; source_terms=source_terms)
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
        coordinates_2d = [0.0 1.0
                          0.0 0.0]
        mass_2d = [1.0, 1.0]
        density_2d = [1000.0, 1000.0]
        ic_2d = apply_angular_velocity(InitialCondition(; coordinates=coordinates_2d,
                                                        mass=mass_2d,
                                                        density=density_2d),
                                       2.0)

        system_2d = RigidBodySystem(ic_2d; particle_spacing=0.1)
        v0_2d = zeros(2, 2)
        TrixiParticles.write_v0!(v0_2d, system_2d)

        @test system_2d.angular_velocity[] == 2.0
        @test v0_2d == [0.0 0.0
                        -1.0 1.0]
        dt_2d = TrixiParticles.calculate_dt(v0_2d, zeros(size(v0_2d)), 0.25, system_2d,
                                            nothing)
        @test isapprox(dt_2d, 0.25 * 0.1 / 1.0)
        dt_2d_larger_cfl = TrixiParticles.calculate_dt(v0_2d, zeros(size(v0_2d)), 0.5,
                                                       system_2d, nothing)
        @test isapprox(dt_2d_larger_cfl, 0.5 * 0.1 / 1.0)

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
        v0_3d = zeros(3, 2)
        TrixiParticles.write_v0!(v0_3d, system_3d)

        @test system_3d.angular_velocity[] == [0.0, 0.0, 2.0]
        @test v0_3d == [0.0 0.0
                        -1.0 1.0
                        0.0 0.0]
    end

    @trixi_testset "Time Step Estimate 3D Gyroscopic" begin
        coordinates = [-0.5 0.5
                       0.0 0.0
                       0.0 0.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]
        initial_condition = InitialCondition(; coordinates, mass, density,
                                             particle_spacing=0.1)
        system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0, 0.0))

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
        system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0))

        # No angular velocity was applied explicitly, so this must be reconstructed
        # from the initial velocity field.
        @test system.angular_velocity[] == 1.0

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

        system_ref = RigidBodySystem(initial_condition; acceleration=(0.0, -9.81))
        system_shifted = RigidBodySystem(initial_condition; acceleration=(0.0, -1000.0))

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
        rigid_system = RigidBodySystem(initial_condition;
                                       acceleration=(0.0, 0.0))

        v = copy(velocity)
        u = copy(coordinates)
        TrixiParticles.update_final!(rigid_system, v, u, nothing, nothing, nothing, 0.0)

        @test rigid_system.angular_velocity[] == 1.0
        @test rigid_system.inertia[] == 2.0

        dv = zeros(size(v))
        TrixiParticles.interact!(dv, v, u, v, u, rigid_system, rigid_system,
                                 DummySemidiscretization())

        @test dv == [1.0 -1.0
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
        @test data.relative_coordinates == rigid_system.relative_coordinates
        @test !(:local_coordinates in fields)
    end

    @trixi_testset "Restart" begin
        coordinates = [0.0 1.0 2.0
                       0.0 0.0 0.0]
        velocity = [0.0 0.0 0.0
                    0.0 0.0 0.0]
        mass = [1.0, 1.0, 1.0]
        density = [1000.0, 1000.0, 1000.0]

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)
        rigid_system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0))

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

        @test rigid_system.center_of_mass[] == expected_center_of_mass
        @test rigid_system.relative_coordinates == expected_relative_coordinates
        @test rigid_system.center_of_mass_velocity[] == [2.0, 5.0]
        @test rigid_system.angular_velocity[] == 0.5
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

        rigid_system = RigidBodySystem(initial_condition;
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
        rigid_system = RigidBodySystem(rigid_ic)

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 0.12
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=7.0)
        fluid_system = WeaklyCompressibleSPHSystem(rigid_ic, SummationDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length)

        @test_throws ArgumentError Semidiscretization(fluid_system, rigid_system)

        rigid_boundary_model = BoundaryModelDummyParticles(density, mass,
                                                           SummationDensity(),
                                                           smoothing_kernel,
                                                           smoothing_length)
        rigid_system_with_dummy = RigidBodySystem(rigid_ic;
                                                  boundary_model=rigid_boundary_model)
        fluid_with_surface_tension = WeaklyCompressibleSPHSystem(rigid_ic,
                                                                 SummationDensity(),
                                                                 state_equation,
                                                                 smoothing_kernel,
                                                                 smoothing_length;
                                                                 surface_tension=SurfaceTensionMorris(surface_tension_coefficient=0.072),
                                                                 reference_particle_spacing=0.1)

        @test_throws ArgumentError Semidiscretization(fluid_with_surface_tension,
                                                      rigid_system_with_dummy)
    end

    @trixi_testset "Akinci Adhesion Matches Wall Boundary" begin
        particle_spacing = 1.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0
        fluid_density = 1000.0
        rigid_density = 2000.0
        particle_volume = particle_spacing^2
        adhesion_coefficient = 0.25

        state_equation = StateEquationCole(sound_speed=10.0,
                                           reference_density=fluid_density,
                                           exponent=1.0)

        function run_setup(boundary_kind)
            fluid_ic = InitialCondition(coordinates=reshape([0.0, 0.0], 2, 1),
                                        velocity=zeros(2, 1),
                                        mass=[particle_volume * fluid_density],
                                        density=[fluid_density],
                                        particle_spacing=particle_spacing)

            fluid_system = WeaklyCompressibleSPHSystem(fluid_ic, SummationDensity(),
                                                       state_equation,
                                                       smoothing_kernel,
                                                       smoothing_length;
                                                       surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.05),
                                                       reference_particle_spacing=particle_spacing)

            boundary_coordinates = reshape([1.5, 0.0], 2, 1)
            boundary_model = BoundaryModelDummyParticles([fluid_density],
                                                         [particle_volume * fluid_density],
                                                         AdamiPressureExtrapolation(),
                                                         smoothing_kernel,
                                                         smoothing_length,
                                                         state_equation=state_equation,
                                                         reference_particle_spacing=particle_spacing)

            boundary_system = if boundary_kind == :wall
                wall_ic = InitialCondition(coordinates=boundary_coordinates,
                                           velocity=zeros(2, 1),
                                           mass=[particle_volume * fluid_density],
                                           density=[fluid_density],
                                           particle_spacing=particle_spacing)
                WallBoundarySystem(wall_ic, boundary_model,
                                   adhesion_coefficient=adhesion_coefficient)
            else
                rigid_ic = InitialCondition(coordinates=boundary_coordinates,
                                            velocity=zeros(2, 1),
                                            mass=[particle_volume * rigid_density],
                                            density=[rigid_density],
                                            particle_spacing=particle_spacing)
                RigidBodySystem(rigid_ic;
                                boundary_model=boundary_model,
                                adhesion_coefficient=adhesion_coefficient)
            end

            semi = Semidiscretization(fluid_system, boundary_system)
            ode = semidiscretize(semi, (0.0, 0.01))

            v_ode, u_ode = ode.u0.x
            dv_ode = zero(v_ode)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

            fluid = ode.p.systems[1]
            boundary = ode.p.systems[2]
            dv_fluid = TrixiParticles.wrap_v(dv_ode, fluid, ode.p)

            return fluid, boundary, copy(dv_fluid[:, 1])
        end

        _, _, dv_wall = run_setup(:wall)
        fluid_rigid, rigid_system, dv_rigid = run_setup(:rigid)

        @test isapprox(dv_rigid, dv_wall; rtol=sqrt(eps()), atol=sqrt(eps()))
        @test isapprox(rigid_system.resultant_force[],
                       -fluid_rigid.mass[1] * dv_rigid;
                       rtol=sqrt(eps()), atol=sqrt(eps()))
    end

    @trixi_testset "Open Boundary Interaction Matches Fluid Interaction" begin
        particle_spacing = 1.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0
        fluid_density = 1000.0
        rigid_density = 2000.0
        particle_volume = particle_spacing^2
        boundary_pressure = 2.5

        state_equation = StateEquationCole(sound_speed=10.0,
                                           reference_density=fluid_density,
                                           exponent=1.0)

        boundary_model = BoundaryModelDummyParticles([fluid_density],
                                                     [particle_volume * fluid_density],
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     state_equation=state_equation,
                                                     reference_particle_spacing=particle_spacing)

        function run_setup(neighbor_kind)
            rigid_ic = InitialCondition(coordinates=reshape([0.0, 0.0], 2, 1),
                                        velocity=zeros(2, 1),
                                        mass=[particle_volume * rigid_density],
                                        density=[rigid_density],
                                        particle_spacing=particle_spacing)
            rigid_system = RigidBodySystem(rigid_ic;
                                           boundary_model=boundary_model,
                                           acceleration=(0.0, 0.0))

            neighbor_ic = InitialCondition(coordinates=reshape([1.5, 0.0], 2, 1),
                                           velocity=zeros(2, 1),
                                           mass=[particle_volume * fluid_density],
                                           density=[fluid_density],
                                           particle_spacing=particle_spacing)

            fluid_support_ic = InitialCondition(coordinates=reshape([10.0, 10.0], 2, 1),
                                                velocity=zeros(2, 1),
                                                mass=[particle_volume * fluid_density],
                                                density=[fluid_density],
                                                particle_spacing=particle_spacing)
            fluid_system = WeaklyCompressibleSPHSystem(fluid_support_ic, SummationDensity(),
                                                       state_equation,
                                                       smoothing_kernel,
                                                       smoothing_length)

            neighbor_system = if neighbor_kind == :fluid
                WeaklyCompressibleSPHSystem(neighbor_ic, SummationDensity(),
                                            state_equation,
                                            smoothing_kernel,
                                            smoothing_length)
            else
                boundary_face = ([2.0, -0.5], [2.0, 0.5])
                zone = BoundaryZone(; boundary_face, face_normal=(1.0, 0.0),
                                    density=fluid_density,
                                    particle_spacing=particle_spacing,
                                    initial_condition=neighbor_ic,
                                    open_boundary_layers=1,
                                    boundary_type=InFlow())

                OpenBoundarySystem(zone; fluid_system,
                                   boundary_model=BoundaryModelMirroringTafuni(),
                                   buffer_size=0)
            end

            semi = Semidiscretization(fluid_system, rigid_system, neighbor_system)
            ode = semidiscretize(semi, (0.0, 0.01))

            rigid = ode.p.systems[2]
            neighbor = ode.p.systems[3]

            v_ode, u_ode = ode.u0.x
            TrixiParticles.current_density(v_ode, neighbor) .= fluid_density
            TrixiParticles.current_pressure(v_ode, neighbor) .= boundary_pressure

            dv_ode = zero(v_ode)
            TrixiParticles.interact!(dv_ode, v_ode, u_ode, rigid, neighbor, ode.p)

            dv_rigid = TrixiParticles.wrap_v(dv_ode, rigid, ode.p)

            return rigid, copy(dv_rigid[:, 1])
        end

        rigid_fluid, dv_fluid = run_setup(:fluid)
        rigid_open_boundary, dv_open_boundary = run_setup(:open_boundary)

        @test isapprox(dv_open_boundary, dv_fluid; rtol=sqrt(eps()), atol=sqrt(eps()))
        @test isapprox(rigid_open_boundary.resultant_force[],
                       rigid_fluid.resultant_force[];
                       rtol=sqrt(eps()), atol=sqrt(eps()))
    end
end
