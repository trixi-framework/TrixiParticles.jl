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

        # todo: fix this
        # dt = TrixiParticles.calculate_dt(system.initial_velocity,
        #                                  system.initial_condition.coordinates,
        #                                  0.25, system, nothing)
        # @test isinf(dt)
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
        u0_2d = zeros(2, 2)
        v0_2d = zeros(2, 2)
        TrixiParticles.write_u0!(u0_2d, system_2d)
        TrixiParticles.write_v0!(v0_2d, system_2d)

        @test iszero(system_2d.angular_velocity[])
        @test v0_2d == [0.0 0.0
                        -1.0 1.0]
        # todo: fix this
        # dt_2d = TrixiParticles.calculate_dt(v0_2d, u0_2d, 0.25, system_2d,
        #                                     nothing)
        # @test isapprox(dt_2d, 0.25 * 0.1 / 1.0)
        # dt_2d_larger_cfl = TrixiParticles.calculate_dt(v0_2d, u0_2d, 0.5,
        #                                                system_2d, nothing)
        # @test isapprox(dt_2d_larger_cfl, 0.5 * 0.1 / 1.0)

        semi_2d = Semidiscretization(system_2d, neighborhood_search=nothing)
        ode_2d = semidiscretize(semi_2d, (0.0, 0.0); reset_threads=false)
        dt_2d_semi = TrixiParticles.calculate_dt(ode_2d.u0.x[1], ode_2d.u0.x[2], 0.25,
                                                 ode_2d.p)

        # todo: fix this
        # @test isapprox(dt_2d_semi, dt_2d)

        TrixiParticles.update_final!(system_2d, v0_2d, u0_2d, nothing, nothing, nothing,
                                     0.0)
        @test system_2d.angular_velocity[] == 2.0

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

    # todo: fix this
    # @trixi_testset "Time Step Estimate 3D Gyroscopic" begin
    #     coordinates = [1.0 -1.0 0.0 0.0 0.0 0.0
    #                    0.0 0.0 2.0 -2.0 0.0 0.0
    #                    0.0 0.0 0.0 0.0 3.0 -3.0]
    #     mass = fill(1.0, 6)
    #     density = fill(1000.0, 6)
    #     initial_condition = apply_angular_velocity(InitialCondition(; coordinates, mass,
    #                                                                 density,
    #                                                                 particle_spacing=10.0),
    #                                                (1.0, 2.0, 3.0))
    #     system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0, 0.0))

    #     angular_velocity = SVector(1.0, 2.0, 3.0)
    #     gyroscopic_acceleration = SVector(-30 / 13, 12 / 5, -6 / 5)
    #     acceleration_scale = 3.0 * (norm(angular_velocity)^2 +
    #                           norm(gyroscopic_acceleration))
    #     dt_acceleration = 0.25 * sqrt(10.0 / acceleration_scale)
    #     dt_velocity = 0.25 * 10.0 / (3.0 * norm(angular_velocity))

    #     dt = TrixiParticles.calculate_dt(system.initial_velocity,
    #                                      system.initial_condition.coordinates,
    #                                      0.25, system, nothing)
    #     @test isapprox(dt, min(dt_acceleration, dt_velocity))
    # end

    # todo: fix this
    # @trixi_testset "Time Step Estimate from Initial Velocity" begin
    #     coordinates = [-1.0 1.0
    #                    0.0 0.0]
    #     velocity = [0.0 0.0
    #                 -1.0 1.0]
    #     mass = [1.0, 1.0]
    #     density = [1000.0, 1000.0]
    #     initial_condition = InitialCondition(; coordinates, velocity, mass, density,
    #                                          particle_spacing=0.1)
    #     system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0))

    #     @test iszero(system.angular_velocity[])

    #     dt = TrixiParticles.calculate_dt(system.initial_velocity,
    #                                      system.initial_condition.coordinates,
    #                                      0.25, system, nothing)
    #     @test isapprox(dt, 0.25 * 0.1 / 1.0)

    #     TrixiParticles.update_final!(system, system.initial_velocity,
    #                                  system.initial_condition.coordinates,
    #                                  nothing, nothing, nothing, 0.0)
    #     @test system.angular_velocity[] == 1.0
    # end

    # todo: fix this
    # @trixi_testset "Time Step Invariance under Uniform Acceleration" begin
    #     coordinates = [-1.0 1.0
    #                    0.0 0.0]
    #     velocity = [1.0 1.0
    #                 0.0 0.0]
    #     mass = [1.0, 1.0]
    #     density = [1000.0, 1000.0]
    #     initial_condition = InitialCondition(; coordinates, velocity, mass, density,
    #                                          particle_spacing=0.1)

    #     system_ref = RigidBodySystem(initial_condition; acceleration=(0.0, -9.81))
    #     system_shifted = RigidBodySystem(initial_condition; acceleration=(0.0, -1000.0))

    #     dt_ref = TrixiParticles.calculate_dt(system_ref.initial_velocity,
    #                                          system_ref.initial_condition.coordinates,
    #                                          0.25, system_ref, nothing)
    #     dt_shifted = TrixiParticles.calculate_dt(system_shifted.initial_velocity,
    #                                              system_shifted.initial_condition.coordinates,
    #                                              0.25, system_shifted, nothing)

    #     @test isapprox(dt_ref, 0.25 * 0.1 / 1.0)
    #     @test dt_shifted == dt_ref
    # end

    @trixi_testset "Rotational Kinematics" begin
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

    @trixi_testset "IO Data" begin
        coordinates = [-1.0 1.0
                       0.0 0.0]
        velocity = [0.0 0.0
                    -1.0 1.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]

        initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                             particle_spacing=1.0)
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
        @test data.contact_count == 0
        @test data.max_contact_penetration == 0.0
        @test data.relative_coordinates == rigid_system.relative_coordinates
        @test :contact_count in fields
        @test :max_contact_penetration in fields
        @test !(:local_coordinates in fields)
    end

    @trixi_testset "Restart" begin
        coordinates = [0.0 1.0 2.0
                       0.0 0.0 0.0]
        velocity = [0.0 0.0 0.0
                    0.0 0.0 0.0]
        mass = [1.0, 1.0, 1.0]
        density = [1000.0, 1000.0, 1000.0]

        initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                             particle_spacing=1.0)
        rigid_system = RigidBodySystem(initial_condition; acceleration=(0.0, 0.0))

        u_new = [2.0 4.0 6.0
                 3.0 3.0 3.0]
        v_new = [1.0 2.0 3.0
                 4.0 5.0 6.0]

        TrixiParticles.update_final!(rigid_system, rigid_system.initial_velocity,
                                     rigid_system.initial_condition.coordinates,
                                     nothing, nothing, nothing, 0.0)
        stale_relative_coordinates = copy(rigid_system.relative_coordinates)
        stale_center_of_mass = rigid_system.center_of_mass[]
        stale_center_of_mass_velocity = rigid_system.center_of_mass_velocity[]
        stale_angular_velocity = rigid_system.angular_velocity[]
        stale_force = SVector(7.0, -11.0)
        stale_torque = 5.0
        stale_angular_acceleration_force = 2.0
        rigid_system.resultant_force[] = stale_force
        rigid_system.resultant_torque[] = stale_torque
        rigid_system.angular_acceleration_force[] = stale_angular_acceleration_force

        restarted_system = TrixiParticles.restart_with!(rigid_system, v_new, u_new)

        @test restarted_system === rigid_system
        @test rigid_system.initial_condition.coordinates == u_new
        @test rigid_system.initial_condition.velocity == v_new
        @test rigid_system.initial_velocity == v_new
        @test rigid_system.relative_coordinates == stale_relative_coordinates
        @test rigid_system.center_of_mass[] == stale_center_of_mass
        @test rigid_system.center_of_mass_velocity[] == stale_center_of_mass_velocity
        @test rigid_system.angular_velocity[] == stale_angular_velocity
        @test rigid_system.resultant_force[] == stale_force
        @test rigid_system.resultant_torque[] == stale_torque
        @test rigid_system.angular_acceleration_force[] == stale_angular_acceleration_force

        expected_center_of_mass = [4.0, 3.0]
        expected_relative_coordinates = u_new .- expected_center_of_mass
        # todo: fix this
        # dt_restarted = TrixiParticles.calculate_dt(v_new, u_new, 0.25, rigid_system,
        #                                            nothing)

        TrixiParticles.update_final!(rigid_system, v_new, u_new, nothing, nothing, nothing,
                                     0.0)
        # todo: fix this
        # dt_updated = TrixiParticles.calculate_dt(v_new, u_new, 0.25, rigid_system, nothing)

        @test rigid_system.center_of_mass[] == expected_center_of_mass
        @test rigid_system.relative_coordinates == expected_relative_coordinates
        @test rigid_system.center_of_mass_velocity[] == [2.0, 5.0]
        @test rigid_system.angular_velocity[] == 0.5
        #todo: fix this
        # @test isapprox(dt_restarted, dt_updated)
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

    @trixi_testset "Rigid Interaction Caches Stay Zero without Fluid Neighbors" begin
        rigid_ic = InitialCondition(coordinates=reshape([0.0, 0.0], 2, 1),
                                    velocity=zeros(2, 1),
                                    mass=[1.0],
                                    density=[1.0],
                                    particle_spacing=1.0)
        rigid_system = RigidBodySystem(rigid_ic; acceleration=(0.0, 0.0))

        semi = Semidiscretization(rigid_system)
        ode = semidiscretize(semi, (0.0, 0.01))

        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

        rigid = only(ode.p.systems)
        dv_rigid = TrixiParticles.wrap_v(dv_ode, rigid, ode.p)

        @test all(iszero, dv_rigid)
        @test iszero(rigid.resultant_force[])
        @test iszero(rigid.resultant_torque[])
        @test iszero(rigid.angular_acceleration_force[])
    end

    @trixi_testset "Rigid Resultants Accumulate over Multiple Fluid Systems" begin
        particle_spacing = 1.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0
        fluid_density = 1000.0
        rigid_density = 2000.0
        particle_volume = particle_spacing^2

        state_equation = StateEquationCole(sound_speed=10.0,
                                           reference_density=fluid_density,
                                           exponent=1.0)

        boundary_model = BoundaryModelDummyParticles(fill(fluid_density, 2),
                                                     fill(particle_volume * fluid_density,
                                                          2),
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     state_equation=state_equation,
                                                     reference_particle_spacing=particle_spacing)

        function run_setup(fluid_positions)
            rigid_ic = InitialCondition(coordinates=[-0.5 0.5
                                                     0.0 0.0],
                                        velocity=zeros(2, 2),
                                        mass=fill(particle_volume * rigid_density, 2),
                                        density=fill(rigid_density, 2),
                                        particle_spacing=particle_spacing)
            rigid_system = RigidBodySystem(rigid_ic;
                                           boundary_model=boundary_model,
                                           acceleration=(0.0, 0.0))

            fluid_systems = map(fluid_positions) do position
                fluid_ic = InitialCondition(coordinates=reshape(collect(position), 2, 1),
                                            velocity=zeros(2, 1),
                                            mass=[particle_volume * fluid_density],
                                            density=[fluid_density],
                                            particle_spacing=particle_spacing)

                WeaklyCompressibleSPHSystem(fluid_ic, SummationDensity(), state_equation,
                                            smoothing_kernel, smoothing_length)
            end

            semi = Semidiscretization(fluid_systems..., rigid_system)
            ode = semidiscretize(semi, (0.0, 0.01))

            v_ode, u_ode = ode.u0.x
            dv_ode = zero(v_ode)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

            rigid = last(ode.p.systems)
            dv_rigid = TrixiParticles.wrap_v(dv_ode, rigid, ode.p)

            return rigid, copy(dv_rigid)
        end

        fluid_positions = ((1.5, 0.0), (-1.5, 1.0))

        rigid_1, dv_1 = run_setup((fluid_positions[1],))
        rigid_2, dv_2 = run_setup((fluid_positions[2],))
        rigid_both, dv_both = run_setup(fluid_positions)

        @test isapprox(dv_both, dv_1 .+ dv_2; rtol=sqrt(eps()), atol=sqrt(eps()))
        @test isapprox(rigid_both.resultant_force[],
                       rigid_1.resultant_force[] + rigid_2.resultant_force[];
                       rtol=sqrt(eps()), atol=sqrt(eps()))
        @test isapprox(rigid_both.resultant_torque[],
                       rigid_1.resultant_torque[] + rigid_2.resultant_torque[];
                       rtol=sqrt(eps()), atol=sqrt(eps()))
        @test isapprox(rigid_both.angular_acceleration_force[],
                       rigid_1.angular_acceleration_force[] +
                       rigid_2.angular_acceleration_force[];
                       rtol=sqrt(eps()), atol=sqrt(eps()))
    end

    @trixi_testset "Rigid Bodies Ignore Open Boundary Interactions" begin
        particle_spacing = 1.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0
        fluid_density = 1000.0
        rigid_density = 2000.0
        particle_volume = particle_spacing^2

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

        rigid_ic = InitialCondition(coordinates=reshape([0.0, 0.0], 2, 1),
                                    velocity=zeros(2, 1),
                                    mass=[particle_volume * rigid_density],
                                    density=[rigid_density],
                                    particle_spacing=particle_spacing)
        rigid_system = RigidBodySystem(rigid_ic;
                                       boundary_model=boundary_model,
                                       acceleration=(0.0, 0.0))

        open_boundary_ic = InitialCondition(coordinates=reshape([1.5, 0.0], 2, 1),
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

        boundary_face = ([2.0, -0.5], [2.0, 0.5])
        zone = BoundaryZone(; boundary_face, face_normal=(1.0, 0.0),
                            density=fluid_density,
                            particle_spacing=particle_spacing,
                            initial_condition=open_boundary_ic,
                            open_boundary_layers=1,
                            boundary_type=InFlow())

        open_boundary_system = OpenBoundarySystem(zone; fluid_system,
                                                  boundary_model=BoundaryModelDynamicalPressureZhang(),
                                                  buffer_size=0)

        semi = Semidiscretization(fluid_system, rigid_system, open_boundary_system)
        ode = semidiscretize(semi, (0.0, 0.01))

        rigid = ode.p.systems[2]
        open_boundary = ode.p.systems[3]

        @test iszero(TrixiParticles.compact_support(rigid, open_boundary))
        @test iszero(TrixiParticles.compact_support(open_boundary, rigid))

        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)

        TrixiParticles.interact!(dv_ode, v_ode, u_ode, rigid, open_boundary, ode.p)
        TrixiParticles.interact!(dv_ode, v_ode, u_ode, open_boundary, rigid, ode.p)

        dv_rigid = TrixiParticles.wrap_v(dv_ode, rigid, ode.p)
        dv_open_boundary = TrixiParticles.wrap_v(dv_ode, open_boundary, ode.p)

        @test all(iszero, dv_rigid[:, 1])
        @test all(iszero, dv_open_boundary[:, 1])
        @test iszero(rigid.resultant_force[])
        @test iszero(rigid.resultant_torque[])
    end

    @trixi_testset "Rigid Contact Model" begin
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

        pair_contact_distance = max(contact_model_1.contact_distance,
                                    contact_model_2.contact_distance)
        pair_normal_stiffness = (contact_model_1.normal_stiffness +
                                 contact_model_2.normal_stiffness) / 2
        pair_normal_damping = (contact_model_1.normal_damping +
                               contact_model_2.normal_damping) / 2
        pair_penetration = pair_contact_distance - 0.08
        normal_velocity = -1.5
        pair_contact_dt = sqrt((rigid_mass_1[1] * rigid_mass_2[1] /
                                (rigid_mass_1[1] + rigid_mass_2[1])) /
                               pair_normal_stiffness)
        expected_force_magnitude = pair_normal_stiffness * pair_penetration -
                                   pair_normal_damping * normal_velocity
        expected_force = SVector(-expected_force_magnitude, 0.0)

        @test vec(force_after_forward_1[:, 1]) ≈ collect(expected_force)
        @test vec(rigid_system_2.force_per_particle[:, 1]) ≈ collect(-expected_force)
        @test rigid_system_1.cache.contact_count[] == 1
        @test rigid_system_2.cache.contact_count[] == 1
        @test rigid_system_1.cache.max_contact_penetration[] ≈ pair_penetration
        @test rigid_system_2.cache.max_contact_penetration[] ≈ pair_penetration

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
        @test TrixiParticles.contact_time_step(rigid_system_1, semi_rigid) ≈ pair_contact_dt
        @test TrixiParticles.contact_time_step(rigid_system_2, semi_rigid) ≈ pair_contact_dt
        zero_velocity_ode = zero(v_ode_rigid)
        @test TrixiParticles.calculate_dt(zero_velocity_ode, u_ode_rigid, 0.25,
                                          rigid_system_1, semi_rigid) ≈
              0.25 * pair_contact_dt
        @test TrixiParticles.calculate_dt(zero_velocity_ode, u_ode_rigid, 0.25,
                                          semi_rigid) ≈ 0.25 * pair_contact_dt

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
            @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["contact_count"]))) > 0
            @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["max_contact_penetration"]))) ≈
                  rigid_system_1.cache.max_contact_penetration[]
            @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["max_contact_penetration"]))) > 0
        end

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

        runtime_model = TrixiParticles.copy_contact_model(contact_model, 0.1, Float64)
        @test runtime_model isa RigidContactModel
        @test runtime_model.normal_stiffness ≈ 2.0e4
        @test runtime_model.normal_damping ≈ 20.0
        @test runtime_model.contact_distance ≈ 0.1

        spacing_scaled_model = RigidContactModel(; normal_stiffness=5.0)
        spacing_scaled_runtime = TrixiParticles.copy_contact_model(spacing_scaled_model,
                                                                   0.125,
                                                                   Float64)
        @test spacing_scaled_runtime.contact_distance ≈ 0.125

        @test TrixiParticles.normal_contact_force(contact_model, 0.02, -0.5) ≈ 410.0
        @test TrixiParticles.normal_contact_force(contact_model, 0.02, 30.0) ≈ 0.0

        @test_throws ArgumentError RigidContactModel(; normal_stiffness=0.0)
        @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                     normal_damping=-1.0)
        @test_throws ArgumentError RigidContactModel(; normal_stiffness=1.0,
                                                     contact_distance=-1.0)

        rigid_system = RigidBodySystem(rigid_ic;
                                       acceleration=(0.0, 0.0),
                                       contact_model=contact_model)
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
        @test rigid_system.contact_model isa RigidContactModel
        @test rigid_system.contact_model.normal_stiffness ≈ contact_model.normal_stiffness
        @test rigid_system.contact_model.normal_damping ≈ contact_model.normal_damping
        @test rigid_system.contact_model.contact_distance ≈ contact_model.contact_distance
        @test size(rigid_system_custom_manifolds.cache.contact_manifold_weight_sum, 1) == 3
        @test TrixiParticles.compact_support(rigid_system, boundary_system) ≈
              contact_model.contact_distance
        @test TrixiParticles.compact_support(rigid_system_with_boundary,
                                             boundary_system) ≈
              contact_model.contact_distance
        @test iszero(TrixiParticles.compact_support(rigid_system_without_contact,
                                                    boundary_system))
        @test_throws ArgumentError RigidBodySystem(rigid_ic;
                                                   contact_model=contact_model,
                                                   max_manifolds=0)

        system_meta_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(system_meta_data, rigid_system)
        @test system_meta_data["contact_model"]["normal_stiffness"] ≈ 2.0e4
        @test system_meta_data["contact_model"]["normal_damping"] ≈ 20.0
        @test system_meta_data["contact_model"]["contact_distance"] ≈ 0.1

        semi = Semidiscretization(rigid_system, boundary_system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)

        TrixiParticles.interact!(dv_ode, v_ode, u_ode, rigid_system, boundary_system, semi)
        dv = TrixiParticles.wrap_v(dv_ode, rigid_system, semi)
        v_rigid = TrixiParticles.wrap_v(v_ode, rigid_system, semi)
        u_rigid = TrixiParticles.wrap_u(u_ode, rigid_system, semi)
        TrixiParticles.finalize_interaction!(rigid_system, dv, v_rigid, u_rigid,
                                             dv_ode, v_ode, u_ode, semi)

        @test dv[2, 1] > 0
        @test rigid_system.cache.contact_count[] == 1
        @test rigid_system.cache.max_contact_penetration[] ≈ 0.05
    end
end
