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
        fluid_ic = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                    velocity=zeros(2, 1),
                                    mass=[particle_volume * fluid_density],
                                    density=[fluid_density], particle_spacing)

        fluid_system = WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel,
                                                   smoothing_length,
                                                   density_calculator=SummationDensity(),
                                                   state_equation,
                                                   surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.05),
                                                   reference_particle_spacing=particle_spacing)

        boundary_coordinates = reshape([1.5, 0.0], 2, 1)
        boundary_model = BoundaryModelDummyParticles([fluid_density],
                                                     [particle_volume * fluid_density],
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length;
                                                     state_equation,
                                                     reference_particle_spacing=particle_spacing)

        boundary_system = if boundary_kind == :wall
            wall_ic = InitialCondition(; coordinates=boundary_coordinates,
                                       velocity=zeros(2, 1),
                                       mass=[particle_volume * fluid_density],
                                       density=[fluid_density], particle_spacing)
            WallBoundarySystem(wall_ic, boundary_model; adhesion_coefficient)
        else
            rigid_ic = InitialCondition(; coordinates=boundary_coordinates,
                                        velocity=zeros(2, 1),
                                        mass=[particle_volume * rigid_density],
                                        density=[rigid_density], particle_spacing)
            RigidBodySystem(rigid_ic; boundary_model, adhesion_coefficient)
        end

        semi_ = Semidiscretization(fluid_system, boundary_system)
        ode = semidiscretize(semi_, (0.0, 0.01))
        semi = ode.p.semi

        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

        fluid = semi.systems[1]
        boundary = semi.systems[2]
        dv_fluid = TrixiParticles.wrap_v(dv_ode, fluid, semi)

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

    semi_ = Semidiscretization(rigid_system)
    ode = semidiscretize(semi_, (0.0, 0.01))
    semi = ode.p.semi

    v_ode, u_ode = ode.u0.x
    dv_ode = zero(v_ode)
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

    rigid = only(semi.systems)
    dv_rigid = TrixiParticles.wrap_v(dv_ode, rigid, semi)

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
                                                      2), AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length;
                                                 state_equation,
                                                 reference_particle_spacing=particle_spacing)

    function run_setup(fluid_positions)
        rigid_ic = InitialCondition(; coordinates=[-0.5 0.5
                                                   0.0 0.0],
                                    velocity=zeros(2, 2),
                                    mass=fill(particle_volume * rigid_density, 2),
                                    density=fill(rigid_density, 2), particle_spacing)
        rigid_system = RigidBodySystem(rigid_ic; boundary_model,
                                       acceleration=(0.0, 0.0))

        fluid_systems = map(fluid_positions) do position
            fluid_ic = InitialCondition(; coordinates=reshape(collect(position), 2, 1),
                                        velocity=zeros(2, 1),
                                        mass=[particle_volume * fluid_density],
                                        density=[fluid_density], particle_spacing)

            WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel,
                                        smoothing_length,
                                        density_calculator=SummationDensity(),
                                        state_equation)
        end

        semi_ = Semidiscretization(fluid_systems..., rigid_system)
        ode = semidiscretize(semi_, (0.0, 0.01))
        semi = ode.p.semi

        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

        rigid = last(semi.systems)
        dv_rigid = TrixiParticles.wrap_v(dv_ode, rigid, semi)

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
                                                 smoothing_kernel, smoothing_length;
                                                 state_equation,
                                                 reference_particle_spacing=particle_spacing)

    rigid_ic = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                velocity=zeros(2, 1),
                                mass=[particle_volume * rigid_density],
                                density=[rigid_density], particle_spacing)
    rigid_system = RigidBodySystem(rigid_ic; boundary_model, acceleration=(0.0, 0.0))

    open_boundary_ic = InitialCondition(; coordinates=reshape([1.5, 0.0], 2, 1),
                                        velocity=zeros(2, 1),
                                        mass=[particle_volume * fluid_density],
                                        density=[fluid_density], particle_spacing)

    fluid_support_ic = InitialCondition(; coordinates=reshape([10.0, 10.0], 2, 1),
                                        velocity=zeros(2, 1),
                                        mass=[particle_volume * fluid_density],
                                        density=[fluid_density], particle_spacing)
    fluid_system = WeaklyCompressibleSPHSystem(fluid_support_ic; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=SummationDensity(),
                                               state_equation)

    boundary_face = ([2.0, -0.5], [2.0, 0.5])
    zone = BoundaryZone(; boundary_face, face_normal=(1.0, 0.0), density=fluid_density,
                        particle_spacing, initial_condition=open_boundary_ic,
                        open_boundary_layers=1, boundary_type=InFlow())

    open_boundary_system = OpenBoundarySystem(zone; fluid_system,
                                              boundary_model=BoundaryModelDynamicalPressureZhang(),
                                              buffer_size=0)

    semi_ = Semidiscretization(fluid_system, rigid_system, open_boundary_system)
    ode = semidiscretize(semi_, (0.0, 0.01))
    semi = ode.p.semi

    rigid = semi.systems[2]
    open_boundary = semi.systems[3]

    @test iszero(TrixiParticles.compact_support(rigid, open_boundary))
    @test iszero(TrixiParticles.compact_support(open_boundary, rigid))

    v_ode, u_ode = ode.u0.x
    dv_ode = zero(v_ode)

    TrixiParticles.interact!(dv_ode, v_ode, u_ode, rigid, open_boundary, semi)
    TrixiParticles.interact!(dv_ode, v_ode, u_ode, open_boundary, rigid, semi)

    dv_rigid = TrixiParticles.wrap_v(dv_ode, rigid, semi)
    dv_open_boundary = TrixiParticles.wrap_v(dv_ode, open_boundary, semi)

    @test all(iszero, dv_rigid[:, 1])
    @test all(iszero, dv_open_boundary[:, 1])
    @test iszero(rigid.resultant_force[])
    @test iszero(rigid.resultant_torque[])
end
