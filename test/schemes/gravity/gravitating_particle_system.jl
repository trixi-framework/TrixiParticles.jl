@trixi_testset "GravitatingParticleSystem" begin
    coordinates = Float32[0 2;
                          0 0]
    velocity = Float32[0 0;
                       0 0]
    mass = Float32[1, 3]
    acceleration = Float32[0 0;
                           1 1]

    system = GravitatingParticleSystem(; coordinates, velocity, mass, acceleration,
                                       particle_ids=[10, 20],
                                       gravity=NewtonianGravity())

    @test ndims(system) == 2
    @test eltype(system) === Float32
    @test coordinates_eltype(system) === Float32
    @test nparticles(system) == 2
    @test system.particle_ids == [10, 20]
    @test system.gravity.gravitational_constant === 1.0f0
    @test TrixiParticles.gravitational_mass(system, 2) === 3.0f0
    @test TrixiParticles.current_acceleration(system, 1) == SVector(0.0f0, 1.0f0)

    system3d = GravitatingParticleSystem(; coordinates=zeros(3, 1), mass=[1.0])

    @test ndims(system3d) == 3
    @test system3d.gravity.gravitational_constant === 1.0

    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(1, 1),
                                                         mass=[1.0])
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0])
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0, 1.0],
                                                         velocity=zeros(3, 2))
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0, 1.0],
                                                         particle_ids=[1, 1])
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0, -1.0])

    semi = Semidiscretization(system, neighborhood_search=nothing)
    ode = semidiscretize(semi, (0.0f0, 1.0f0))
    v, u = ode.u0.x
    dv = similar(v)

    TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0f0)
    dv_system = TrixiParticles.wrap_v(dv, system, semi)

    @test dv_system[:, 1] ≈ Float32[0.75, 1.0]
    @test dv_system[:, 2] ≈ Float32[-0.25, 1.0]

    no_gravity_system = GravitatingParticleSystem(; coordinates, velocity, mass,
                                                  acceleration, gravity=nothing)
    no_gravity_semi = Semidiscretization(no_gravity_system, neighborhood_search=nothing)
    no_gravity_ode = semidiscretize(no_gravity_semi, (0.0f0, 1.0f0))
    v_no_gravity, u_no_gravity = no_gravity_ode.u0.x
    dv_no_gravity = similar(v_no_gravity)

    TrixiParticles.kick!(dv_no_gravity, v_no_gravity, u_no_gravity,
                         (; semi=no_gravity_semi, split_integration_data=nothing),
                         0.0f0)
    dv_no_gravity_system = TrixiParticles.wrap_v(dv_no_gravity, no_gravity_system,
                                                 no_gravity_semi)

    @test dv_no_gravity_system == acceleration

    vtk_directory = mktempdir()
    trixi2vtk(ode.u0, semi, 0.0f0, output_directory=vtk_directory)

    @test isfile(joinpath(vtk_directory, "gravitating_particles_1_current.vtu"))
end
