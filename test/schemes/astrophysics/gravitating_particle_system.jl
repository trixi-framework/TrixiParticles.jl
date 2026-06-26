@trixi_testset "GravitatingParticleSystem" begin
    struct TestGravityModel <: TrixiParticles.AbstractGravityModel end

    coordinates = Float32[0 2;
                          0 0]
    velocity = Float32[0 0;
                       0 0]
    mass = Float32[1, 3]
    acceleration = Float32[0, 1]
    initial_condition = InitialCondition(; coordinates, velocity, mass,
                                         density=ones(Float32, 2),
                                         particle_spacing=-1.0f0)

    system = GravitatingParticleSystem(initial_condition; acceleration,
                                       particle_ids=[10, 20],
                                       gravity=NewtonianGravity())

    @test ndims(system) == 2
    @test eltype(system) === Float32
    @test system isa TrixiParticles.AbstractAstrophysicsSystem{2}
    @test system.initial_condition === initial_condition
    @test coordinates_eltype(system) === Float32
    @test nparticles(system) == 2
    @test system.particle_ids == [10, 20]
    @test system.gravity.gravitational_constant === 1.0f0
    @test TrixiParticles.gravitational_mass(system, 2) === 3.0f0
    @test TrixiParticles.current_acceleration(system, 1) == SVector(0.0f0, 1.0f0)
    @test TrixiParticles.particle_spacing(system, 1) === -1.0f0

    keyword_system = GravitatingParticleSystem(; coordinates, velocity, mass)

    @test keyword_system.initial_condition.coordinates == coordinates
    @test keyword_system.initial_condition.velocity == velocity
    @test keyword_system.mass == mass

    system3d = GravitatingParticleSystem(; coordinates=zeros(3, 1), mass=[1.0])

    @test ndims(system3d) == 3
    @test system3d.gravity.gravitational_constant === 1.0

    custom_gravity_system = GravitatingParticleSystem(initial_condition;
                                                      gravity=TestGravityModel())

    @test custom_gravity_system.gravity isa TestGravityModel
    @test TrixiParticles.compact_support(custom_gravity_system,
                                         custom_gravity_system) === Inf32
    custom_system_data = Dict{String, Any}()
    TrixiParticles.add_system_data!(custom_system_data, custom_gravity_system)
    @test custom_system_data["gravity_model"] == "TestGravityModel"
    @test !haskey(custom_system_data, "gravitational_constant")

    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(1, 1),
                                                         mass=[1.0])
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0])
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0, 1.0],
                                                         velocity=zeros(3, 2))
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0, 1.0],
                                                         acceleration=zeros(3))
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0, 1.0],
                                                         particle_ids=[1, 1])
    @test_throws ArgumentError GravitatingParticleSystem(; coordinates=zeros(2, 2),
                                                         mass=[1.0, -1.0])
    @test_throws ArgumentError GravitatingParticleSystem(initial_condition; gravity=1)

    semi = Semidiscretization(system, neighborhood_search=nothing)
    ode = semidiscretize(semi, (0.0f0, 1.0f0))
    v, u = ode.u0.x
    dv = similar(v)

    TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0f0)
    dv_system = TrixiParticles.wrap_v(dv, system, semi)

    @test dv_system[:, 1] ≈ Float32[0.75, 1.0]
    @test dv_system[:, 2] ≈ Float32[-0.25, 1.0]
    @test :external_acceleration in TrixiParticles.available_data(system)

    data = TrixiParticles.system_data(system, dv, nothing, v, u, semi)
    @test data.acceleration == dv_system
    @test data.external_acceleration == SVector(0.0f0, 1.0f0)

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

    @test dv_no_gravity_system[:, 1] == acceleration
    @test dv_no_gravity_system[:, 2] == acceleration

    vtk_directory = mktempdir()
    dvdu = similar(ode.u0)
    dvdu.x[1] .= dv
    fill!(dvdu.x[2], 0)
    trixi2vtk(dvdu, ode.u0, semi, 0.0f0, output_directory=vtk_directory,
              acceleration=(system, data, t) -> data.acceleration)

    vtk_file = joinpath(vtk_directory, "gravitating_particles_1_current.vtu")
    @test isfile(vtk_file)

    vtk = TrixiParticles.ReadVTK.VTKFile(vtk_file)
    point_data = TrixiParticles.ReadVTK.get_point_data(vtk)

    @test "mass" in keys(point_data)
    @test "particle_id" in keys(point_data)
    @test "external_acceleration" in keys(point_data)
    @test "acceleration" in keys(point_data)

    vtk_mass = TrixiParticles.ReadVTK.get_data(point_data["mass"])
    vtk_particle_id = TrixiParticles.ReadVTK.get_data(point_data["particle_id"])
    vtk_external_acceleration = TrixiParticles.ReadVTK.get_data(
        point_data["external_acceleration"])
    vtk_acceleration = TrixiParticles.ReadVTK.get_data(point_data["acceleration"])

    @test vec(vtk_mass) == mass
    @test vec(vtk_particle_id) == [10, 20]
    @test vtk_external_acceleration[1:2, :] == Float32[0 0; 1 1]
    @test vtk_acceleration[1:2, :] ≈ Float32[0.75 -0.25; 1.0 1.0]
end
