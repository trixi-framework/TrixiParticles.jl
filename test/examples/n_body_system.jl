@trixi_testset "n_body/n_body_system.jl" begin
    include(joinpath(examples_dir(), "n_body", "n_body_system.jl"))

    coordinates32 = Float32[0 1;
                            0 0]
    velocity32 = zeros(Float32, 2, 2)
    masses32 = Float32[1, 2]
    initial_condition32 = InitialCondition(; coordinates=coordinates32,
                                           velocity=velocity32,
                                           density=1.0f0,
                                           mass=masses32,
                                           particle_spacing=-1.0f0)
    gravity32 = NewtonianGravity(; gravitational_constant=1.0,
                                 softening_length=0.25,
                                 cutoff_radius=2.0)
    particle_system32 = NBodySystem(initial_condition32, gravity32)

    @test particle_system32.G === 1.0f0
    @test particle_system32.gravity.gravitational_constant === 1.0f0
    @test particle_system32.gravity.softening_length === 0.25f0
    @test particle_system32.gravity.cutoff_radius === 2.0f0

    duplicate_coordinates = zeros(Float32, 2, 2)
    duplicate_ic = InitialCondition(; coordinates=duplicate_coordinates,
                                    velocity=velocity32,
                                    density=1.0f0,
                                    mass=masses32,
                                    particle_spacing=-1.0f0)
    duplicate_system = NBodySystem(duplicate_ic, 1.0f0)
    duplicate_semi = Semidiscretization(duplicate_system,
                                        neighborhood_search=nothing)
    duplicate_ode = semidiscretize(duplicate_semi, (0.0f0, 1.0f0))
    v_duplicate, u_duplicate = duplicate_ode.u0.x
    dv_duplicate = similar(v_duplicate)
    TrixiParticles.kick!(dv_duplicate, v_duplicate, u_duplicate,
                         (; semi=duplicate_semi,
                          split_integration_data=nothing), 0.0f0)

    @test all(iszero, dv_duplicate)
end
