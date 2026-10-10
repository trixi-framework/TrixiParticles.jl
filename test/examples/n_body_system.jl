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
                                 softening=PlummerSoftening(0.25),
                                 cutoff_radius=2.0)
    particle_system32 = NBodySystem(initial_condition32, gravity32)

    @test particle_system32.G === 1.0f0
    @test particle_system32.gravity.gravitational_constant === 1.0f0
    @test particle_system32.gravity.softening isa PlummerSoftening
    @test particle_system32.gravity.softening.softening_length === 0.25f0
    @test particle_system32.gravity.cutoff_radius === 2.0f0
    @test TrixiParticles.gravitational_mass(particle_system32, 2) === 2.0f0

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

    energy_coordinates = Float64[0 3;
                                 0 0]
    energy_velocity = zeros(2, 2)
    energy_masses = [2.0, 4.0]
    energy_ic = InitialCondition(; coordinates=energy_coordinates,
                                 velocity=energy_velocity, density=1.0,
                                 mass=energy_masses, particle_spacing=-1.0)
    softened_energy_system = NBodySystem(energy_ic,
                                         NewtonianGravity(; gravitational_constant=5.0,
                                                          softening=PlummerSoftening(4.0),
                                                          cutoff_radius=10.0))
    softened_energy_semi = Semidiscretization(softened_energy_system,
                                              neighborhood_search=nothing)
    softened_energy_ode = semidiscretize(softened_energy_semi, (0.0, 1.0))

    @test energy(softened_energy_ode.u0.x...,
                 softened_energy_system, softened_energy_semi) ≈ -8.0

    cutoff_energy_system = NBodySystem(energy_ic,
                                       NewtonianGravity(; gravitational_constant=5.0,
                                                        cutoff_radius=2.0))
    cutoff_energy_semi = Semidiscretization(cutoff_energy_system,
                                            neighborhood_search=nothing)
    cutoff_energy_ode = semidiscretize(cutoff_energy_semi, (0.0, 1.0))

    @test energy(cutoff_energy_ode.u0.x..., cutoff_energy_system,
                 cutoff_energy_semi) == 0.0

    function test_nbody_rhs_allocations()
        coordinates = Float64[0 1;
                              0 0]
        velocity = zeros(2, 2)
        masses = [1.0, 2.0]
        initial_condition = InitialCondition(; coordinates, velocity, density=1.0,
                                             mass=masses, particle_spacing=-1.0)
        particle_system = NBodySystem(initial_condition, 1.0)
        semi = Semidiscretization(particle_system, neighborhood_search=nothing)
        ode = semidiscretize(semi, (0.0, 1.0))

        sol = solve(ode, SymplecticEuler(), dt=0.1, save_everystep=false)

        @test count_rhs_allocations(sol) == 0
    end

    test_nbody_rhs_allocations()
end
