@trixi_testset "n_body/n_body_system.jl" begin
    include(joinpath(examples_dir(), "n_body", "n_body_system.jl"))

    # The system converts gravity parameters to its mass precision for a type-stable RHS.
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

    # Exercise softened acceleration through `kick!`, not just the scalar helper.
    semi32 = Semidiscretization(particle_system32, neighborhood_search=nothing,
                                parallelization_backend=SerialBackend())
    ode32 = semidiscretize(semi32, (0.0f0, 1.0f0))
    v32, u32 = ode32.u0.x
    dv32 = similar(v32)
    TrixiParticles.kick!(dv32, v32, u32,
                         (; semi=semi32, split_integration_data=nothing), 0.0f0)
    acceleration32 = TrixiParticles.wrap_v(dv32, particle_system32, semi32)
    inverse_softened_distance_cube = inv(1.0f0 + 0.25f0^2)^(3 / 2)

    @test acceleration32 ≈
          Float32[2inverse_softened_distance_cube -inverse_softened_distance_cube;
                  0 0]

    # The same RHS must omit a pair whose separation exceeds the finite cutoff.
    outside_coordinates32 = Float32[0 3;
                                    0 0]
    outside_ic32 = InitialCondition(; coordinates=outside_coordinates32,
                                    velocity=velocity32, density=1.0f0,
                                    mass=masses32, particle_spacing=-1.0f0)
    outside_system32 = NBodySystem(outside_ic32, gravity32)
    outside_semi32 = Semidiscretization(outside_system32, neighborhood_search=nothing,
                                        parallelization_backend=SerialBackend())
    outside_ode32 = semidiscretize(outside_semi32, (0.0f0, 1.0f0))
    v_outside32, u_outside32 = outside_ode32.u0.x
    dv_outside32 = similar(v_outside32)
    TrixiParticles.kick!(dv_outside32, v_outside32, u_outside32,
                         (; semi=outside_semi32, split_integration_data=nothing), 0.0f0)

    @test all(iszero, dv_outside32)

    # Softening makes coincident distinct particles regular; the unsoftened model rejects them.
    duplicate_coordinates = zeros(Float32, 2, 2)
    duplicate_ic = InitialCondition(; coordinates=duplicate_coordinates,
                                    velocity=velocity32,
                                    density=1.0f0,
                                    mass=masses32,
                                    particle_spacing=-1.0f0)
    duplicate_system = NBodySystem(duplicate_ic,
                                   NewtonianGravity(; gravitational_constant=1.0f0,
                                                    softening_length=0.25f0))
    duplicate_semi = Semidiscretization(duplicate_system,
                                        neighborhood_search=nothing,
                                        parallelization_backend=SerialBackend())
    duplicate_ode = semidiscretize(duplicate_semi, (0.0f0, 1.0f0))
    v_duplicate, u_duplicate = duplicate_ode.u0.x
    dv_duplicate = similar(v_duplicate)
    TrixiParticles.kick!(dv_duplicate, v_duplicate, u_duplicate,
                         (; semi=duplicate_semi,
                          split_integration_data=nothing), 0.0f0)

    @test all(iszero, dv_duplicate)

    singular_system = NBodySystem(duplicate_ic, 1.0f0)
    singular_semi = Semidiscretization(singular_system, neighborhood_search=nothing,
                                       parallelization_backend=SerialBackend())
    singular_ode = semidiscretize(singular_semi, (0.0f0, 1.0f0))
    v_singular, u_singular = singular_ode.u0.x
    dv_singular = similar(v_singular)

    @test_throws DomainError TrixiParticles.kick!(dv_singular, v_singular, u_singular,
                                                  (; semi=singular_semi,
                                                   split_integration_data=nothing),
                                                  0.0f0)

    # Potential-energy diagnostics use the same Plummer distance as the force model.
    energy_coordinates = Float64[0 3;
                                 0 0]
    energy_velocity = zeros(2, 2)
    energy_masses = [2.0, 4.0]
    energy_ic = InitialCondition(; coordinates=energy_coordinates,
                                 velocity=energy_velocity, density=1.0,
                                 mass=energy_masses, particle_spacing=-1.0)
    softened_energy_system = NBodySystem(energy_ic,
                                         NewtonianGravity(; gravitational_constant=5.0,
                                                          softening_length=4.0))
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

    # Shifting the finite-cutoff potential preserves its force and makes it continuous at r_c.
    shifted_energy_system = NBodySystem(energy_ic,
                                        NewtonianGravity(; gravitational_constant=5.0,
                                                         cutoff_radius=4.0))
    shifted_energy_semi = Semidiscretization(shifted_energy_system,
                                             neighborhood_search=nothing)
    shifted_energy_ode = semidiscretize(shifted_energy_semi, (0.0, 1.0))

    @test energy(shifted_energy_ode.u0.x..., shifted_energy_system,
                 shifted_energy_semi) ≈ -10 / 3

    boundary_energy_system = NBodySystem(energy_ic,
                                         NewtonianGravity(; gravitational_constant=5.0,
                                                          cutoff_radius=3.0))
    boundary_energy_semi = Semidiscretization(boundary_energy_system,
                                              neighborhood_search=nothing)
    boundary_energy_ode = semidiscretize(boundary_energy_semi, (0.0, 1.0))

    @test energy(boundary_energy_ode.u0.x..., boundary_energy_system,
                 boundary_energy_semi) == 0.0

    # Equal pair models produce equal-and-opposite forces across separate N-body systems.
    first_ic = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                velocity=zeros(2, 1), density=1.0,
                                mass=[2.0], particle_spacing=-1.0)
    second_ic = InitialCondition(; coordinates=reshape([1.0, 0.0], 2, 1),
                                 velocity=zeros(2, 1), density=1.0,
                                 mass=[3.0], particle_spacing=-1.0)
    shared_gravity = NewtonianGravity(; gravitational_constant=1.0,
                                      softening_length=0.25,
                                      cutoff_radius=2.0)
    first_system = NBodySystem(first_ic, shared_gravity)
    second_system = NBodySystem(second_ic, shared_gravity)
    coupled_semi = Semidiscretization(first_system, second_system,
                                      neighborhood_search=nothing,
                                      parallelization_backend=SerialBackend())
    coupled_ode = semidiscretize(coupled_semi, (0.0, 1.0))
    v_coupled, u_coupled = coupled_ode.u0.x
    dv_coupled = similar(v_coupled)
    TrixiParticles.kick!(dv_coupled, v_coupled, u_coupled,
                         (; semi=coupled_semi, split_integration_data=nothing), 0.0)
    first_acceleration = TrixiParticles.wrap_v(dv_coupled, first_system, coupled_semi)
    second_acceleration = TrixiParticles.wrap_v(dv_coupled, second_system, coupled_semi)
    total_force = first_system.mass[1] * first_acceleration[:, 1] +
                  second_system.mass[1] * second_acceleration[:, 1]

    @test total_force ≈ zeros(2)

    # Any differing force parameter would make the two ordered interactions non-reciprocal.
    incompatible_gravities = (NewtonianGravity(; gravitational_constant=2.0,
                                               softening_length=0.25,
                                               cutoff_radius=2.0),
                              NewtonianGravity(; gravitational_constant=1.0,
                                               softening_length=0.5,
                                               cutoff_radius=2.0),
                              NewtonianGravity(; gravitational_constant=1.0,
                                               softening_length=0.25,
                                               cutoff_radius=0.5))
    for incompatible_gravity in incompatible_gravities
        incompatible_system = NBodySystem(second_ic, incompatible_gravity)
        incompatible_semi = Semidiscretization(first_system, incompatible_system,
                                               neighborhood_search=nothing,
                                               parallelization_backend=SerialBackend())
        incompatible_ode = semidiscretize(incompatible_semi, (0.0, 1.0))
        v_incompatible, u_incompatible = incompatible_ode.u0.x
        dv_incompatible = similar(v_incompatible)
        @test_throws ArgumentError TrixiParticles.kick!(dv_incompatible, v_incompatible,
                                                        u_incompatible,
                                                        (; semi=incompatible_semi,
                                                         split_integration_data=nothing),
                                                        0.0)
    end

    # Keep the common unsoftened single-system RHS allocation-free.
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
