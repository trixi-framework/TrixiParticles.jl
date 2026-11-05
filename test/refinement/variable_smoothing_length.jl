@testset verbose=true "Variable Smoothing Length" begin
    coordinates = [0.0 0.1 0.2
                   0.0 0.0 0.0]
    mass = [10.0, 10.0, 2.5]
    density = fill(1000.0, 3)
    initial_condition = InitialCondition(; coordinates, mass, density, particle_spacing=0.1)
    smoothing_kernel = WendlandC2Kernel{2}()
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)

    wcsph(;
          kwargs...) = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                   smoothing_length=0.12, state_equation,
                                                   density_calculator=ContinuityDensity(),
                                                   kwargs...)
    edac(;
         kwargs...) = EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                                  smoothing_length=0.12,
                                                  sound_speed=10.0, kwargs...)

    @testset "`ParticleRefinement` Constructor" begin
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=0.5,
                                                       min_spacing=0.05, max_spacing=0.1,
                                                       reference_density=1000.0)

        @test refinement.spacing_ratio == 0.5
        @test refinement.min_spacing == 0.05
        @test refinement.max_spacing == 0.1
        @test refinement.reference_density == 1000.0
        @test refinement.smoothing_length_factor == 1.2

        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=0.5,
                                                       min_spacing=0.05, max_spacing=0.1,
                                                       reference_density=1000.0,
                                                       smoothing_length_factor=1.5)
        @test refinement.smoothing_length_factor == 1.5
    end

    @testset "$(nameof(typeof(create_system()))) Without Refinement" for create_system in
                                                                         (wcsph,
                                                                          edac)
        system = create_system()

        @test isnothing(TrixiParticles.get_refinement(system))
        @test !TrixiParticles.has_variable_smoothing_length(system)
        @test TrixiParticles.smoothing_length(system) == 0.12
        @test TrixiParticles.smoothing_length(system, 2) == 0.12
        @test TrixiParticles.initial_smoothing_length(system) == 0.12
        @test TrixiParticles.particle_spacing(system, 3) == 0.1

        # The smoothing length is constant, so this must not change anything
        TrixiParticles.set_particle_smoothing_length!(system, 2, 0.2)
        @test TrixiParticles.smoothing_length(system, 2) == 0.12
    end

    @testset "$(nameof(typeof(create_system()))) With Refinement" for create_system in
                                                                      (wcsph,
                                                                       edac)
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=0.5,
                                                       min_spacing=0.05, max_spacing=0.1,
                                                       reference_density=1000.0)
        system = create_system(; particle_refinement=refinement)

        @test TrixiParticles.get_refinement(system) === refinement
        @test TrixiParticles.has_variable_smoothing_length(system)
        @test TrixiParticles.smoothing_length(system) == fill(0.12, 3)
        @test eltype(TrixiParticles.smoothing_length(system)) == eltype(system)
        @test TrixiParticles.initial_smoothing_length(system) ≈ 1.2 * 0.1

        TrixiParticles.set_particle_smoothing_length!(system, 2, 0.2)
        @test TrixiParticles.smoothing_length(system) == [0.12, 0.2, 0.12]
        @test TrixiParticles.smoothing_length(system, 2) == 0.2

        # The particle spacing follows from the mass
        @test TrixiParticles.particle_spacing(system, 1) ≈ 0.1
        @test TrixiParticles.particle_spacing(system, 3) ≈ 0.05

        TrixiParticles.set_particle_mass!(system, 1, 2.5)
        @test TrixiParticles.particle_spacing(system, 1) ≈ 0.05
    end

    @testset "Buffer Particles Have a Smoothing Length" begin
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=0.5,
                                                       min_spacing=0.05, max_spacing=0.1,
                                                       reference_density=1000.0)
        system = wcsph(; particle_refinement=refinement, buffer_size=2)

        @test length(TrixiParticles.smoothing_length(system)) == 5
    end
end
