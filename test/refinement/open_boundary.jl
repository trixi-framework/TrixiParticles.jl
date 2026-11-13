@testset verbose=true "Open Boundary" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    sound_speed = 10.0
    reference_density = 1000.0
    max_spacing = 0.1
    open_boundary_spacing = 0.05
    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2, min_spacing=0.05,
                                                   max_spacing, reference_density)

    # The default update strategy does not support inactive particles
    neighborhood_search = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    # The fluid is right of the face at `x = 0`, the inflow zone left of it
    fluid = RectangularShape(max_spacing, (5, 5), (0.0, 0.0); density=reference_density)
    boundary_zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 0.5]),
                                 face_normal=[1.0, 0.0], boundary_type=InFlow(),
                                 open_boundary_layers=5, density=reference_density,
                                 particle_spacing=open_boundary_spacing)

    function create_systems(; particle_refinement=refinement)
        fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length=1.2 * max_spacing,
                                                   sound_speed,
                                                   density_calculator=ContinuityDensity(),
                                                   particle_refinement, buffer_size=4)
        open_boundary = OpenBoundarySystem(boundary_zone; fluid_system, buffer_size=4,
                                           boundary_model=BoundaryModelDynamicalPressureZhang())

        return fluid_system, open_boundary
    end

    @testset verbose=true "Smoothing Length" begin
        # Without refinement, the open boundary uses the smoothing length of the fluid
        _, open_boundary = create_systems(particle_refinement=nothing)
        @test !TrixiParticles.has_variable_smoothing_length(open_boundary)
        @test TrixiParticles.smoothing_length(open_boundary, 1) == 1.2 * max_spacing

        # With refinement, every open boundary particle has the smoothing length
        # of the spacing of the boundary zone
        _, open_boundary = create_systems()
        @test TrixiParticles.has_variable_smoothing_length(open_boundary)
        @test TrixiParticles.smoothing_length(open_boundary) isa Vector
        @test all(==(1.2 * open_boundary_spacing),
                  TrixiParticles.smoothing_length(open_boundary))
        @test TrixiParticles.initial_smoothing_length(open_boundary) ≈
              1.2 * open_boundary_spacing
    end

    @testset verbose=true "Particle Transfer" begin
        semi = Semidiscretization(create_systems()...; neighborhood_search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x
        semi = ode.p.semi
        fluid_system, open_boundary = semi.systems

        v = TrixiParticles.wrap_v(v_ode, open_boundary, semi)
        u = TrixiParticles.wrap_u(u_ode, open_boundary, semi)
        u_fluid = TrixiParticles.wrap_u(u_ode, fluid_system, semi)

        n_fluid = nparticles(fluid)
        fluid_mass = reference_density * max_spacing^2
        boundary_mass = reference_density * open_boundary_spacing^2

        # An open boundary particle that carries the smoothing length of a coarse fluid
        # particle (e.g., after leaving the fluid domain in a bidirectional zone)
        # enters the fluid domain
        inflow_particle = 1
        TrixiParticles.set_particle_smoothing_length!(open_boundary, inflow_particle,
                                                      1.2 * max_spacing)
        u[:, inflow_particle] .= (0.01, 0.25)

        # A fluid particle enters the boundary zone
        outflow_particle = 1
        TrixiParticles.set_particle_position!(u_fluid, fluid_system, outflow_particle,
                                              (-0.01, 0.05))

        TrixiParticles.check_domain!(open_boundary, v, u, v_ode, u_ode, semi)

        # The open boundary particle became the first free fluid particle
        new_fluid_particle = n_fluid + 1
        @test fluid_system.buffer.active_particle[new_fluid_particle]
        @test fluid_system.mass[new_fluid_particle] ≈ boundary_mass
        @test TrixiParticles.smoothing_length(fluid_system, new_fluid_particle) ≈
              1.2 * max_spacing

        # Its target spacing follows from its mass, not from the values of the buffer slot
        @test fluid_system.cache.target_spacing[new_fluid_particle] ≈ open_boundary_spacing
        @test !fluid_system.cache.is_fixed[new_fluid_particle]

        # The open boundary particle is reset to the back of the zone with the smoothing
        # length and the mass of the zone
        @test open_boundary.buffer.active_particle[inflow_particle]
        @test TrixiParticles.smoothing_length(open_boundary, inflow_particle) ≈
              1.2 * open_boundary_spacing
        density = TrixiParticles.current_density(v, open_boundary, inflow_particle)
        @test open_boundary.mass[inflow_particle] ≈ density * open_boundary_spacing^2

        # The fluid particle became the first free open boundary particle
        @test !fluid_system.buffer.active_particle[outflow_particle]
        new_boundary_particle = nparticles(boundary_zone.initial_condition) + 1
        @test open_boundary.buffer.active_particle[new_boundary_particle]
        @test open_boundary.mass[new_boundary_particle] ≈ fluid_mass
        @test TrixiParticles.smoothing_length(open_boundary, new_boundary_particle) ≈
              1.2 * max_spacing
    end

    @testset verbose=true "EDAC Damping Term" begin
        _, open_boundary = create_systems()
        grad_kernel = SVector(1.0, 2.0)
        grad_kernel_a = SVector(3.0, 4.0)

        # Open boundary particles are fluid neighbors (`fluids_with_io` in the code of
        # Haftu et al.), so they use the averaged kernel gradient and `beta`
        @test TrixiParticles.damping_kernel_grad_and_beta(open_boundary, grad_kernel,
                                                          grad_kernel_a, 0.5) ==
              (grad_kernel, 0.5)
    end
end
