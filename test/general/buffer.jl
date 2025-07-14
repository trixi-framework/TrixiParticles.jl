@testset verbose=true "`SystemBuffer`" begin
    # Mock fluid system
    struct FluidSystemMock3 <: TrixiParticles.FluidSystem{2} end
    TrixiParticles.initial_smoothing_length(system::FluidSystemMock3) = 1.0
    TrixiParticles.nparticles(system::FluidSystemMock3) = 1

    zone = BoundaryZone(; plane=([0.0, 0.0], [0.0, 1.0]), particle_spacing=0.2,
                        open_boundary_layers=2, density=1.0, plane_normal=[1.0, 0.0],
                        boundary_type=InFlow())
    system = OpenBoundarySPHSystem(zone; fluid_system=FluidSystemMock3(),
                                   reference_density=0.0, reference_pressure=0.0,
                                   reference_velocity=[0, 0],
                                   boundary_model=BoundaryModelLastiwka(), buffer_size=0)
    system_buffer = OpenBoundarySPHSystem(zone; buffer_size=5,
                                          reference_density=0.0, reference_pressure=0.0,
                                          reference_velocity=[0, 0],
                                          boundary_model=BoundaryModelLastiwka(),
                                          fluid_system=FluidSystemMock3())

    n_particles = nparticles(system)

    @testset "Iterators" begin
        @test TrixiParticles.each_moving_particle(system) == 1:n_particles

        @test TrixiParticles.each_moving_particle(system_buffer) == 1:n_particles

        # Activate a particle
        particle_id = findfirst(==(false), system_buffer.buffer.active_particle)
        system_buffer.buffer.active_particle[particle_id] = true

        TrixiParticles.update_system_buffer!(system_buffer.buffer,
                                             DummySemidiscretization())

        @test TrixiParticles.each_moving_particle(system_buffer) == 1:(n_particles + 1)

        TrixiParticles.deactivate_particle!(system_buffer, particle_id,
                                            ones(2, particle_id))

        TrixiParticles.update_system_buffer!(system_buffer.buffer,
                                             DummySemidiscretization())

        @test TrixiParticles.each_moving_particle(system_buffer) == 1:n_particles

        particle_id = 5
        TrixiParticles.deactivate_particle!(system_buffer, particle_id,
                                            ones(2, particle_id))

        TrixiParticles.update_system_buffer!(system_buffer.buffer,
                                             DummySemidiscretization())

        @test TrixiParticles.each_moving_particle(system_buffer) ==
              setdiff(1:n_particles, particle_id)
    end

    @testset "Allocate Buffer" begin
        initial_condition = rectangular_patch(0.1, (3, 3), perturbation_factor=0.0)
        buffer = TrixiParticles.SystemBuffer(nparticles(initial_condition), 7)

        ic_with_buffer = TrixiParticles.allocate_buffer(initial_condition, buffer)

        @test nparticles(ic_with_buffer) == nparticles(initial_condition) + 7

        masses = initial_condition.mass[1] .* ones(nparticles(ic_with_buffer))
        @test masses == ic_with_buffer.mass

        densities = initial_condition.density[1] .* ones(nparticles(ic_with_buffer))
        @test densities == ic_with_buffer.density

        pressures = initial_condition.pressure[1] .* ones(nparticles(ic_with_buffer))
        @test pressures == ic_with_buffer.pressure

        @testset "Illegal Input" begin
            # The rectangular patch has a perturbed, non-constant density
            ic = rectangular_patch(0.1, (3, 3))
            buffer = TrixiParticles.SystemBuffer(9, 7)

            error_str = "`initial_condition.density` needs to be constant when using `SystemBuffer`"
            @test_throws ArgumentError(error_str) TrixiParticles.allocate_buffer(ic, buffer)
        end
    end
end
