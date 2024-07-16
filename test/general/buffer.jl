@testset verbose=true "`SystemBuffer`" begin
    # Mock fluid system
    struct FluidSystemMock3 <: TrixiParticles.FluidSystem{2, Nothing} end

    inflow = InFlow(; plane=([0.0, 0.0], [0.0, 1.0]), particle_spacing=0.2,
                    open_boundary_layers=2, density=1.0, flow_direction=[1.0, 0.0])
    system = OpenBoundarySPHSystem(inflow; sound_speed=1.0, fluid_system=FluidSystemMock3(),
                                   buffer_size=0)
    system_buffer = OpenBoundarySPHSystem(inflow; sound_speed=1.0, buffer_size=5,
                                          fluid_system=FluidSystemMock3())

    n_particles = nparticles(system)

    @testset "Iterators" begin
        @test TrixiParticles.each_moving_particle(system) == 1:n_particles

        @test TrixiParticles.each_moving_particle(system_buffer) == 1:n_particles

        particle_id = TrixiParticles.activate_next_particle(system_buffer)

        TrixiParticles.update_system_buffer!(system_buffer.buffer)

        @test TrixiParticles.each_moving_particle(system_buffer) == 1:(n_particles + 1)

        TrixiParticles.deactivate_particle!(system_buffer, particle_id,
                                            ones(2, particle_id))

        TrixiParticles.update_system_buffer!(system_buffer.buffer)

        @test TrixiParticles.each_moving_particle(system_buffer) == 1:n_particles

        particle_id = 5
        TrixiParticles.deactivate_particle!(system_buffer, particle_id,
                                            ones(2, particle_id))

        TrixiParticles.update_system_buffer!(system_buffer.buffer)

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
