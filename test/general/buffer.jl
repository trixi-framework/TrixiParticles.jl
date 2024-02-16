@testset verbose=true "SystemBuffer" begin
    system = OpenBoundarySPHSystem(([0.0, 0.0], [0.0, 1.0]), InFlow(), 1.0;
                                   flow_direction=[1.0, 0.0], particle_spacing=0.2,
                                   open_boundary_layers=2, density=1.0)
    system_buffer = OpenBoundarySPHSystem(([0.0, 0.0], [0.0, 1.0]), InFlow(), 1.0;
                                          flow_direction=[1.0, 0.0], particle_spacing=0.2,
                                          open_boundary_layers=2, density=1.0, buffer=5)

    n_particles = nparticles(system)

    @testset "Iterators" begin
        @test Base.OneTo(n_particles) == TrixiParticles.each_moving_particle(system)

        @test Base.OneTo(n_particles) == TrixiParticles.each_moving_particle(system_buffer)

        particle_ID = TrixiParticles.available_particle(system_buffer)

        TrixiParticles.update!(system_buffer.buffer)

        @test Base.OneTo(n_particles + 1) ==
              TrixiParticles.each_moving_particle(system_buffer)

        TrixiParticles.deactivate_particle!(system_buffer, particle_ID,
                                            ones(2, particle_ID))

        TrixiParticles.update!(system_buffer.buffer)

        @test Base.OneTo(n_particles) == TrixiParticles.each_moving_particle(system_buffer)

        particle_ID = 5
        TrixiParticles.deactivate_particle!(system_buffer, particle_ID,
                                            ones(2, particle_ID))

        TrixiParticles.update!(system_buffer.buffer)

        @test setdiff(Base.OneTo(n_particles), particle_ID) ==
              TrixiParticles.each_moving_particle(system_buffer)
    end

    @testset "Allocate Buffer" begin
        initial_condition = rectangular_patch(0.1, (3, 3), perturbation_factor=0.0)
        buffer = TrixiParticles.SystemBuffer(nparticles(initial_condition), 7)

        ic_with_buffer = TrixiParticles.allocate_buffer(initial_condition, buffer)

        @test nparticles(initial_condition) + 7 == nparticles(ic_with_buffer)

        masses = initial_condition.mass[1] .* ones(nparticles(ic_with_buffer))
        @test masses == ic_with_buffer.mass

        densities = initial_condition.density[1] .* ones(nparticles(ic_with_buffer))
        @test densities == ic_with_buffer.density

        pressures = initial_condition.pressure[1] .* ones(nparticles(ic_with_buffer))
        @test pressures == ic_with_buffer.pressure

        @testset "Illegal Input" begin
            ic = rectangular_patch(0.1, (3, 3))
            buffer = TrixiParticles.SystemBuffer(9, 7)
            error_str = "`density` needs to be constant when using `SystemBuffer`"
            @test_throws ArgumentError(error_str) TrixiParticles.allocate_buffer(ic, buffer)
        end
    end
end
