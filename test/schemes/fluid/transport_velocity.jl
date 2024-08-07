@trixi_testset "Transport Velocity Formulation" begin
    particle_spacing = 0.1
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.2particle_spacing

    fluid = rectangular_patch(particle_spacing, (3, 3), seed=1)

    v0_tvf = zeros(5, nparticles(fluid))

    system_tvf = EntropicallyDampedSPHSystem(fluid, smoothing_kernel,
                                             transport_velocity=TransportVelocityAdami(0.0),
                                             smoothing_length, 0.0)
    system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length, 0.0)

    @testset "Number of Variables" begin
        @test TrixiParticles.v_nvariables(system_tvf) == 5
        @test TrixiParticles.v_nvariables(system) == 3
    end

    @testset "write_v0!" begin
        TrixiParticles.write_v0!(v0_tvf, system_tvf)

        @test vcat(fluid.velocity, fluid.velocity, fluid.pressure') ≈ v0_tvf
    end

    @testset "Update" begin
        semi = Semidiscretization(system_tvf)
        fill!(v0_tvf, 1.5)
        v0_tvf[1:2, :] .= 2.5

        TrixiParticles.update_transport_velocity!(system_tvf, vec(v0_tvf), semi)

        @test fill(2.5, (4, nparticles(system_tvf))) ≈ v0_tvf[1:4, :]
    end
end
