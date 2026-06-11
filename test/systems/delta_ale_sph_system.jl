@testset verbose=true "DeltaALESPHSystem" begin
    coordinates = [0.0 0.1 0.0 0.1
                   0.0 0.0 0.1 0.1]
    velocity = [0.2 -0.1 0.3 -0.2
                0.1 0.25 -0.15 -0.05]
    mass = [0.009, 0.011, 0.0105, 0.0095]
    density = [0.98, 1.02, 1.01, 0.99]
    particle_spacing = 0.1

    initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                         particle_spacing)
    smoothing_kernel = WendlandC2Kernel{2}()
    smoothing_length = 2particle_spacing

    system = DeltaALESPHSystem(initial_condition; smoothing_kernel, smoothing_length,
                               sound_speed=10.0, reference_density=1.0,
                               maximum_velocity=1.0)

    @test system isa DeltaALESPHSystem{2}
    @test system.mass == mass
    @test system.density_calculator isa ContinuityDensity
    @test system.state_equation.exponent == 1
    @test system.density_diffusion isa DensityDiffusionAntuono
    @test system.density_diffusion.delta == 0.1
    @test system.shifting_technique isa TrixiParticles.DeltaALEShifting
    @test TrixiParticles.v_nvariables(system) == 4

    v0 = zeros(TrixiParticles.v_nvariables(system), length(mass))
    TrixiParticles.write_v0!(v0, system)
    @test v0 == vcat(velocity, density', mass')

    vtk = Dict{String, Any}()
    TrixiParticles.write2vtk!(vtk, v0, coordinates, 0.0, system)
    @test vtk["mass"] == mass

    @test_throws ArgumentError DeltaALESPHSystem(initial_condition;
                                                 smoothing_kernel, smoothing_length,
                                                 sound_speed=10.0,
                                                 reference_density=1.0,
                                                 maximum_velocity=0.0)

    semi = Semidiscretization(system; neighborhood_search=nothing)
    ode = semidiscretize(semi, (0.0, 0.1))
    v_ode, u_ode = ode.u0.x
    dv_ode = similar(v_ode)

    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

    semi = ode.p.semi
    system = first(semi.systems)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    dv = TrixiParticles.wrap_v(dv_ode, system, semi)

    dm = view(dv, 4, :)
    @test isapprox(sum(dm), 0.0, atol=5e-15)

    momentum_derivative = sum(eachindex(mass)) do particle
        m = TrixiParticles.current_mass(v, system, particle)
        velocity_particle = TrixiParticles.current_velocity(v, system, particle)
        acceleration_particle = TrixiParticles.current_velocity(dv, system, particle)

        m * acceleration_particle + velocity_particle * dm[particle]
    end
    @test isapprox(momentum_derivative, zeros(2), atol=5e-14)

    maximum_shifting_velocity = maximum(eachindex(mass)) do particle
        norm(TrixiParticles.delta_v(system, particle))
    end
    @test maximum_shifting_velocity <= 0.5 + eps()
end
