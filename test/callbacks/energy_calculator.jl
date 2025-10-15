@testset verbose=true "EnergyCalculatorCallback" begin
    # Mock system
    struct MockSystem <: TrixiParticles.AbstractSystem{2} end
    TrixiParticles.nparticles(::MockSystem) = 4

    function TrixiParticles.create_neighborhood_search(neighborhood_search,
                                                       system::MockSystem, neighbor)
        return nothing
    end

    system = MockSystem()
    semi = Semidiscretization(system)

    @testset "Constructor and Basic Properties" begin
        # Test default constructor
        callback = EnergyCalculatorCallback{Float64}(system, semi)
        @test callback.affect!.system_index == 1
        @test callback.affect!.interval == 1
        @test callback.affect!.t[] == 0.0
        @test callback.affect!.energy[] == 0.0
        @test callback.affect!.dv isa Array{Float64, 2}
        @test size(callback.affect!.dv) == (2, 4)
        @test callback.affect!.eachparticle == 1:4
        @test calculated_energy(callback) == 0.0

        # Test constructor with interval
        callback = EnergyCalculatorCallback{Float64}(system, semi; interval=5)
        @test callback.affect!.interval == 5

        # Test with specific element type
        callback = EnergyCalculatorCallback{Float32}(system, semi; interval=2)
        @test eltype(callback.affect!.energy) == Float32
        @test eltype(callback.affect!.t) == Float32
    end

    @testset "show" begin
        callback = EnergyCalculatorCallback{Float64}(system, semi; interval=10)

        # Test compact representation
        show_compact = "EnergyCalculatorCallback{Float64}(interval=10)"
        @test repr(callback) == show_compact

        # Test detailed representation - check against expected box format
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ EnergyCalculatorCallback{Float64}                                                                │
        │ ═════════════════════════════════                                                                │
        │ interval: ……………………………………………………… 10                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box
    end

    @testset "update_energy_calculator!" begin
        # In the first test, we just move the 2x2 grid of particles up against gravity
        # and test that the energy calculated is just the potential energy difference.
        # In the other tests, we clamp the top row of particles and offset them to create
        # stress. We can then test how much energy is required to pull the particles further
        # apart against the elastic forces.
        n_clamped_particles = [4, 2, 2, 2]
        E = [1e6, 1e1, 1e6, 1e8]
        @testset "n_clamped_particles = $(n_clamped_particles[i]), E = $(E[i])" for i in
                                                                                    eachindex(E)
            # Create a simple 2D system with 2x2 particles
            coordinates = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]
            mass = [1.0, 1.0, 1.0, 1.0]
            density = [1000.0, 1000.0, 1000.0, 1000.0]

            smoothing_kernel = SchoenbergCubicSplineKernel{2}()
            smoothing_length = sqrt(2)
            young_modulus = E[i]
            poisson_ratio = 0.4

            initial_condition = InitialCondition(; coordinates, mass, density)

            function movement_function(initial_pos, t)
                # Offset clamped particles to create stress
                return initial_pos + SVector(0.0, t + 0.5)
            end
            is_moving(t) = true
            prescribed_motion = PrescribedMotion(movement_function, is_moving)

            # Create TLSPH system with energy calculator support
            system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                              smoothing_length, young_modulus,
                                              poisson_ratio,
                                              clamped_particles_motion=prescribed_motion,
                                              n_clamped_particles=n_clamped_particles[i],
                                              acceleration=(0.0, -2.0),
                                              use_with_energy_calculator_callback=true)

            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 1.0))

            # Create dummy ODE state vectors
            v = zeros(2, TrixiParticles.n_integrated_particles(system))
            u = coordinates[:, 1:TrixiParticles.n_integrated_particles(system)]
            v_ode = vec(v)
            u_ode = vec(u)

            # Update system
            TrixiParticles.initialize!(system, semi)
            TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi, 0.0)
            TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi, 0.0)

            # Set up test parameters
            energy1 = Ref(0.0)
            dt1 = 0.1

            # Test that energy is integrated, i.e., values of instantaneous power
            # are accumulated over time. This initial energy should just be an offset.
            # Also, half the step size means half the energy increase.
            energy2 = Ref(1.0)
            dt2 = 0.05

            eachparticle = (TrixiParticles.n_integrated_particles(system) + 1):nparticles(system)
            dv = zeros(2, nparticles(system))

            TrixiParticles.update_energy_calculator!(energy1, system, eachparticle, dv,
                                                     v_ode, u_ode, semi, 0.0, dt1)
            TrixiParticles.update_energy_calculator!(energy2, system, eachparticle, dv,
                                                     v_ode, u_ode, semi, 0.0, dt2)

            if i == 1
                @test isapprox(energy1[], 0.8)
                @test isapprox(energy2[], 1.0 + 0.4)
            elseif i == 2
                # For very soft material, we can just pull up the top row of particles
                # and the energy required is almost just the potential energy difference.
                @test isapprox(energy1[], 0.4080357142857143)
                @test isapprox(energy2[], 1.0 + 0.5 * 0.4080357142857143)
            elseif i == 3
                # For a stiffer material, the stress from the offset creates larger forces
                # pulling the clamped particles back down, so we need a lot of energy
                # to pull the material apart.
                @test isapprox(energy1[], 803.9714285714281)
                @test isapprox(energy2[], 1.0 + 0.5 * 803.9714285714281)
            elseif i == 4
                # For a very stiff material, the energy is even larger.
                @test isapprox(energy1[], 80357.5428571428)
                @test isapprox(energy2[], 1.0 + 0.5 * 80357.5428571428)
            end
        end
    end
end
