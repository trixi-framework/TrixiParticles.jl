@testset verbose=true "MechanicalWorkCalculator" begin
    # Mock system
    struct MockSystem <: TrixiParticles.AbstractStructureSystem{2}
        eltype::Type
    end
    TrixiParticles.nparticles(::MockSystem) = 4
    Base.eltype(system::MockSystem) = system.eltype

    function TrixiParticles.create_neighborhood_search(neighborhood_search,
                                                       system::MockSystem, neighbor)
        return nothing
    end

    system32 = MockSystem(Float32)
    semi32 = Semidiscretization(system32)
    system64 = MockSystem(Float64)
    semi64 = Semidiscretization(system64)

    @testset "Constructor and Basic Properties" begin
        # Test default constructor
        calculator = MechanicalWorkCalculator(system64, semi64)
        @test calculator.system_index == 1
        @test calculator.t == 0.0
        @test calculator.work == 0.0
        @test !calculator.initialized
        @test calculator.dv isa Array{Float64, 2}
        @test size(calculator.dv) == (2, 4)
        @test calculator.eachparticle == 5:4
        @test calculated_mechanical_work(calculator) == 0.0

        # Test constructor with explicit particle range
        calculator = MechanicalWorkCalculator(system64, semi64; eachparticle=1:2)
        @test calculator.eachparticle == 1:2
        @test eltype(calculator.work) == Float64
        @test eltype(calculator.t) == Float64

        # Test with specific element type
        calculator = MechanicalWorkCalculator(system32, semi32)
        @test eltype(calculator.work) == Float32
        @test eltype(calculator.t) == Float32
    end



    @testset "update_mechanical_work_calculator!" begin
        # In the first test, we just move the 2x2 grid of particles up against gravity
        # and test that the accumulated work is just the potential energy difference.
        # In the other tests, we clamp the top row of particles and offset them to create
        # stress. We can then test how much work is required to pull the particles further
        # apart against the elastic forces.
        clamped_particles = [1:4, 3:4, 3:4, 3:4]
        E = [1e6, 1e1, 1e6, 1e8]
        @testset "clamped_particles = $(clamped_particles[i]), E = $(E[i])" for i in
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

            # Create TLSPH system with mechanical work calculator support
            system_ = TotalLagrangianSPHSystem(initial_condition; smoothing_kernel,
                                               smoothing_length, young_modulus,
                                               poisson_ratio,
                                               clamped_particles_motion=prescribed_motion,
                                               clamped_particles=clamped_particles[i],
                                               acceleration=(0.0, -2.0))

            semi = Semidiscretization(system_)
            ode = semidiscretize(semi, (0.0, 1.0))
            system = ode.p.semi.systems[1]

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
            work1 = 0.0
            dt1 = 0.1

            # Test that mechanical work is integrated, i.e., values of instantaneous
            # power are accumulated over time. This initial work should just be an offset.
            # Also, half the step size means half the work increase.
            work2 = 1.0
            dt2 = 0.05

            # Test `only_compute_force_on_fluid`
            work3 = 0.0
            dt3 = 0.1

            eachparticle = (TrixiParticles.n_integrated_particles(system) + 1):nparticles(system)
            dv = zeros(2, nparticles(system))

            work1 = TrixiParticles.update_mechanical_work!(work1, system, eachparticle,
                                                           false, dv, v_ode, u_ode, semi,
                                                           0.0, dt1)
            work2 = TrixiParticles.update_mechanical_work!(work2, system, eachparticle,
                                                           false, dv, v_ode, u_ode, semi,
                                                           0.0, dt2)
            work3 = TrixiParticles.update_mechanical_work!(work3, system, eachparticle,
                                                           true, dv, v_ode, u_ode, semi,
                                                           0.0, dt3)

            if i == 1
                @test isapprox(work1, 0.8)
                @test isapprox(work2, 1.0 + 0.4)
            elseif i == 2
                # For very soft material, we can just pull up the top row of particles
                # and the work required is almost just the potential energy difference.
                @test isapprox(work1, 0.4080357142857143)
                @test isapprox(work2, 1.0 + 0.5 * 0.4080357142857143)
            elseif i == 3
                # For a stiffer material, the stress from the offset creates larger forces
                # pulling the clamped particles back down, so we need a lot of work
                # to pull the material apart.
                @test isapprox(work1, 803.9714285714281)
                @test isapprox(work2, 1.0 + 0.5 * 803.9714285714281)
            elseif i == 4
                # For a very stiff material, the work is even larger.
                @test isapprox(work1, 80357.5428571428)
                @test isapprox(work2, 1.0 + 0.5 * 80357.5428571428)
            end

            @test isapprox(work3, 0.0)
        end
    end
end
