@testset "Rigid-body corrected summation density" begin
    configurations = Iterators.product((2, 3), (false, true), (false, true))
    @testset "$config" for config in configurations
        dimensions, reverse_order, precomputed = config
        spacing = 0.1
        density = 1000.0
        smoothing_length = 2spacing
        smoothing_kernel = WendlandC6Kernel{dimensions}()
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=density, exponent=1)
        fluid_corner = ntuple(i -> i == dimensions ? 0.3 : 0.0, dimensions)
        fluid_particles = RectangularShape(spacing, ntuple(_ -> 3, dimensions),
                                           fluid_corner; density)
        rigid_particles = RectangularShape(spacing, ntuple(_ -> 3, dimensions),
                                           ntuple(_ -> 0.0, dimensions); density=2density)
        fluid = WeaklyCompressibleSPHSystem(fluid_particles;
                                            smoothing_kernel, smoothing_length,
                                            state_equation,
                                            density_calculator=ContinuityDensity())
        n_rigid = TrixiParticles.nparticles(rigid_particles)
        hydrodynamic_mass = fill(density * spacing^dimensions, n_rigid)
        boundary_model = BoundaryModelDummyParticles(fill(density, n_rigid),
                                                     hydrodynamic_mass,
                                                     SummationDensity(), smoothing_kernel,
                                                     smoothing_length; state_equation,
                                                     density_correction=ShepardKernelCorrection())
        rigid = RigidBodySystem(rigid_particles; boundary_model)
        systems = reverse_order ? (rigid, fluid) : (fluid, rigid)
        neighborhood_search = precomputed ? PrecomputedNeighborhoodSearch{dimensions}() :
                              nothing
        semi = Semidiscretization(systems...; neighborhood_search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode, u_ode = map(Array, ode.u0.x)
        dv_ode = zero(v_ode)

        # Re-evaluating the RHS at an unchanged equilibrium must not repeatedly normalize
        # an old density or produce pressure/acceleration from truncated kernel support.
        for _ in 1:3
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
            @test boundary_model.cache.density ≈ fill(density, n_rigid)
            @test isapprox(boundary_model.pressure, zeros(n_rigid); atol=1e-8)
            @test isapprox(dv_ode, zero(dv_ode); atol=1e-8)
        end

        # Change both the neighborhood and fluid volumes. Independent pairwise sums
        # verify that the next update uses a fresh hydrodynamic numerator and coefficient.
        v_fluid = TrixiParticles.wrap_v(v_ode, fluid, semi)
        u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
        u_rigid = TrixiParticles.wrap_u(u_ode, rigid, semi)
        v_fluid[end, :] .*= 1.05
        u_rigid[1, :] .+= spacing / 4
        neighbor_coords = hcat(Array(u_fluid), Array(u_rigid))
        neighbor_masses = vcat(fluid.mass, hydrodynamic_mass)
        neighbor_densities = vcat(collect(v_fluid[end, :]),
                                  copy(boundary_model.cache.density))
        expected_density = map(1:n_rigid) do particle
            values = [TrixiParticles.kernel(smoothing_kernel,
                                            norm(u_rigid[:, particle] - coords),
                                            smoothing_length)
                      for coords in eachcol(neighbor_coords)]
            numerator = dot(neighbor_masses, values)
            coefficient = dot(neighbor_masses ./ neighbor_densities, values)
            numerator / coefficient
        end

        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
        @test boundary_model.cache.density ≈ expected_density
        @test isapprox(boundary_model.pressure, state_equation.(expected_density);
                       rtol=1e-10, atol=1e-8)
    end
end
