@testset verbose=true "Dummy Particles RHS" begin
    # This is a more complicated version of the total energy preservation test in
    # `test/schemes/fluid/weakly_compressible_sph/rhs.jl`. See the comments there.
    #
    # Note that this only works for boundary models where the potential energy is included
    # in the internal energy. For repulsive boundary particles (Monghan-Kajtar), this does
    # not work. A fluid particle close to a repulsive particle has a high potential energy
    # that will be converted to kinetic energy when running a simulation.
    # In order to make this test work, we would have to calculate this potential energy
    # somehow. The same holds for `PressureMirroring`.
    # For `AdamiPressureExtrapolation`, we would have to calculate a complicated derivative
    # of the boundary particle density.
    @testset verbose=true "Momentum and Total Energy Conservation" begin
        # We are testing the momentum conservation of SPH with random initial configurations
        density_calculators = [ContinuityDensity(), SummationDensity()]

        # Create several neighbor systems to test fluid-neighbor interaction
        function second_systems(initial_condition, density_calculator, state_equation,
                                smoothing_kernel, smoothing_length)
            second_fluid_system = WeaklyCompressibleSPHSystem(initial_condition,
                                                              density_calculator,
                                                              state_equation,
                                                              smoothing_kernel,
                                                              smoothing_length)

            # Overwrite `second_fluid_system.pressure` because we skip the update step
            second_fluid_system.pressure .= initial_condition.pressure

            u_fluid = initial_condition.coordinates
            if density_calculator isa SummationDensity
                # Density is stored in the cache
                v_fluid = initial_condition.velocity
                second_fluid_system.cache.density .= initial_condition.density
            else
                # Density is integrated with `ContinuityDensity`
                v_fluid = vcat(initial_condition.velocity, initial_condition.density')
            end

            # Boundary systems.
            # Note that we don't need to explicitly test dummy
            # particles with `SummationDensity` and `AdamiPressureExtrpolation`,
            # since the same code will be run as for `ContinuityDensity`.
            # The advantage of `ContinuityDensity` is that the density will be
            # perturbed without having to run a summation or Adami update step.
            boundary_model_zeroing = BoundaryModelDummyParticles(initial_condition.density,
                                                                 initial_condition.mass,
                                                                 PressureZeroing(),
                                                                 smoothing_kernel,
                                                                 smoothing_length)
            boundary_system_zeroing = WallBoundarySystem(initial_condition,
                                                         boundary_model_zeroing)
            boundary_model_continuity = BoundaryModelDummyParticles(initial_condition.density,
                                                                    initial_condition.mass,
                                                                    ContinuityDensity(),
                                                                    smoothing_kernel,
                                                                    smoothing_length)
            # Overwrite `boundary_model_continuity.pressure` because we skip the update step
            boundary_model_continuity.pressure .= initial_condition.pressure
            boundary_system_continuity = WallBoundarySystem(initial_condition,
                                                            boundary_model_continuity)

            boundary_model_summation = BoundaryModelDummyParticles(initial_condition.density,
                                                                   initial_condition.mass,
                                                                   SummationDensity(),
                                                                   smoothing_kernel,
                                                                   smoothing_length)
            # Overwrite `boundary_model_summation.pressure` because we skip the update step
            boundary_model_summation.pressure .= initial_condition.pressure
            # Density is stored in the cache
            boundary_model_summation.cache.density .= initial_condition.density
            boundary_system_summation = WallBoundarySystem(initial_condition,
                                                           boundary_model_summation)

            u_boundary = zeros(0, TrixiParticles.nparticles(initial_condition))
            v_boundary = zeros(0, TrixiParticles.nparticles(initial_condition))

            v_boundary_continuity = copy(initial_condition.density')

            # TLSPH system
            structure_system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                                        smoothing_length, 0.0, 0.0,
                                                        boundary_model=boundary_model_continuity)

            # Positions of the structure particles are not used here
            u_structure = zeros(0, TrixiParticles.nparticles(structure_system))
            v_structure = vcat(initial_condition.velocity,
                               initial_condition.density')

            systems = Dict(
                "Fluid-Fluid" => second_fluid_system,
                "Fluid-BoundaryDummyPressureZeroing" => boundary_system_zeroing,
                "Fluid-BoundaryDummyContinuityDensity" => boundary_system_continuity,
                "Fluid-TLSPH" => structure_system
            )

            if density_calculator isa SummationDensity
                # Dummy particles with `SummationDensity` will only pass energy preservation
                # if the fluid is also using `SummationDensity`.
                systems["Fluid-BoundaryDummySummationDensity"] = boundary_system_summation
            end

            vu = Dict(
                "Fluid-Fluid" => (v_fluid, u_fluid),
                "Fluid-BoundaryDummyPressureZeroing" => (v_boundary, u_boundary),
                "Fluid-BoundaryDummyContinuityDensity" => (v_boundary_continuity,
                                                           u_boundary),
                "Fluid-BoundaryDummySummationDensity" => (v_boundary, u_boundary),
                "Fluid-TLSPH" => (v_structure, u_structure)
            )

            return systems, vu
        end

        particle_spacing = 0.1

        # The state equation is only needed to unpack `sound_speed`, so we can mock
        # it by using a `NamedTuple`.
        state_equation = (; sound_speed=0.0)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2particle_spacing
        search_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        @testset "`$(nameof(typeof(density_calculator)))`" for density_calculator in
                                                               density_calculators
            # Run three times with different seed for the random initial condition
            for seed in 1:3
                # A larger number of particles will increase accumulated errors in the
                # summation. A larger tolerance will have to be used for the tests below.
                ic = rectangular_patch(particle_spacing, (3, 3), seed=seed)

                # Split initial condition at center particle into two systems
                center_particle = ceil(Int, TrixiParticles.nparticles(ic) / 2)
                fluid = InitialCondition{ndims(ic)}(ic.coordinates[:, 1:center_particle],
                                                    ic.velocity[:, 1:center_particle],
                                                    ic.mass[1:center_particle],
                                                    ic.density[1:center_particle],
                                                    ic.pressure[1:center_particle],
                                                    ic.particle_spacing)

                boundary = InitialCondition{ndims(ic)}(ic.coordinates[:,
                                                                      (center_particle + 1):end],
                                                       ic.velocity[:,
                                                                   (center_particle + 1):end],
                                                       ic.mass[(center_particle + 1):end],
                                                       ic.density[(center_particle + 1):end],
                                                       ic.pressure[(center_particle + 1):end],
                                                       ic.particle_spacing)

                fluid_system = WeaklyCompressibleSPHSystem(fluid,
                                                           density_calculator,
                                                           state_equation,
                                                           smoothing_kernel,
                                                           smoothing_length)
                n_particles = TrixiParticles.nparticles(fluid_system)

                # Overwrite `fluid_system.pressure` because we skip the update step
                fluid_system.pressure .= fluid.pressure

                u = fluid.coordinates
                if density_calculator isa SummationDensity
                    # Density is stored in the cache
                    v = fluid.velocity
                    fluid_system.cache.density .= fluid.density
                else
                    # Density is integrated with `ContinuityDensity`
                    v = vcat(fluid.velocity, fluid.density')
                end

                # Create several neighbor systems to test fluid-neighbor interaction
                systems,
                vu = second_systems(boundary, density_calculator,
                                    state_equation,
                                    smoothing_kernel, smoothing_length)

                @testset "$key" for key in keys(systems)
                    neighbor_system = systems[key]
                    v_neighbor, u_neighbor = vu[key]

                    semi = DummySemidiscretization()

                    # Compute interactions
                    dv = zero(v)

                    # Fluid-fluid interact
                    TrixiParticles.interact!(dv, v, u, v, u,
                                             fluid_system, fluid_system, semi)

                    # Fluid-neighbor interact
                    TrixiParticles.interact!(dv, v, u, v_neighbor, u_neighbor,
                                             fluid_system, neighbor_system, semi)

                    # Neighbor-fluid interact
                    dv_neighbor = zero(v_neighbor)
                    TrixiParticles.interact!(dv_neighbor, v_neighbor, u_neighbor, v, u,
                                             neighbor_system, fluid_system, semi)

                    if neighbor_system isa WeaklyCompressibleSPHSystem
                        # If both are fluids, neighbor-neighbor interaction is necessary
                        # for energy preservation.
                        TrixiParticles.interact!(dv_neighbor, v_neighbor, u_neighbor,
                                                 v_neighbor, u_neighbor,
                                                 neighbor_system, neighbor_system, semi)
                    end

                    # Quantities needed to test energy conservation
                    function drho(dv, ::ContinuityDensity, system, neighbor_system,
                                  v, u, v_neighbor, u_neighbor, particle)
                        return dv[end, particle]
                    end

                    function drho(dv, ::SummationDensity, system, neighbor_system,
                                  v, u, v_neighbor, u_neighbor, particle)
                        sum1 = sum(neighbor -> drho_particle(system, system,
                                                             v, u, v, u,
                                                             particle, neighbor),
                                   TrixiParticles.eachparticle(system))

                        sum2 = sum(neighbor -> drho_particle(system, neighbor_system,
                                                             v, u, v_neighbor, u_neighbor,
                                                             particle, neighbor),
                                   TrixiParticles.eachparticle(neighbor_system))

                        return sum1 + sum2
                    end

                    function drho(dv, ::PressureZeroing, system, neighbor_system,
                                  v, u, v_neighbor, u_neighbor, particle)
                        return 0.0
                    end

                    function drho(dv, ::PressureMirroring, system, neighbor_system,
                                  v, u, v_neighbor, u_neighbor, particle)
                        return 0.0
                    end

                    # Derivative of the density summation. This is a slightly different
                    # formulation of the continuity equation.
                    function drho_particle(system, neighbor_system,
                                           v, u, v_neighbor, u_neighbor, particle, neighbor)
                        m_b = TrixiParticles.hydrodynamic_mass(neighbor_system, neighbor)
                        v_diff = TrixiParticles.current_velocity(v, system, particle) -
                                 TrixiParticles.current_velocity(v_neighbor,
                                                                 neighbor_system, neighbor)

                        pos_diff = TrixiParticles.current_coords(u, system, particle) -
                                   TrixiParticles.current_coords(u_neighbor,
                                                                 neighbor_system, neighbor)
                        distance = norm(pos_diff)

                        # Only consider particles with a distance > 0
                        distance < sqrt(eps()) && return 0.0

                        grad_kernel = TrixiParticles.smoothing_kernel_grad(fluid_system,
                                                                           pos_diff,
                                                                           distance,
                                                                           particle)

                        return m_b * dot(v_diff, grad_kernel)
                    end

                    # m_a (v_a ⋅ dv_a + dte_a),
                    # where `te` is the thermal energy, called `u` in the Price paper.
                    function deriv_energy(dv, v, u, v_neighbor, u_neighbor,
                                          system, neighbor_system, particle)
                        m_a = TrixiParticles.hydrodynamic_mass(system, particle)
                        p_a = TrixiParticles.current_pressure(v, system, particle)
                        rho_a = TrixiParticles.current_density(v, system, particle)

                        if system isa WeaklyCompressibleSPHSystem
                            dc = system.density_calculator
                        else
                            dc = system.boundary_model.density_calculator
                        end
                        d_rho = drho(dv, dc, system, neighbor_system,
                                     v, u, v_neighbor, u_neighbor, particle)

                        dte_a = p_a / rho_a^2 * d_rho
                        v_a = TrixiParticles.current_velocity(v, system, particle)
                        dv_a = TrixiParticles.current_velocity(dv, system, particle)

                        return m_a * (dot(v_a, dv_a) + dte_a)
                    end

                    # ∑ m_a (v_a ⋅ dv_a + dte_a)
                    deriv_total_energy = sum(particle -> deriv_energy(dv, v, u,
                                                                      v_neighbor,
                                                                      u_neighbor,
                                                                      fluid_system,
                                                                      neighbor_system,
                                                                      particle),
                                             TrixiParticles.eachparticle(fluid_system))

                    deriv_total_energy += sum(particle -> deriv_energy(dv_neighbor,
                                                                       v_neighbor,
                                                                       u_neighbor, v, u,
                                                                       neighbor_system,
                                                                       fluid_system,
                                                                       particle),
                                              TrixiParticles.eachparticle(neighbor_system))

                    @test isapprox(deriv_total_energy, 0.0, atol=6e-15)
                end
            end
        end
    end
end
