# Set up a single semidiscretized fluid system with the given correction.
# Re-extract the system from the ODE, since `semidiscretize` replaces systems by
# runtime copies.
function correction_setup(correction=nothing; n=9, perturbation=false,
                          density_calculator=ContinuityDensity(), edac=false,
                          pressure_acceleration=:default,
                          velocity=(pos -> SVector(pos[1], pos[2])), buffer_size=nothing,
                          neighborhood_search=GridNeighborhoodSearch{2}())
    particle_spacing = 1.0 / n
    smoothing_length = 2.0 * particle_spacing
    smoothing_kernel = WendlandC6Kernel{2}()
    fluid = RectangularShape(particle_spacing, (n, n), (0.0, 0.0);
                             density=1000.0, velocity,
                             coordinates_perturbation=perturbation ? 0.1 : nothing)

    if edac
        if pressure_acceleration === :default
            system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                 smoothing_length, sound_speed=10.0,
                                                 density_calculator, correction,
                                                 buffer_size)
        else
            system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                 smoothing_length, sound_speed=10.0,
                                                 density_calculator, correction,
                                                 pressure_acceleration, buffer_size)
        end
    else
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=1000.0, exponent=1)
        if pressure_acceleration === :default
            system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                 smoothing_length, density_calculator,
                                                 state_equation, correction, buffer_size)
        else
            system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                 smoothing_length, density_calculator,
                                                 state_equation, correction,
                                                 pressure_acceleration, buffer_size)
        end
    end

    semi = Semidiscretization(system; neighborhood_search,
                              parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v_ode = Array(ode.u0.x[1])
    u_ode = Array(ode.u0.x[2])
    semi = ode.p.semi
    system = first(semi.systems)

    return (; system, semi, v_ode, u_ode, particle_spacing)
end

# Poison all correction caches with `NaN` to verify that the next update recomputes them.
function fill_correction_cache!(system, value)
    for name in (:kernel_correction_coefficient, :dw_gamma, :correction_matrix)
        hasproperty(system.cache, name) || continue
        fill!(getproperty(system.cache, name), value)
    end
    return system
end

# Recompute all correction caches by running a full `update_systems_and_nhs` pass.
function update_correction!(setup)
    (; system, semi, v_ode, u_ode) = setup
    fill_correction_cache!(system, NaN)
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    return setup
end

# Recompute, in a naive loop, the zeroth and first gradient moments and the direct and
# difference kernel-gradient interpolations of a scalar field. These reference values are
# compared against the corrections computed by TrixiParticles.
function correction_moments(setup; field=(pos -> 1.0))
    (; system, semi, v_ode, u_ode) = setup
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    values = [field(SVector{2}(view(coordinates, :, particle)))
              for particle in TrixiParticles.eachparticle(system)]
    n_particles = TrixiParticles.nparticles(system)

    zeroth_gradient_moment = zeros(2, n_particles)
    first_gradient_moment = zeros(2, 2, n_particles)
    direct_gradient = zeros(2, n_particles)
    difference_gradient = zeros(2, n_particles)

    GC.@preserve v_ode u_ode begin
        TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                              semi) do particle, neighbor, pos_diff,
                                                       distance
            pos_diff_ = SVector(pos_diff)
            volume = TrixiParticles.hydrodynamic_mass(system, neighbor) /
                     TrixiParticles.current_density(v, system, neighbor)
            gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff_, distance,
                                                            particle)
            neighbor_offset = -pos_diff_

            for i in 1:2
                zeroth_gradient_moment[i, particle] += volume * gradient[i]
                direct_gradient[i, particle] += volume * values[neighbor] * gradient[i]
                difference_gradient[i,
                                    particle] += volume *
                                                 (values[neighbor] - values[particle]) *
                                                 gradient[i]
                for j in 1:2
                    first_gradient_moment[i, j,
                                          particle] += volume * gradient[i] *
                                                       neighbor_offset[j]
                end
            end
        end
    end

    return (; zeroth_gradient_moment, first_gradient_moment, direct_gradient,
            difference_gradient)
end

# Find the particle closest to the lower-left corner, where the correction is
# least accurate due to missing neighbors.
function corner_particle(system)
    coordinates = TrixiParticles.initial_coordinates(system)
    return argmin(eachindex(axes(coordinates, 2))) do particle
        coordinates[1, particle] + coordinates[2, particle]
    end
end
