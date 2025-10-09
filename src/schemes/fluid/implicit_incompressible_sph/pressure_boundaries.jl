function calculate_diagonal_elements_and_predicted_density!(system::WallBoundarySystem{<:BoundaryModelDummyParticles{PressureBoundaries}},
                                                            v, u, v_ode, u_ode, semi)
    (; boundary_model) = system
    (; density_calculator) = boundary_model
    (; a_ii, predicted_density, density, time_step) = boundary_model.cache

    set_zero!(a_ii)
    predicted_density .= density

    # Calculation the diagonal elements (a_ii-values) according to eq. 12 in Ihmsen et al. (2013)
    foreach_system(semi) do neighbor_system
        calculate_diagonal_elements_and_predicted_density(a_ii, predicted_density, system, density_calculator,
                                     neighbor_system, v, u, v_ode, u_ode, semi, time_step)
    end
end

function calculate_diagonal_elements_and_predicted_density(a_ii, predicted_density, system, ::PressureBoundaries,
                                                            neighbor_system::AbstractFluidSystem, v, u, v_ode,
                                                            u_ode, semi, time_step)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords,
                           semi; points=eachparticle(system)) do particle, neighbor,
                                                                pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

        # Compute d_ji
        # According to eq. 9 in Ihmsen et al. (2013).
        # Note that we compute d_ji and not d_ij. We can use the antisymmetry
        # of the kernel gradient and just flip the sign of W_ij to obtain W_ji.
        d_ji_ = calculate_d_ji(system, neighbor_system, particle, -grad_kernel, time_step)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        # According to eq. 12 in Ihmsen et al. (2013)
        a_ii[particle] -= m_b * dot(d_ji_, grad_kernel)

        # Calculate the predicted velocity differences
        advection_velocity_diff = predicted_velocity(system, particle) -
                                    predicted_velocity(neighbor_system, neighbor)

        # Compute \rho_adv in eq. 4 in Ihmsen et al. (2013)
        predicted_density[particle] += time_step * m_b *
                                        dot(advection_velocity_diff, grad_kernel)
    end
end

function calculate_diagonal_elements_and_predicted_density(a_ii, predicted_density, system, ::PressureBoundaries,
                                                        neighbor_system::AbstractBoundarySystem, v, u, v_ode,
                                                        u_ode, semi, time_step)
    # Calculation of the predicted density
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords,
                            neighbor_system_coords, semi,
                            points=eachparticle(system)) do particle, neighbor,
                                                            pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        # Calculate the predicted velocity differences
        advection_velocity_diff = predicted_velocity(system, particle) -
                                    predicted_velocity(neighbor_system, neighbor)

        # Compute \rho_adv in eq. 4 in Ihmsen et al. (2013)
        predicted_density[particle] += time_step * m_b *
                                        dot(advection_velocity_diff, grad_kernel)
    end
end

function initialize_pressure(system::AbstractBoundarySystem, semi)
    (; boundary_model) = system
    (; density_calculator) = boundary_model

    return initialize_pressure(system, boundary_model, density_calculator, semi)
end

function initialize_pressure(system, boundary_model, density_calculator, semi)
    return system
end

function initialize_pressure(system, boundary_model, ::PressureBoundaries, semi)
    (; pressure) = boundary_model

    # Set initial pressure (p_0) to a half of the current pressure value
    @threaded semi for particle in eachparticle(system)
        pressure[particle] = pressure[particle] / 2
    end
end

function calculate_sum_d_ij_pj(system::AbstractBoundarySystem, u, u_ode, semi)
    return system
end

function calculate_sum_term_values(system::AbstractBoundarySystem, u, u_ode, semi)
    (; boundary_model) = system
    (; density_calculator) = boundary_model
    return calculate_sum_term_values(system, boundary_model, density_calculator, u, u_ode,
                                     semi)
end

function calculate_sum_term_values(system, boundary_model, density_calculator, u, u_ode,
                                   semi)
    return system
end

function calculate_sum_term_values(system, boundary_model, ::PressureBoundaries, u, u_ode,
                                   semi)
    (; sum_term, time_step) = boundary_model.cache

    # Calculate the large sum in eq. 13 of Ihmsen et al. (2013) for each particle (as `sum_term`)
    set_zero!(sum_term)

    foreach_system(semi) do neighbor_system
        calculate_sum_term!(sum_term, system, neighbor_system, u, u_ode, semi, time_step)
    end
end

function pressure_update(system::AbstractBoundarySystem, u, u_ode, semi)
    (; boundary_model) = system
    (; density_calculator) = boundary_model
    return pressure_update(system, boundary_model, density_calculator, u, u_ode, semi)
end

function pressure_update(system, boundary_model, density_calculator, u, u_ode, semi)
    return 0.0
end

function pressure_update(system, boundary_model, ::PressureBoundaries, u, u_ode, semi)
    (; reference_density, a_ii, sum_term, omega, density_error) = boundary_model.cache
    (; pressure) = boundary_model

    # Update the pressure values
    relative_density_error = zero(eltype(system))

    @threaded semi for particle in eachparticle(system)
        # Removing instabilities by avoiding to divide by very low values of `a_ii`.
        # This is not mentioned in the paper but done in SPlisHSPlasH as well.
        if abs(a_ii[particle]) > 1.0e-9
            pressure[particle] = max((1-omega) * pressure[particle] +
                                     omega / a_ii[particle] *
                                     (iisph_source_term(system, particle) -
                                      sum_term[particle]), 0)
        else
            pressure[particle] = zero(pressure[particle])
        end
        # Calculate the average density error for the termination condition
        if (pressure[particle] != 0.0)
            new_density = a_ii[particle] * pressure[particle] + sum_term[particle] -
                          iisph_source_term(system, particle) +
                          reference_density
            density_error[particle] = (new_density - reference_density)
        end
    end
    relative_density_error = sum(density_error) / reference_density

    return relative_density_error
end

@propagate_inbounds function predicted_velocity(system::AbstractBoundarySystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function d_ii(system::AbstractBoundarySystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

function calculate_d_ji(system::AbstractBoundarySystem, neighbor_system, particle_i,
                        grad_kernel, time_step)
    (; boundary_model) = system
    return -time_step^2 * hydrodynamic_mass(system, particle_i) /
           boundary_model.cache.density[particle_i]^2 * grad_kernel
end

function calculate_sum_term(system::AbstractBoundarySystem, neighbor_system,
                            particle, neighbor, grad_kernel, time_step)
    (; boundary_model) = system
    (; density_calculator) = boundary_model
    return calculate_sum_term(system, boundary_model, density_calculator, neighbor_system,
                              particle, neighbor, grad_kernel, time_step)
end

function calculate_sum_term(system, boundary_model, density_calculator, neighbor_system,
                            particle, neighbor, grad_kernel, time_step)
    return system
end

# Calculate the large sum in eq. 13 of Ihmsen et al. (2013) for each particle (as `sum_term`)
function calculate_sum_term(system, boundary_model, ::PressureBoundaries,
                            neighbor_system::AbstractFluidSystem,
                            particle, neighbor, grad_kernel, time_step)
    pressure_system = boundary_model.pressure
    pressure_neighbor = neighbor_system.pressure

    m_j = hydrodynamic_mass(neighbor_system, neighbor)
    d_jj = d_ii(neighbor_system, neighbor)
    p_i = pressure_system[particle]
    p_j = pressure_neighbor[neighbor]
    sum_djk_pk = sum_dij_pj(neighbor_system, neighbor)
    d_ji = calculate_d_ji(system, neighbor_system, particle, -grad_kernel, time_step)

    # Equation 13 of Ihmsen et al. (2013):
    # m_j * (\sum_k d_ik * p_k - d_jj * p_j - \sum_{k != i} d_jk * p_k) * grad_W_ij
    return m_j * dot(- d_jj * p_j - (sum_djk_pk - d_ji * p_i), grad_kernel)
end

function calculate_sum_term(system, boundary_model, ::PressureBoundaries,
                            neighbor_system::AbstractBoundarySystem,
                            particle, neighbor, grad_kernel, time_step)
    return 0.0
end

function iisph_source_term(system::AbstractBoundarySystem, particle)
    (; boundary_model) = system
    (; density_calculator) = boundary_model
    return iisph_source_term(system, boundary_model, density_calculator, particle)
end

function iisph_source_term(system, boundary_model, density_calculator, particle)
    return system
end

function iisph_source_term(system, boundary_model, ::PressureBoundaries, particle)
    (; reference_density, predicted_density) = boundary_model.cache
    return reference_density - predicted_density[particle]
end
