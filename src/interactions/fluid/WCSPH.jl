@inline function interaction!(particle_container::FluidParticleContainer{<:WCSPH},
                              neighbor_container,
                              dv, v_particle_container, v_neighbor_container,
                              particle, neighbor, pos_diff, distance)
    @unpack density_calculator = particle_container

    calc_dv!(dv, v_particle_container, v_neighbor_container,
             particle, neighbor, pos_diff, distance,
             particle_container, neighbor_container)

    continuity_equation!(dv, density_calculator,
                         v_particle_container, v_neighbor_container,
                         particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)
end

# Fluid-fluid interaction
@inline function calc_dv!(dv, v_particle_container, v_neighbor_container,
                          particle, neighbor, pos_diff, distance,
                          particle_container::FluidParticleContainer{<:WCSPH},
                          neighbor_container::FluidParticleContainer{<:WCSPH})
    @unpack SPH_scheme, smoothing_length, viscosity = particle_container
    @unpack state_equation = SPH_scheme

    rho_a = get_particle_density(particle, v_particle_container, particle_container)
    rho_b = get_particle_density(neighbor, v_neighbor_container, neighbor_container)

    # Viscosity
    v_diff = get_particle_vel(particle, v_particle_container, particle_container) -
             get_particle_vel(neighbor, v_neighbor_container, neighbor_container)
    rho_mean = (rho_a + rho_b) / 2
    pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                      distance, rho_mean, smoothing_length)

    grad_kernel = smoothing_kernel_grad(particle_container, pos_diff, distance)
    m_b = neighbor_container.mass[neighbor]

    dv_pressure = -m_b *
                  (particle_container.pressure[particle] / rho_a^2 +
                   neighbor_container.pressure[neighbor] / rho_b^2) * grad_kernel
    dv_viscosity = -m_b * pi_ab * grad_kernel

    for i in 1:ndims(particle_container)
        dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
    end

    return dv
end

# Fluid-boundary and fluid-solid interaction
@inline function calc_dv!(dv, v_particle_container, v_neighbor_container, particle,
                          neighbor, pos_diff, distance,
                          particle_container::FluidParticleContainer{<:WCSPH},
                          neighbor_container)
    @unpack SPH_scheme, smoothing_length = particle_container
    @unpack state_equation = SPH_scheme
    @unpack boundary_model = neighbor_container

    rho_a = get_particle_density(particle, v_particle_container,
                                 particle_container)

    # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
    # corresponding to the rest density of the fluid and not the material density.
    m_b = get_hydrodynamic_mass(neighbor, neighbor_container)

    grad_kernel = smoothing_kernel_grad(particle_container, pos_diff, distance)

    dv_boundary = boundary_particle_impact(particle, neighbor,
                                           v_particle_container,
                                           v_neighbor_container,
                                           particle_container,
                                           neighbor_container,
                                           pos_diff, distance, m_b)

    dv_viscosity = calc_viscosity(boundary_model, particle_container, neighbor_container,
                                  v_particle_container, v_neighbor_container,
                                  particle, neighbor, pos_diff, distance, rho_a,
                                  grad_kernel, state_equation.sound_speed, smoothing_length,
                                  m_b)

    for i in 1:ndims(particle_container)
        dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
    end

    return dv
end

@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::FluidParticleContainer,
                                      neighbor_container)
    mass = get_hydrodynamic_mass(neighbor, neighbor_container)

    vdiff = get_particle_vel(particle, v_particle_container, particle_container) -
            get_particle_vel(neighbor, v_neighbor_container, neighbor_container)

    NDIMS = ndims(particle_container)

    dv[NDIMS + 1, particle] += sum(mass * vdiff .*
                                   smoothing_kernel_grad(particle_container, pos_diff,
                                                         distance))

    return dv
end

@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container, neighbor_container)
    return dv
end

@inline function calc_viscosity(model,
                                particle_container, neighbor_container,
                                v_particle_container, v_neighbor_container,
                                particle, neighbor, pos_diff, distance, rho_a,
                                grad_kernel, sound_speed, smoothing_length, m_b)
    return SVector(ntuple(_ -> 0.0, Val(ndims(particle_container))))
end

@inline function calc_viscosity(model::BoundaryModelMonaghanKajtar,
                                particle_container, neighbor_container,
                                v_particle_container, v_neighbor_container,
                                particle, neighbor, pos_diff, distance, rho_a,
                                grad_kernel, sound_speed, smoothing_length, m_b)
    @unpack viscosity, smoothing_length = particle_container
    v_diff = get_particle_vel(particle, v_particle_container, particle_container) -
             get_particle_vel(neighbor, v_neighbor_container, neighbor_container)

    pi_ab = viscosity(sound_speed, v_diff, pos_diff,
                      distance, rho_a, smoothing_length)

    return -m_b * pi_ab * grad_kernel
end
