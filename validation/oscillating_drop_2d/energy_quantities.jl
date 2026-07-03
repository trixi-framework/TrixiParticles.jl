using LinearAlgebra: dot
using TrixiParticles

"""
    potential_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t)

Potential energy of the oscillating-drop central force field `-OMEGA^2 * x`.
This is `-sum(m_i * phi_i)` from the paper with
`phi_i = -OMEGA^2 * norm(x_i)^2 / 2`.
"""
function potential_energy(system::WeaklyCompressibleSPHSystem,
                          dv_ode, du_ode, v_ode, u_ode, semi, t)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    energy = zero(eltype(system))
    for particle in TrixiParticles.each_active_particle(system)
        coords = TrixiParticles.current_coords(u, system, particle)
        energy += system.mass[particle] * OMEGA^2 * dot(coords, coords) / 2
    end

    return energy
end

potential_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

"""
    compressible_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t)

Compressible energy associated with the barotropic equation of state. The specific
energy is the primitive of `p / rho^2` and is normalized to vanish at `rho0`.
For the linear Cole EOS used in this validation, this reduces to
`c0^2 * (log(rho / rho0) + rho0 / rho - 1)`.
"""
function compressible_energy(system::WeaklyCompressibleSPHSystem,
                             dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    state_equation = system.state_equation

    energy = zero(eltype(system))
    for particle in TrixiParticles.each_active_particle(system)
        rho = TrixiParticles.current_density(v, system, particle)
        energy += system.mass[particle] * specific_compressible_energy(state_equation,
                                                                       rho)
    end

    return energy
end

compressible_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

function specific_compressible_energy(state_equation::Union{StateEquationCole,
                                                            StateEquationAdaptiveCole},
                                      rho)
    (; exponent, reference_density, background_pressure) = state_equation

    c0 = TrixiParticles.sound_speed(state_equation)
    rho_ratio = rho / reference_density

    pressure_energy = cole_pressure_energy(c0, exponent, rho_ratio)
    background_energy = background_pressure * (inv(reference_density) - inv(rho))

    return pressure_energy + background_energy
end

function specific_compressible_energy(state_equation::StateEquationIdealGas, rho)
    (; gamma, reference_density, background_pressure) = state_equation

    c0 = TrixiParticles.sound_speed(state_equation)
    rho_ratio = rho / reference_density

    pressure_energy = c0^2 / gamma *
                      (log(rho_ratio) + inv(rho_ratio) - one(rho_ratio))
    background_energy = background_pressure * (inv(reference_density) - inv(rho))

    return pressure_energy + background_energy
end

function cole_pressure_energy(sound_speed, exponent, rho_ratio)
    if isapprox(exponent, one(exponent))
        return sound_speed^2 *
               (log(rho_ratio) + inv(rho_ratio) - one(rho_ratio))
    end

    return sound_speed^2 / exponent *
           (rho_ratio^(exponent - 1) / (exponent - 1) + inv(rho_ratio) -
            exponent / (exponent - 1))
end

"""
    delta_sph_diffusive_power(system, dv_ode, du_ode, v_ode, u_ode, semi, t)

Power `P_delta = dE_C / dt |_delta` associated with the density-diffusion term.
For the delta-SPH term in the paper this is non-positive at leading order; the dissipated
energy is therefore `Q_delta = -int(P_delta) dt`.
"""
function delta_sph_diffusive_power(system::WeaklyCompressibleSPHSystem,
                                   dv_ode, du_ode, v_ode, u_ode, semi, t)
    density_diffusion = TrixiParticles.density_diffusion(system)
    density_diffusion === nothing && return zero(eltype(system))

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    density_rate = delta_sph_density_rate(system, density_diffusion, v_ode, u_ode, semi)

    power = zero(eltype(system))
    for particle in TrixiParticles.each_active_particle(system)
        rho = TrixiParticles.current_density(v, system, particle)
        pressure = TrixiParticles.current_pressure(v, system, particle)
        volume = system.mass[particle] / rho

        power += volume * pressure / rho * density_rate[particle]
    end

    return power
end

delta_sph_diffusive_power(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

function delta_sph_density_rate(system::WeaklyCompressibleSPHSystem,
                                density_diffusion, v_ode, u_ode, semi)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = TrixiParticles.current_coordinates(u, system)

    density_rate = zeros(eltype(system), TrixiParticles.nparticles(system))
    sound_speed = TrixiParticles.system_sound_speed(system)
    almost_zero_distance2 = eps(TrixiParticles.initial_smoothing_length(system)^2)

    TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates, semi;
                                          points=TrixiParticles.each_active_particle(system)) do particle,
                                                                                                 neighbor,
                                                                                                 pos_diff,
                                                                                                 distance
        distance^2 < almost_zero_distance2 && return nothing

        rho_particle = TrixiParticles.current_density(v, system, particle)
        rho_neighbor = TrixiParticles.current_density(v, system, neighbor)
        mass_neighbor = TrixiParticles.hydrodynamic_mass(system, neighbor)
        volume_neighbor = mass_neighbor / rho_neighbor
        grad_kernel = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                           particle)
        psi = TrixiParticles.density_diffusion_psi(density_diffusion,
                                                   rho_particle, rho_neighbor,
                                                   pos_diff, distance, system,
                                                   particle, neighbor)

        smoothing_length = (TrixiParticles.smoothing_length(system, particle) +
                            TrixiParticles.smoothing_length(system, neighbor)) / 2
        density_rate[particle] += density_diffusion.delta * smoothing_length *
                                  sound_speed * volume_neighbor * dot(psi, grad_kernel)

        return nothing
    end

    return density_rate
end

mutable struct DeltaSPHHeat
    previous_time::Float64
    previous_power::Float64
    q_delta::Float64
    initialized::Bool
end

DeltaSPHHeat() = DeltaSPHHeat(0.0, 0.0, 0.0, false)

"""
    DeltaSPHHeat()

Stateful postprocessing quantity for `Q_delta = -int(P_delta) dt`, integrated with
the trapezoidal rule over postprocessing times.
"""
function (quantity::DeltaSPHHeat)(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    power = delta_sph_diffusive_power(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    power === nothing && return nothing

    time = Float64(t)
    power = Float64(power)

    if !quantity.initialized
        quantity.previous_time = time
        quantity.previous_power = power
        quantity.q_delta = 0.0
        quantity.initialized = true

        return quantity.q_delta
    end

    dt = time - quantity.previous_time
    if iszero(dt)
        return quantity.q_delta
    end

    quantity.q_delta -= (quantity.previous_power + power) * dt / 2
    quantity.previous_time = time
    quantity.previous_power = power

    return quantity.q_delta
end
