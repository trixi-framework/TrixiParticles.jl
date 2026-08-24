using LinearAlgebra: dot
using TrixiParticles

"""
    potential_energy(omega)

Returns a custom quantity for the potential energy of the oscillating-drop
central force field `-omega^2 * x`. This is `-sum(m_i * phi_i)` from the paper with
`phi_i = -omega^2 * norm(x_i)^2 / 2`.
"""
function potential_energy(omega)
    function f(system::WeaklyCompressibleSPHSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
        u = TrixiParticles.wrap_u(u_ode, system, semi)

        energy = zero(eltype(system))
        for particle in TrixiParticles.each_active_particle(system)
            coords = TrixiParticles.current_coords(u, system, particle)
            energy += system.mass[particle] * omega^2 * dot(coords, coords) / 2
        end

        return energy
    end

    function f(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        return nothing
    end

    return f
end

"""
    compressible_energy

Custom quantity for the compressible energy associated with the barotropic equation of state.
The specific energy is the primitive of `p / rho^2` and is normalized to vanish at `rho0`.
For the linear Cole EOS used in this validation, this reduces to
`c^2 * (log(rho / rho0) + rho0 / rho - 1)`.
"""
function compressible_energy(system::WeaklyCompressibleSPHSystem,
                             dv_ode, du_ode, v_ode, u_ode, semi, t)
    (; sound_speed, exponent, reference_density,
     background_pressure) = system.state_equation

    v = TrixiParticles.wrap_v(v_ode, system, semi)

    energy = zero(eltype(system))
    for particle in TrixiParticles.each_active_particle(system)
        rho = TrixiParticles.current_density(v, system, particle)

        pressure_energy = cole_pressure_energy(sound_speed, exponent,
                                               rho / reference_density)
        background_energy = background_pressure * (inv(reference_density) - inv(rho))
        specific_compressible_energy = pressure_energy + background_energy

        energy += system.mass[particle] * specific_compressible_energy
    end

    return energy
end

compressible_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

function cole_pressure_energy(sound_speed, exponent, rho_ratio)
    if isapprox(exponent, 1)
        return sound_speed^2 * (log(rho_ratio) + inv(rho_ratio) - one(rho_ratio))
    end

    return sound_speed^2 / exponent *
           (rho_ratio^(exponent - 1) / (exponent - 1) + inv(rho_ratio) -
            exponent / (exponent - 1))
end

"""
    delta_sph_diffusive_power

Custom quantity for the diffusive power `P_δ = dE_C / dt |_δ` associated with
the density-diffusion term.
For the delta-SPH term in the paper this is non-positive at leading order; the dissipated
energy is therefore `Q_δ = -∫ P_δ dt`.
"""
function delta_sph_diffusive_power(system::WeaklyCompressibleSPHSystem,
                                   dv_ode, du_ode, v_ode, u_ode, semi, t)
    density_diffusion = TrixiParticles.density_diffusion(system)
    isnothing(density_diffusion) && return zero(eltype(system))

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    density_rate = delta_sph_density_rate(system, density_diffusion, v, u, semi)

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
                                density_diffusion, v, u, semi)
    coordinates = TrixiParticles.current_coordinates(u, system)

    density_rate = zeros(eltype(system), TrixiParticles.nparticles(system))
    almostzero = eps(TrixiParticles.initial_smoothing_length(system)^2)

    TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                          semi) do particle, neighbor, pos_diff, distance
        distance^2 < almostzero && return nothing

        rho_particle = TrixiParticles.current_density(v, system, particle)
        rho_neighbor = TrixiParticles.current_density(v, system, neighbor)
        mass_neighbor = TrixiParticles.hydrodynamic_mass(system, neighbor)
        grad_kernel = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                           particle)

        drho_particle = TrixiParticles.add_density_diffusion(zero(rho_particle),
                                                             density_diffusion, system,
                                                             particle, neighbor, pos_diff,
                                                             distance,
                                                             mass_neighbor, rho_particle,
                                                             rho_neighbor, grad_kernel)
        density_rate[particle] += drho_particle[]

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

Custom quantity to compute `Q_delta = -∫ P_δ dt`, integrated with
the trapezoidal rule over postprocessing times.
Note that the interval used for the `PostprocessCallback` determines the step size
and therefore the accuracy of the integration, not just the frequency of output.
"""
function (quantity::DeltaSPHHeat)(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    power = delta_sph_diffusive_power(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    isnothing(power) && return nothing

    dt = t - quantity.previous_time
    quantity.previous_time = t

    if !quantity.initialized
        quantity.q_delta = 0.0
        quantity.initialized = true

        return quantity.q_delta
    end

    if iszero(dt)
        return quantity.q_delta
    end

    quantity.q_delta -= (quantity.previous_power + power) * dt / 2

    quantity.previous_power = power

    return quantity.q_delta
end
