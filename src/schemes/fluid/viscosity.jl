
@doc raw"""
    ViscosityCarreauYasuda(; nu0, nu_inf, lambda, a, n, epsilon=0.01)

Non-Newtonian Carreau–Yasuda viscosity model.

The kinematic viscosity is modeled as

```math
\nu(\dot\gamma) = \nu_\infty + (\nu_0 - \nu_\infty)
\left[ 1 + (\lambda \dot\gamma)^a \right]^{\frac{n-1}{a}}.
```
where

- ``\nu_0``: zero-shear kinematic viscosity,
- ``\nu_\infty``: infinite-shear kinematic viscosity,
- ``\lambda``: time constant,
- ``a``: Yasuda parameter,
- ``n``: power-law index (``n < 1`` for shear-thinning, ``n > 1`` for shear-thickening),
- ``\dot\gamma``: shear-rate magnitude.

In this implementation the shear-rate magnitude is approximated per particle pair as
``\dot\gamma \approx \frac{\lVert \mathbf{v}_{ab} \rVert}{\lVert \mathbf{r}_{ab} \rVert + \epsilon}``,
with ``\mathbf{v}_{ab}`` the relative velocity, ``\mathbf{r}_{ab}`` the position difference,
and ``\epsilon`` a small regularization parameter.

All viscosities here are kinematic viscosities (m²/s); dynamic viscosity is obtained internally
via ``\eta = \rho \nu``. A Newtonian fluid is recovered for ``n = 1`` and
``\nu_0 = \nu_\infty``.
"""


struct ViscosityCarreauYasuda{ELTYPE}
    nu0    :: ELTYPE  # zero-shear kinematic viscosity
    nu_inf :: ELTYPE  # infinite-shear kinematic viscosity
    lambda :: ELTYPE  # time constant
    a      :: ELTYPE  # Yasuda parameter
    n      :: ELTYPE  # power-law index
    epsilon:: ELTYPE  # regularization
end

ViscosityCarreauYasuda(; nu0, nu_inf, lambda, a, n, epsilon=0.01) =
    ViscosityCarreauYasuda(nu0, nu_inf, lambda, a, n, epsilon)

@inline function carreau_yasuda_nu(viscosity::ViscosityCarreauYasuda, gamma_dot)
    (; nu0, nu_inf, lambda, a, n) = viscosity
    return nu_inf + (nu0 - nu_inf) *
           (1 + (lambda * gamma_dot)^a)^((n - 1) / a)
end

@propagate_inbounds function (viscosity::ViscosityCarreauYasuda)(
    particle_system,
    neighbor_system,
    v_particle_system,
    v_neighbor_system,
    particle, neighbor,
    pos_diff, distance,
    sound_speed,
    m_a, m_b,
    rho_a, rho_b,
    grad_kernel,
)
    epsilon = viscosity.epsilon

    smoothing_length_particle = smoothing_length(particle_system, particle)
    smoothing_length_neighbor = smoothing_length(particle_system, neighbor)
    smoothing_length_average = (smoothing_length_particle + smoothing_length_neighbor) / 2

    v_a = viscous_velocity(v_particle_system, particle_system, particle)
    v_b = viscous_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b

    gamma_dot = norm(v_diff) / (distance + epsilon)

    nu_eff = carreau_yasuda_nu(viscosity, gamma_dot)
    nu_a = nu_eff
    nu_b = nu_eff

    return adami_viscosity_force(smoothing_length_average, pos_diff, distance, grad_kernel,
                                 m_a, m_b, rho_a, rho_b, v_diff, nu_a, nu_b, epsilon)
end

function kinematic_viscosity(system, viscosity::ViscosityCarreauYasuda,
                             smoothing_length, sound_speed)
    return viscosity.nu0
end