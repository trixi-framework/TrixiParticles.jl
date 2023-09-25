@doc raw"""
    StateEquationIdealGas(sound_speed, reference_density, reference_pressure;
                          background_pressure=0.0)

Equation of state to describe a linear relationship between pressure and density as
```math
p = c^2 (\rho - \rho_0) + p_0 - p_{\text{background}},
```
where ``c`` denotes the speed of sound, ``\rho`` the density, ``p`` the pressure,
``\rho_0`` an arbitrary reference density, ``p_0`` the pressure
at the reference density, and ``p_{\text{background}}`` the atmospheric or
background pressure (to be used with free surfaces).
"""
struct StateEquationIdealGas{ELTYPE}
    sound_speed         :: ELTYPE
    reference_density   :: ELTYPE
    reference_pressure  :: ELTYPE
    background_pressure :: ELTYPE

    function StateEquationIdealGas(sound_speed, reference_density, reference_pressure;
                                   background_pressure=0.0)
        new{typeof(sound_speed)}(sound_speed, reference_density, reference_pressure,
                                 background_pressure)
    end
end

function (state_equation::StateEquationIdealGas)(density)
    (; sound_speed, reference_density, reference_pressure, background_pressure) = state_equation

    return sound_speed^2 * (density - reference_density) + reference_pressure -
           background_pressure
end

@doc raw"""
    StateEquationCole(sound_speed, gamma, reference_density, reference_pressure;
                      background_pressure=0.0)

Equation of state to describe the relationship between pressure and density
of water up to high pressures by Cole (Cole 1948, pp. 39 and 43).
The pressure ``p`` is calculated as
```math
p = \frac{\rho_0 c^2}{\gamma} \left(\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right) + p_0 - p_{\text{background}},
```
where ``c`` denotes the speed of sound, ``\rho`` the density,
``\rho_0`` an arbitrary reference density, ``p_0`` the pressure
at the reference density, and ``p_{\text{background}}`` the atmospheric or
background pressure (to be used with free surfaces).

For water, an average value of ``\gamma = 7.15`` is usually used (Cole 1948, p. 39).

## References:
- Robert H. Cole. "Underwater Explosions". Princeton University Press, 1948.
"""
struct StateEquationCole{ELTYPE}
    sound_speed         :: ELTYPE
    gamma               :: ELTYPE
    reference_density   :: ELTYPE
    reference_pressure  :: ELTYPE
    background_pressure :: ELTYPE

    function StateEquationCole(sound_speed, gamma, reference_density, reference_pressure;
                               background_pressure=0.0)
        new{typeof(sound_speed)}(sound_speed, gamma, reference_density, reference_pressure,
                                 background_pressure)
    end
end

function (state_equation::StateEquationCole)(density)
    (; sound_speed, gamma, reference_density, reference_pressure, background_pressure) = state_equation

    return reference_density * sound_speed^2 / gamma *
           ((density / reference_density)^gamma - 1) +
           reference_pressure - background_pressure
end

function inverse_state_equation(state_equation::StateEquationCole, pressure)
    (; sound_speed, gamma, reference_density, reference_pressure, background_pressure) = state_equation

    tmp = gamma * (pressure + background_pressure - reference_pressure) /
          (reference_density * sound_speed^2) + 1
    density = reference_density * tmp^(1 / gamma)

    return density
end
