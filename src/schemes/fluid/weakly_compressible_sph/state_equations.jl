@doc raw"""
    StateEquationIdealGas(sound_speed, reference_density, reference_pressure;
                          background_pressure=0.0, clip_negative_pressure=false)

Equation of state to describe a linear relationship between pressure and density as
```math
p = c^2 (\rho - \rho_0) + p_0 - p_{\text{background}},
```
where ``c`` denotes the speed of sound, ``\rho`` the density, ``p`` the pressure,
``\rho_0`` an arbitrary reference density, ``p_0`` the pressure
at the reference density, and ``p_{\text{background}}`` the atmospheric or
background pressure (to be used with free surfaces).

When using [`SummationDensity`](@ref) (or [`DensityReinitializationCallback`](@ref))
and free surfaces, initializing particles with equal spacing will cause underestimated
density and therefore strong attractive forces between particles at the free surface.
Setting `clip_negative_pressure=true` can avoid this.
"""
struct StateEquationIdealGas{ELTYPE, CLIP} # Boolean to clip negative pressure
    sound_speed         :: ELTYPE
    reference_density   :: ELTYPE
    reference_pressure  :: ELTYPE
    background_pressure :: ELTYPE

    function StateEquationIdealGas(sound_speed, reference_density, reference_pressure;
                                   background_pressure=0.0, clip_negative_pressure=false)
        # `clip_negative_pressure` is made a type parameter, so that we can
        new{typeof(sound_speed),
            clip_negative_pressure}(sound_speed, reference_density, reference_pressure,
                                    background_pressure)
    end
end

clip_negative_pressure(::StateEquationIdealGas{<:Any, CLIP}) where {CLIP} = CLIP

function (state_equation::StateEquationIdealGas)(density)
    (; sound_speed, reference_density, reference_pressure, background_pressure) = state_equation

    pressure = sound_speed^2 * (density - reference_density) + reference_pressure -
               background_pressure

    # This is determined statically and has therefore no overhead
    if clip_negative_pressure(state_equation)
        return max(0.0, pressure)
    end

    return pressure
end

function inverse_state_equation(state_equation::StateEquationIdealGas, pressure)
    (; sound_speed, reference_density, reference_pressure, background_pressure) = state_equation

    return (pressure + background_pressure - reference_pressure) / sound_speed^2 +
           reference_density
end

@doc raw"""
    StateEquationCole(sound_speed, gamma, reference_density, reference_pressure;
                      background_pressure=0.0, clip_negative_pressure=false)

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

When using [`SummationDensity`](@ref) (or [`DensityReinitializationCallback`](@ref))
and free surfaces, initializing particles with equal spacing will cause underestimated
density and therefore strong attractive forces between particles at the free surface.
Setting `clip_negative_pressure=true` can avoid this.

## References:
- Robert H. Cole. "Underwater Explosions". Princeton University Press, 1948.
"""
struct StateEquationCole{ELTYPE, CLIP} # Boolean to clip negative pressure
    sound_speed         :: ELTYPE
    gamma               :: ELTYPE
    reference_density   :: ELTYPE
    reference_pressure  :: ELTYPE
    background_pressure :: ELTYPE

    function StateEquationCole(sound_speed, gamma, reference_density, reference_pressure;
                               background_pressure=0.0, clip_negative_pressure=false)
        new{typeof(sound_speed),
            clip_negative_pressure}(sound_speed, gamma, reference_density,
                                    reference_pressure, background_pressure)
    end
end

clip_negative_pressure(::StateEquationCole{<:Any, CLIP}) where {CLIP} = CLIP

function (state_equation::StateEquationCole)(density)
    (; sound_speed, gamma, reference_density, reference_pressure, background_pressure) = state_equation

    pressure = reference_density * sound_speed^2 / gamma *
               ((density / reference_density)^gamma - 1) +
               reference_pressure - background_pressure

    # This is determined statically and has therefore no overhead
    if clip_negative_pressure(state_equation)
        return max(0.0, pressure)
    end

    return pressure
end

function inverse_state_equation(state_equation::StateEquationCole, pressure)
    (; sound_speed, gamma, reference_density, reference_pressure, background_pressure) = state_equation

    tmp = gamma * (pressure + background_pressure - reference_pressure) /
          (reference_density * sound_speed^2) + 1
    density = reference_density * tmp^(1 / gamma)

    return density
end
