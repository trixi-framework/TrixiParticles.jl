struct NoSurfaceTension end

function (::NoSurfaceTension)(smoothing_length, ma, mb, dx, distance)
    return 0.0
end

@doc raw"""
cohesionForceAkinci(smoothing_length, ma, mb, dx, distance)
Reference:
Versatile Surface Tension and Adhesion for SPH Fluids, Akinci et al, 2013, Siggraph Asia
"""

struct CohesionForceAkinci{ELTYPE}
    surface_tension_coefficient::ELTYPE

    function CohesionForceAkinci(surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

function (surface_tension::CohesionForceAkinci)(smoothing_length, ma, mb, dx, distance)
    @unpack surface_tension_coefficient = surface_tension

    C = 0
    if 2 * distance > smoothing_length && distance <= smoothing_length
        C = (smoothing_length - distance)^3 * distance^3
    elseif distance > eps(Float64) && 2 * distance <= smoothing_length
        C = 2 * (smoothing_length - distance)^3 * distance^3 - smoothing_length^6/64.0
    end
    C *= 32.0/(pi * smoothing_length^9)

    return - surface_tension_coefficient * ma * mb * C * dx/distance
end