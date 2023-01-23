struct NoSurfaceTension end

function (::NoSurfaceTension)(smoothing_length, ma, mb, na, nb, dx, distance)
    return 0.0 * dx
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

function (surface_tension::CohesionForceAkinci)(smoothing_length, ma, mb, na, nb, dx, distance)
    @unpack surface_tension_coefficient = surface_tension

    # Eq. 2
    C = 0
    if distance < smoothing_length
        if distance > 0.5 * smoothing_length
            # attractive force
            C = (smoothing_length - distance)^3 * distance^3
        else
            # repulsive force
            C = 2 * (smoothing_length - distance)^3 * distance^3 - smoothing_length^6 / 64.0
        end
        C *= 32.0 / (pi * smoothing_length^9)
    end

    # Eq. 1 in acceleration form
    return -surface_tension_coefficient * mb * C * dx / distance
end

@doc raw"""
SurfaceTensionAkinci(smoothing_length, ma, mb, na, nb, dx, distance)
Reference:
Versatile Surface Tension and Adhesion for SPH Fluids, Akinci et al, 2013, Siggraph Asia
"""

struct SurfaceTensionAkinci{ELTYPE}
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

function (surface_tension::SurfaceTensionAkinci)(smoothing_length, ma, mb, na, nb, dx, distance)
    @unpack surface_tension_coefficient = surface_tension

    surface_force = - surface_tension_coefficient * (na - nb)
    cof = CohesionForceAkinci(surface_tension_coefficient)(smoothing_length, ma, mb, na, nb, dx, distance)
    #return cof .+ surface_force
    return surface_force
end
