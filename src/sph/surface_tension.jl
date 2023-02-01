struct NoSurfaceTension end

function (::NoSurfaceTension)(smoothing_length, mb, na, nb, dx, distance)
    return 0.0 * dx
end

@doc raw"""
cohesionForceAkinci(smoothing_length, ma, mb, dx, distance)

Use the cohesion force model by Akinci. This is based on an Intra-particle-force formulation.

# Keywords
- 'surface_tension_coefficient=1.0': Coefficient that linearly scales the surface tension induced force.
- 'support_length=NaN': Defaults to the smoothing length of the SPH Method. Cutoff length of the force calculation

Reference:
Versatile Surface Tension and Adhesion for SPH Fluids, Akinci et al, 2013, Siggraph Asia
"""

struct CohesionForceAkinci{ELTYPE}
    surface_tension_coefficient    :: ELTYPE
    surface_tension_support_length :: ELTYPE

    function CohesionForceAkinci(; surface_tension_coefficient=1.0, support_length=NaN)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient,
                                                 support_length)
    end
end

function (surface_tension::CohesionForceAkinci)(smoothing_length, mb, na, nb, dx, distance)
    @unpack surface_tension_coefficient, surface_tension_support_length = surface_tension

    if !isnan(surface_tension_support_length)
        smoothing_length = surface_tension_support_length
    end

    # Eq. 2
    C = 0
    if distance^2 <= smoothing_length^2
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
SurfaceTensionAkinci(smoothing_length, mb, na, nb, dx, distance)

Use the surface tension model by Akinci. This is based on an Intra-particle-force formulation.

# Keywords
- 'surface_tension_coefficient=1.0': Coefficient that linearly scales the surface tension induced force.
- 'support_length=NaN': Defaults to the smoothing length of the SPH Method. Cutoff length of the force calculation

Reference:
Versatile Surface Tension and Adhesion for SPH Fluids, Akinci et al, 2013, Siggraph Asia
"""

struct SurfaceTensionAkinci{ELTYPE}
    surface_tension_coefficient::ELTYPE
    surface_tension_support_length::ELTYPE
    cof_model::CohesionForceAkinci

    function SurfaceTensionAkinci(; surface_tension_coefficient=1.0, support_length=NaN)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient,
                                                 support_length,
                                                 CohesionForceAkinci(surface_tension_coefficient=surface_tension_coefficient,
                                                                     support_length=support_length))
    end
end

function (surface_tension::SurfaceTensionAkinci)(smoothing_length, mb, na, nb, dx, distance)
    @unpack surface_tension_coefficient, cof_model = surface_tension

    cof = cof_model(smoothing_length, mb, na, nb, dx, distance)
    surface_force = -surface_tension_coefficient * (na - nb)
    return cof .+ surface_force
end
