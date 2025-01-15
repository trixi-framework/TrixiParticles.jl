# Time integration is handles by the package OrdinaryDiffEq.jl.
# See the docs for more details.
# In this file, we define the structs for extra time integration schemes that
# are implemented in the package extension TrixiParticlesOrdinaryDiffEqExt.jl.
"""
    SymplecticPositionVerlet()

Modified leapfrog integration scheme for Weakly Compressible SPH (WCSPH) when integrating
the density with [`ContinuityDensity`](@ref).
This scheme is used by DualSPHysics:
https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#372-symplectic-position-verlet-scheme
"""
SymplecticPositionVerlet(_) = nothing
