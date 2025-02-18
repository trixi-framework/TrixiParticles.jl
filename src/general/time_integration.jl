# Time integration is handled by the package OrdinaryDiffEq.jl.
# See the docs for more details.
# In this file, we define the structs for extra time integration schemes that
# are implemented in the package extension TrixiParticlesOrdinaryDiffEqExt.jl.
"""
    SymplecticPositionVerlet()

Modified leapfrog integration scheme for Weakly Compressible SPH (WCSPH) when integrating
the density with [`ContinuityDensity`](@ref).
This scheme is used by the SPH code [DualSPHysics](https://github.com/DualSPHysics/DualSPHysics).
See
[https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#372-symplectic-position-verlet-scheme]
(https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#372-symplectic-position-verlet-scheme)
and [Domínguez et al. 2022, Section 2.5.2](@cite Dominguez2022).

See [time integration](@ref time_integration) for more details.
"""
function SymplecticPositionVerlet(_...)
    error("the package OrdinaryDiffEq.jl needs to be loaded to use this scheme.")
end
