# Time integration is handled by the package OrdinaryDiffEq.jl.
# See the docs for more details.
# In this file, we define the structs for extra time integration schemes that
# are implemented in the package extension TrixiParticlesOrdinaryDiffEqSymplecticRKExt.
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
    error("the package OrdinaryDiffEqSymplecticRK needs to be loaded to use this scheme.")
end

function calculate_dt(v_ode, u_ode, cfl_number, semi::Semidiscretization)
    (; systems) = semi

    return minimum(systems) do system
        if system isa TotalLagrangianSPHSystem && !semi.integrate_tlsph[]
            # Skip TLSPH systems if they are not integrated
            return Inf
        end
        return calculate_dt(v_ode, u_ode, cfl_number, system, semi)
    end
end
