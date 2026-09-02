"""
    trixi2makie(solution; frame=Makie.automatic, kwargs...)
    trixi2makie(v_ode, u_ode, semi; kwargs...)
    trixi2makie!(axis, args...; kwargs...)

Plot a TrixiParticles solution with Makie using physically sized particle markers. The first
method plots one frame of an ODE solution, while the second accepts the position and state
arrays explicitly. Use `trixi2makie!` to add the plot to an existing Makie axis.

Makie's generic `plot` and `plot!` functions support the same arguments, so `plot(solution)`
and `plot!(axis, solution)` use this recipe as well. This particle-level view is intended for
diagnostics; fluid surface reconstruction requires additional post-processing.

This function is available after loading Makie or one of its backends, such as CairoMakie,
GLMakie, or RayMakie. See the visualization documentation for the supported keyword
arguments.
"""
function trixi2makie end

function trixi2makie! end
