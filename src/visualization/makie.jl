"""
    trixi2makie(scene, solution; frame=lastindex(solution.u), kwargs...)
    trixi2makie(scene, v_ode, u_ode, semi; kwargs...)

Plot a TrixiParticles solution in a Makie `Scene` or `LScene` as physically sized particle
spheres. The first method plots one frame of an ODE solution, while the second accepts the
position and state arrays explicitly. This particle-level view is intended for diagnostics; fluid
surface reconstruction requires additional post-processing.

This function is available after loading Makie or one of its backends, such as CairoMakie,
GLMakie, or RayMakie. See the visualization documentation for the supported keyword
arguments.
"""
function trixi2makie end
