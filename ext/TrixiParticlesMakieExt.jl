module TrixiParticlesMakieExt

using Makie
using TrixiParticles

import TrixiParticles: trixi2makie, trixi2makie!

const TP = TrixiParticles

function default_system_color(system, system_index)
    if system isa TP.AbstractFluidSystem
        return Makie.RGBf(0.02, 0.32, 0.85)
    elseif system isa TP.AbstractBoundarySystem
        return Makie.RGBf(0.62, 0.66, 0.72)
    elseif system isa TP.AbstractStructureSystem
        return Makie.RGBf(0.95, 0.48, 0.08)
    elseif system isa TP.OpenBoundarySystem
        return Makie.RGBf(0.30, 0.58, 0.82)
    end

    return Makie.wong_colors()[mod1(system_index, length(Makie.wong_colors()))]
end

function default_marker_size_scale(system, system_index)
    if system isa TP.AbstractBoundarySystem || system isa TP.OpenBoundarySystem
        return 0.55
    end

    return 0.9
end

@inline function style_value(style::Function, system, system_index)
    return style(system, system_index)
end

@inline function style_value(style::AbstractVector, system, system_index)
    return style[system_index]
end

@inline style_value(style, system, system_index) = style

"""
    trixi2makie(solution; frame=Makie.automatic, kwargs...)
    trixi2makie(v_ode, u_ode, semi; kwargs...)

Plot a TrixiParticles solution using Makie. See `TrixiParticles.trixi2makie` for details.
"""
Makie.@recipe Trixi2Makie begin
    "Frame of the ODE solution to plot. `Makie.automatic` selects the last frame."
    frame = Makie.automatic
    "Indices of the systems to plot. `Makie.automatic` selects all systems."
    system_indices = Makie.automatic
    "A color, vector indexed by system number, or `(system, index)` function."
    system_colors = default_system_color
    "A size scale, vector indexed by system number, or `(system, index)` function."
    marker_size_scales = default_marker_size_scale
    Makie.documented_attributes(Makie.MeshScatter)...
    marker = Makie.automatic
    markerspace = :data
end

Makie.plottype(::TP.TrixiParticlesODESolution) = Trixi2Makie
function Makie.plottype(::AbstractArray, ::AbstractArray, ::TP.Semidiscretization)
    return Trixi2Makie
end

# Avoid the generic SciMLBase Makie conversion for ODE solutions after `plottype` selects
# this recipe.
function Makie.convert_arguments(::Type{<:Trixi2Makie},
                                 solution::TP.TrixiParticlesODESolution)
    return (solution,)
end

function visualization_state(args::Tuple{TP.TrixiParticlesODESolution}, frame)
    solution = only(args)
    frame_index = frame isa Makie.Automatic ? lastindex(solution.u) : frame
    v_ode, u_ode = solution.u[frame_index].x
    return v_ode, u_ode, solution.prob.p.semi
end

function visualization_state(args::Tuple{<:AbstractArray, <:AbstractArray,
                                         <:TP.Semidiscretization}, frame)
    return args
end

function semidiscretization(args::Tuple{TP.TrixiParticlesODESolution})
    return only(args).prob.p.semi
end

function semidiscretization(args::Tuple{<:AbstractArray, <:AbstractArray,
                                        <:TP.Semidiscretization})
    return last(args)
end

function Makie.preferred_axis_type(plot::Trixi2Makie)
    semi = semidiscretization(plot.args[])
    ndims_ = ndims(first(semi.systems))

    if ndims_ == 2
        return Makie.Axis
    elseif ndims_ == 3
        return Makie.Axis3
    end

    throw(ArgumentError("Makie visualization is only supported in two or three dimensions"))
end

function Makie.preferred_axis_attributes(::Type{Makie.Axis}, ::Trixi2Makie)
    return (; aspect=Makie.DataAspect())
end

function Makie.preferred_axis_attributes(::Type{Makie.Axis3}, ::Trixi2Makie)
    return (; aspect=:data)
end

function Makie.plot!(plot::Trixi2Makie)
    v_ode, u_ode, semi = visualization_state(plot.args[], plot.frame[])
    system_indices = plot.system_indices[]
    system_indices isa Makie.Automatic && (system_indices = eachindex(semi.systems))

    for system_index in system_indices
        system = semi.systems[system_index]
        particles = TP.eachparticle(system)
        isempty(particles) && continue

        u = TP.wrap_u(u_ode, system, semi)
        coordinates = Array(TP.active_coordinates(u, system))
        spacing = TP.particle_spacing(system, first(particles))
        color = style_value(plot.system_colors[], system, system_index)
        marker_size_scale = style_value(plot.marker_size_scales[], system, system_index)
        marker = plot.marker[]

        if ndims(system) == 2
            marker isa Makie.Automatic && (marker = Makie.Circle(Makie.Point2f(0), 0.5f0))
            points = makie_points(coordinates, Val(2))
            Makie.meshscatter!(plot, plot.attributes, points; marker,
                               markersize=marker_size_scale * spacing, color)
        else
            marker isa Makie.Automatic &&
                (marker = Makie.Sphere(Makie.Point3f(0), 0.5f0))
            points = makie_points(coordinates, Val(3))
            Makie.meshscatter!(plot, plot.attributes, points; marker,
                               markersize=marker_size_scale * spacing, color)
        end
    end

    return plot
end

function makie_points(coordinates, ::Val{2})
    return [Makie.Point2f(coordinates[1, particle], coordinates[2, particle])
            for particle in axes(coordinates, 2)]
end

function makie_points(coordinates, ::Val{3})
    return [Makie.Point3f(coordinates[1, particle], coordinates[2, particle],
                          coordinates[3, particle])
            for particle in axes(coordinates, 2)]
end

end # module
