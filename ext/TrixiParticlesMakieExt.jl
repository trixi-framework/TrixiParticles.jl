module TrixiParticlesMakieExt

using Makie
using TrixiParticles

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

function TP.trixi2makie(scene, solution::TP.TrixiParticlesODESolution;
                        frame=lastindex(solution.u), kwargs...)
    v_ode, u_ode = solution.u[frame].x
    semi = solution.prob.p.semi

    return TP.trixi2makie(scene, v_ode, u_ode, semi; kwargs...)
end

function TP.trixi2makie(scene, v_ode::AbstractArray, u_ode::AbstractArray,
                        semi::TP.Semidiscretization;
                        system_indices=eachindex(semi.systems),
                        system_colors=default_system_color,
                        marker_size_scales=default_marker_size_scale,
                        kwargs...)
    plots = Any[]
    marker = Makie.Sphere(Makie.Point3f(0), 0.5f0)
    for system_index in system_indices
        system = semi.systems[system_index]
        particles = TP.eachparticle(system)
        isempty(particles) && continue

        u = TP.wrap_u(u_ode, system, semi)
        coordinates = Array(TP.active_coordinates(u, system))
        points = makie_points(coordinates)
        spacing = TP.particle_spacing(system, first(particles))
        color = style_value(system_colors, system, system_index)
        marker_size_scale = style_value(marker_size_scales, system, system_index)

        plot = Makie.meshscatter!(scene, points; marker,
                                  markersize=marker_size_scale * spacing,
                                  color, kwargs...)
        push!(plots, plot)
    end

    return plots
end

function makie_points(coordinates)
    if size(coordinates, 1) == 2
        return [Makie.Point3f(coordinates[1, particle], coordinates[2, particle], 0)
                for particle in axes(coordinates, 2)]
    elseif size(coordinates, 1) == 3
        return [Makie.Point3f(coordinates[1, particle], coordinates[2, particle],
                              coordinates[3, particle])
                for particle in axes(coordinates, 2)]
    end

    throw(ArgumentError("Makie visualization is only supported in two or three dimensions"))
end

end # module
