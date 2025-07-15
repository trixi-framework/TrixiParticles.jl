# Solution type when running a simulation with TrixiParticles.jl and OrdinaryDiffEq.jl
const TrixiParticlesODESolution = ODESolution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any,
                                              <:ODEProblem{<:Any, <:Any, <:Any,
                                                           <:Semidiscretization}}

RecipesBase.@recipe function f(sol::TrixiParticlesODESolution)
    # Redirect everything to the next recipe
    return sol.u[end].x..., sol.prob.p
end

# GPU version
RecipesBase.@recipe function f(sol::TrixiParticlesODESolution, semi::Semidiscretization)
    # Move GPU data to the CPU
    v_ode = Array(sol.u[end].x[1])
    u_ode = Array(sol.u[end].x[2])
    semi_ = Adapt.adapt(Array, sol.prob.p)

    # Redirect everything to the next recipe
    return v_ode, u_ode, semi_, particle_spacings(semi)
end

RecipesBase.@recipe function f(v_ode::AbstractGPUArray, u_ode, semi::Semidiscretization;
                               particle_spacings=nothing,
                               size=(600, 400), # Default size
                               xlims=(-Inf, Inf), ylims=(-Inf, Inf))
    throw(ArgumentError("to plot GPU data, use `plot(sol, semi)`"))
end

RecipesBase.@recipe function f(v_ode, u_ode, semi::Semidiscretization;
                               particle_spacings=TrixiParticles.particle_spacings(semi),
                               size=(600, 400), # Default size
                               xlims=(-Inf, Inf), ylims=(-Inf, Inf))
    return v_ode, u_ode, semi, particle_spacings
end

RecipesBase.@recipe function f(v_ode, u_ode, semi::Semidiscretization, particle_spacings;
                               size=(600, 400), # Default size
                               xlims=(-Inf, Inf), ylims=(-Inf, Inf))
    systems_data = map(enumerate(semi.systems)) do (i, system)
        u = wrap_u(u_ode, system, semi)
        periodic_box = get_neighborhood_search(system, semi).periodic_box
        coordinates = PointNeighbors.periodic_coords(active_coordinates(u, system),
                                                     periodic_box)

        x = collect(coordinates[1, :])
        y = collect(coordinates[2, :])

        particle_spacing = particle_spacings[i]
        if particle_spacing < 0
            particle_spacing = 0.0
        end

        x_min, y_min = minimum(coordinates, dims=2) .- 0.5particle_spacing
        x_max, y_max = maximum(coordinates, dims=2) .+ 0.5particle_spacing

        # `x_min`, `x_max`, etc. are used to automatically set the marker size.
        # When `xlims` or `ylims` are passed explicitly, we have to update these to get the correct marker size.
        isfinite(first(xlims)) && (x_min = xlims[1])
        isfinite(last(xlims)) && (x_max = xlims[2])

        isfinite(first(ylims)) && (y_min = ylims[1])
        isfinite(last(ylims)) && (y_max = ylims[2])

        return (; x, y, x_min, x_max, y_min, y_max, particle_spacing,
                label=timer_name(system))
    end

    return (semi, systems_data...)
end

function particle_spacings(semi::Semidiscretization)
    return [system.initial_condition.particle_spacing for system in semi.systems]
end

RecipesBase.@recipe function f((initial_conditions::InitialCondition)...;
                               xlims=(Inf, Inf), ylims=(Inf, Inf))
    idx = 0
    ics = map(initial_conditions) do ic
        x = collect(ic.coordinates[1, :])
        y = collect(ic.coordinates[2, :])

        particle_spacing = ic.particle_spacing
        if particle_spacing < 0
            particle_spacing = 0.0
        end

        x_min, y_min = minimum(ic.coordinates, dims=2) .- 0.5particle_spacing
        x_max, y_max = maximum(ic.coordinates, dims=2) .+ 0.5particle_spacing

        # `x_min`, `x_max`, etc. are used to automatically set the marker size.
        # When `xlims` or `ylims` are passed explicitly, we have to update these to get the correct marker size.
        isfinite(first(xlims)) && (x_min = xlims[1])
        isfinite(last(xlims)) && (x_max = xlims[2])

        isfinite(first(ylims)) && (y_min = ylims[1])
        isfinite(last(ylims)) && (y_max = ylims[2])

        idx += 1

        return (; x, y, x_min, x_max, y_min, y_max, particle_spacing,
                label="initial condition " * "$idx")
    end

    return (first(initial_conditions), ics...)
end

RecipesBase.@recipe function f(::Union{InitialCondition, Semidiscretization},
                               data...; zcolor=nothing, size=(600, 400), colorbar_title="",
                               xlims=(Inf, Inf), ylims=(Inf, Inf))
    x_min = minimum(obj.x_min for obj in data)
    x_max = maximum(obj.x_max for obj in data)

    y_min = minimum(obj.y_min for obj in data)
    y_max = maximum(obj.y_max for obj in data)

    # `x_min`, `x_max`, etc. are used to automatically set the marker size.
    # When `xlims` or `ylims` are passed explicitly, we have to update these to get the correct marker size.
    isfinite(first(xlims)) && (x_min = xlims[1])
    isfinite(last(xlims)) && (x_max = xlims[2])

    isfinite(first(ylims)) && (y_min = ylims[1])
    isfinite(last(ylims)) && (y_max = ylims[2])

    # Note that this assumes the plot area to be ~10% smaller than `size`,
    # which is the case when showing a single plot with the legend inside.
    # With the legend outside, this is no longer the case, so the `markersize` has to be
    # set manually.
    pixel_size = max((x_max - x_min) / size[1], (y_max - y_min) / size[2])

    xlims --> (x_min, x_max)
    ylims --> (y_min, y_max)
    aspect_ratio --> :equal

    seriestype --> :scatter
    markerstrokewidth --> 0
    grid --> false
    colorbar_title --> colorbar_title
    zcolor --> zcolor

    for obj in data
        @series begin
            if obj.particle_spacing < eps()
                # Fall back to 1px marker radius
                markersize --> 1
            else
                pixels_per_particle = obj.particle_spacing / pixel_size

                # Marker radius in pixels
                markersize --> 0.5 * pixels_per_particle
            end

            label --> obj.label

            # Return data for plotting
            obj.x, obj.y
        end
    end
end

RecipesBase.@recipe function f(geometry::Polygon)
    x = stack(geometry.vertices)[1, :]
    y = stack(geometry.vertices)[2, :]

    aspect_ratio --> :equal
    grid --> false
    label --> "geometry"

    # First plot the vertices as a scatter plot
    @series begin
        seriestype --> :scatter
        return (x, y)
    end

    # Now plot the edges as a line plot
    @series begin
        # Ignore series in legend and color cycling. Note that `:=` forces the attribute,
        # whereas `-->` would only set it if it is not already set.
        primary := false

        return (x, y)
    end
end
