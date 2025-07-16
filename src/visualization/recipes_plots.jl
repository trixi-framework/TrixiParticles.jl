# Solution type when running a simulation with TrixiParticles.jl and OrdinaryDiffEq.jl
const TrixiParticlesODESolution = ODESolution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any,
                                              <:ODEProblem{<:Any, <:Any, <:Any,
                                                           <:Semidiscretization}}

# This is the main recipe
RecipesBase.@recipe function f(sol::TrixiParticlesODESolution)
    # Redirect everything to the next recipe
    return sol.u[end].x..., sol.prob.p
end

# GPU version
RecipesBase.@recipe function f(v_ode::AbstractGPUArray, u_ode, semi::Semidiscretization)
    # Move GPU data to the CPU
    v_ode_ = Array(v_ode)
    u_ode_ = Array(u_ode)
    semi_ = Adapt.adapt(Array, semi)

    # Redirect everything to the next recipe
    return v_ode_, u_ode_, semi_
end

RecipesBase.@recipe function f(v_ode, u_ode, semi::Semidiscretization;
                               particle_spacings=TrixiParticles.particle_spacings(semi),
                               size=(600, 400), # Default size
                               xlims=(-Inf, Inf), ylims=(-Inf, Inf))
    # We need to split this in two recipes in order to find the minimum and maximum
    # coordinates across all systems.
    # In this first recipe, we collect the data for each system,
    # and then pass it to the next recipe.
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

        return (; x, y, x_min, x_max, y_min, y_max, particle_spacing,
                label=timer_name(system))
    end

    # Pass the semidiscretization and the collected data to the next recipe
    return (semi, systems_data...)
end

function particle_spacings(semi::Semidiscretization)
    return [system.initial_condition.particle_spacing for system in semi.systems]
end

RecipesBase.@recipe function f(initial_conditions::InitialCondition...)
    # This recipe is similar to the one above for the semidiscretization
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

        idx += 1

        return (; x, y, x_min, x_max, y_min, y_max, particle_spacing,
                label="initial condition " * "$idx")
    end

    # Pass the first initial condition and the collected data to the next recipe
    return (first(initial_conditions), ics...)
end

RecipesBase.@recipe function f(::Union{InitialCondition, Semidiscretization},
                               data...; size=(600, 400), xlims=(Inf, Inf), ylims=(Inf, Inf))
    # `data` is a tuple of named tuples, passed from the recipe above.
    # Each named tuple contains coordinates and metadata for a system or initial condition.
    #
    # First find the minimum and maximum coordinates across all systems.
    x_min = minimum(obj.x_min for obj in data)
    x_max = maximum(obj.x_max for obj in data)

    y_min = minimum(obj.y_min for obj in data)
    y_max = maximum(obj.y_max for obj in data)

    # `x_min`, `x_max`, etc. are used to automatically set the marker size.
    # So they need to be the minimum and maximum coordinates of the plot area.
    # When `xlims` or `ylims` are passed explicitly, we have to update these
    # to get the correct marker size.
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

    # Just having the kwargs `xlims` and `ylims` (without setting them)
    # is enough to make `widen = :auto`  fall back to no widening.
    # When no explicit limits are passed, we overwrite this.
    if all(!isfinite, xlims) && all(!isfinite, ylims)
        widen --> true
    end

    aspect_ratio --> :equal
    seriestype --> :scatter
    # No border around the markers
    markerstrokewidth --> 0
    grid --> false

    # Now plot all systems or initial conditions
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
