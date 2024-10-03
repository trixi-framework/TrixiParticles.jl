# Solution type when running a simulation with TrixiParticles.jl and OrdinaryDiffEq.jl
const TrixiParticlesODESolution = ODESolution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any,
                                              <:ODEProblem{<:Any, <:Any, <:Any,
                                                           <:Semidiscretization}}

RecipesBase.@recipe function f(sol::TrixiParticlesODESolution)
    # Redirect everything to the recipe
    return sol.u[end].x..., sol.prob.p
end

RecipesBase.@recipe function f(v_ode, u_ode, semi::Semidiscretization;
                               size=(600, 400)) # Default size
    systems_data = map(semi.systems) do system
        u = wrap_u(u_ode, system, semi)
        coordinates = active_coordinates(u, system)
        x = collect(coordinates[1, :])
        y = collect(coordinates[2, :])

        particle_spacing = system.initial_condition.particle_spacing
        if particle_spacing < 0
            particle_spacing = 0.0
        end

        x_min, y_min = minimum(coordinates, dims=2) .- 0.5particle_spacing
        x_max, y_max = maximum(coordinates, dims=2) .+ 0.5particle_spacing

        return (; x, y, x_min, x_max, y_min, y_max, particle_spacing,
                label=timer_name(system))
    end

    return (semi, systems_data...)
end

RecipesBase.@recipe function f((initial_conditions::InitialCondition)...)
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

    return (first(initial_conditions), ics...)
end

RecipesBase.@recipe function f(::Union{InitialCondition, Semidiscretization},
                               data...; zcolor=nothing, size=(600, 400), colorbar_title="")
    x_min = minimum(obj.x_min for obj in data)
    x_max = maximum(obj.x_max for obj in data)
    y_min = minimum(obj.y_min for obj in data)
    y_max = maximum(obj.y_max for obj in data)

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
