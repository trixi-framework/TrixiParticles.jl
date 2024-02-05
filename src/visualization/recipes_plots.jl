const TrixiParticlesODESolution = ODESolution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any,
                                              <:ODEProblem{<:Any, <:Any, <:Any,
                                                           <:Semidiscretization}}

RecipesBase.@recipe function f(sol::TrixiParticlesODESolution)
    # Redirect everything to the recipe
    return sol.u[end].x..., sol.prob.p
end

RecipesBase.@recipe function f(v_ode, u_ode, semi::Semidiscretization)
    systems_data = map(semi.systems) do system
        (; initial_condition) = system

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        coordinates = current_coordinates(u, system)
        x = collect(coordinates[1, :])
        y = collect(coordinates[2, :])

        particle_spacing = system.initial_condition.particle_spacing

        x_min, y_min = minimum(initial_condition.coordinates, dims=2) .- 0.5particle_spacing
        x_max, y_max = maximum(initial_condition.coordinates, dims=2) .+ 0.5particle_spacing

        return (; x, y, x_min, x_max, y_min, y_max, particle_spacing,
                label=timer_name(system))
    end

    x_min = minimum(data.x_min for data in systems_data)
    x_max = maximum(data.x_max for data in systems_data)
    y_min = minimum(data.y_min for data in systems_data)
    y_max = maximum(data.y_max for data in systems_data)

    pixel_size = max((x_max - x_min) / 600, (y_max - y_min) / 400)

    xlims --> (x_min, x_max)
    ylims --> (y_min, y_max)
    aspect_ratio --> :equal

    seriestype --> :scatter
    markerstrokewidth --> 0
    grid --> false

    for system_data in systems_data
        @series begin
            pixels_per_particle = system_data.particle_spacing / pixel_size

            # Marker radius in pixels
            markersize --> 0.5 * pixels_per_particle

            label --> system_data.label

            # Return data for plotting
            system_data.x, system_data.y
        end
    end
end
