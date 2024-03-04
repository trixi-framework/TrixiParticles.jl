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
        (; initial_condition) = system

        v = wrap_u(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        coordinates = current_coordinates(u, system)
        x = collect(coordinates[1, :])
        y = collect(coordinates[2, :])

        particle_spacing = system.initial_condition.particle_spacing
        if particle_spacing < 0
            particle_spacing = 0.0
        end

        x_min, y_min = minimum(initial_condition.coordinates, dims=2) .- 0.5particle_spacing
        x_max, y_max = maximum(initial_condition.coordinates, dims=2) .+ 0.5particle_spacing

        pressure = [particle_pressure(v, system, particle) for particle in eachparticle(system)]

        return (; x, y, x_min, x_max, y_min, y_max, particle_spacing, pressure,
                label=timer_name(system))
    end

    x_min = minimum(data.x_min for data in systems_data)
    x_max = maximum(data.x_max for data in systems_data)
    y_min = minimum(data.y_min for data in systems_data)
    y_max = maximum(data.y_max for data in systems_data)

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

    for system_data in systems_data
        @series begin
            if system_data.particle_spacing < eps()
                # Fall back to 1px marker radius
                markersize --> 1
            else
                pixels_per_particle = system_data.particle_spacing / pixel_size

                # Marker radius in pixels
                markersize --> 0.5 * pixels_per_particle
            end

            label --> system_data.label
            if startswith(system_data.label, "boundary")
                color --> :black
            else
                color --> :coolwarm
                zcolor --> system_data.pressure
            end

            # Return data for plotting
            system_data.x, system_data.y
        end
    end
end
