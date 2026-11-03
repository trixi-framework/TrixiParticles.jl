module HydrostaticInitialState

using TrixiParticles

const POLICY = "operator-offset-hydrostatic"

function prepare!(initial_condition, equation; surface_height, gravity)
    c2 = TrixiParticles.sound_speed(equation)^2
    exponent = equation.exponent
    for particle in eachindex(initial_condition.density)
        depth = max(surface_height - initial_condition.coordinates[2, particle], 0)
        log_ratio = exponent == 1 ? gravity * depth / c2 :
                    log1p((exponent - 1) * gravity * depth / c2) / (exponent - 1)
        density = equation.reference_density * exp(log_ratio)
        initial_condition.density[particle] = density
        initial_condition.pressure[particle] = equation(density)
    end
    return initial_condition
end

function diagnostics(fluid)
    target = fluid.initial_condition.density
    pressure = fluid.state_equation.(target)
    return (; policy=POLICY,
            relative_density_error=maximum(abs.(fluid.cache.density - target)) /
                                   fluid.state_equation.reference_density,
            maximum_pressure_error=maximum(abs.(fluid.pressure - pressure)),
            total_fluid_mass=sum(fluid.mass),
            density_offset_extrema=extrema(fluid.cache.initial_density_offset),
            filtered_density_offset_extrema=extrema(fluid.cache.initial_filtered_density_offset))
end

end # module
