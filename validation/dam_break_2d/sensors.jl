function max_x_coord(system, data, t)
    return maximum(j -> data.coordinates[1, j], axes(data.coordinates, 2))
end

function interpolated_pressure(coord_top, coord_bottom, v_ode, u_ode, t, system, semi) end

function interpolated_pressure(coord_top, coord_bottom, v_ode, u_ode, t,
                               system::TrixiParticles.AbstractFluidSystem, semi)
    n_interpolation_points = 10
    interpolated_values = interpolate_line(coord_top, coord_bottom,
                                           n_interpolation_points, semi, system, v_ode,
                                           u_ode,
                                           smoothing_length=2.0 *
                                                            TrixiParticles.initial_smoothing_length(system),
                                           clip_negative_pressure=true, cut_off_bnd=false)
    return sum(map(x -> isnan(x) ? 0.0 : x, interpolated_values.pressure)) /
           n_interpolation_points
end
