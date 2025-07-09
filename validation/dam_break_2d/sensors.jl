function max_x_coord(system, data, t)
    return maximum(j -> data.coordinates[1, j], axes(data.coordinates, 2))
end

function interpolated_pressure(coord_top, coord_bottom, v_ode, u_ode, t, system, semi) end

function interpolated_pressure(coord_top, coord_bottom, v_ode, u_ode, t, system::TrixiParticles.FluidSystem, semi)
    # use at least 5 interpolation points for low resolution simulations
    # otherwise use at least the number of particles present
    n_interpolation_points = min(5, Int(ceil(sensor_size / particle_spacing)))
    interpolated_values = interpolate_line(coord_top, coord_bottom,
                                           n_interpolation_points, semi, system, v_ode,
                                           u_ode,
                                           smoothing_length=2.0 *
                                                            TrixiParticles.initial_smoothing_length(system),
                                           clip_negative_pressure=true)
    return sum(map(x -> isnan(x) ? 0.0 : x, interpolated_values.pressure)) /
           n_interpolation_points
end

function pressure_probe(coord_top, coord_bottom, v_ode, u_ode, t, system, semi) end

function pressure_probe(coord_top, coord_bottom, v_ode, u_ode, t, system::TrixiParticles.BoundarySystem, semi)
    # The sensor is at the right wall, so its x-coordinate is the same for top and bottom.
    x_sensor = coord_top[1]

    # Use the initial particle spacing as a reference for the thickness of the averaging region.
    # A thickness of one or two particle spacings is usually a good choice.
    region_thickness = 2.0 * particle_spacing

    # Define the rectangular region for averaging
    x_min = x_sensor - region_thickness
    x_max = x_sensor + region_thickness
    y_min = coord_bottom[2]
    y_max = coord_top[2]

    sum_of_pressures = 0.0
    num_particles_in_region = 0

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    # Iterate over each particle in the specified fluid system
    for particle_idx in TrixiParticles.eachparticle(system)
        pc = TrixiParticles.current_coords(u, system, particle_idx)

        # Get coordinates for the current particle from the 1D vector
        px = pc[1] # x-coordinate
        py = pc[2] # y-coordinate

        # Check if the particle is inside the sensor's rectangular region
        if (x_min <= px <= x_max) && (y_min <= py <= y_max)
            # Add its pressure to the sum and increment the count
            sum_of_pressures += TrixiParticles.current_pressure(v, system, particle_idx)
            num_particles_in_region += 1
        end
    end

    # If no particles are in the region (e.g., before the water hits the wall),
    # the pressure is zero.
    if num_particles_in_region == 0
        return 0.0
    end

    # Return the calculated average pressure
    return sum_of_pressures / num_particles_in_region
end
