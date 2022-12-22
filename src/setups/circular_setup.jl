function draw_circle(R, x_center, y_center, particle_spacing)

    # particle_spacing = sqrt( (r_x1 - r_x2)^2 + (r_y1 - r_y2)^2 )
    # particle_spacing = sqrt( ( (R*cos(t)+R) - (R*cos(t+ Δ)+R) )^2 + ( (R*sin(t)+R) - (R*sin(t+ Δ)+R) )^2 )
    delta_t =  acos((2*R^2 - particle_spacing^2)/(2*R^2))

    t = collect(0:delta_t:2*pi-delta_t)

    # force equidistant spacing
    t = LinRange(0, 2*pi-delta_t, length(t))

    particle_coords = Array{Float64,2}(undef, 2, length(t));

    for i in eachindex(t)
        particle_coords[1, i] = R * cos(t[i]) + x_center
        particle_coords[2, i] = R * sin(t[i]) + y_center
    end

    return particle_coords
end

function fill_circle(R, x_center, y_center, particle_spacing)

    x_vec = Vector{Float64}(undef,0)
    y_vec = Vector{Float64}(undef,0)

    r(x,y) = sqrt( (x - x_center)^2 + (y - y_center)^2 )

    for j in -Int(R/particle_spacing):Int(R/particle_spacing),
            i in -Int(R/particle_spacing):Int(R/particle_spacing)

            x = x_center + i * particle_spacing
            y = y_center + j * particle_spacing

        if r(x,y) < R
            append!(x_vec, x)
            append!(y_vec, y)
        end
    end

    particle_coords = Array{Float64, 2}(undef, 2, length(x_vec))
    particle_coords[1,:] = x_vec
    particle_coords[2,:] = y_vec

    return particle_coords
end


function fill_circle_with_recess(R, x_center, y_center, x_recess, y_recess, particle_spacing)

    x_vec = Vector{Float64}(undef,0)
    y_vec = Vector{Float64}(undef,0)

    r(x,y) = sqrt( (x - x_center)^2 + (y - y_center)^2 )

    for j in -round(Int, R/particle_spacing):round(Int, R/particle_spacing),
            i in -round(Int, R/particle_spacing):round(Int, R/particle_spacing)

            x = x_center + i * particle_spacing
            y = y_center + j * particle_spacing

        # recess conditions
        recess     = y_recess[2] >= y >= y_recess[1] && x_recess[2] >= x >= x_recess[1]
        if r(x,y) < R && !recess
            append!(x_vec, x)
            append!(y_vec, y)
        end
    end

    particle_coords = Array{Float64, 2}(undef, 2, size(x_vec,1))
    particle_coords[1,:] = x_vec
    particle_coords[2,:] = y_vec

    return particle_coords
end
