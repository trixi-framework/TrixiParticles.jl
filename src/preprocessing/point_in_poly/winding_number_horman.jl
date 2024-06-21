"""
    WindingNumberHorman()

Algorithm for inside-outside segmentation of a complex shape proposed by Horman et al. (2001).
It is only supported for 2D shapes.
[`WindingNumberHorman`](@rerf) might handle edge cases a bit better, since the winding number is an integer value.
Also, it is faster than [`WindingNumberJacobson`](@ref) for 2D shapes with about more than 100 edges.
"""
struct WindingNumberHorman end

function (point_in_poly::WindingNumberHorman)(shape, points; store_winding_number=false)
    (; edge_vertices) = shape

    inpoly = falses(size(points, 2))

    winding_numbers = Float64[]
    store_winding_number && (winding_numbers = resize!(winding_numbers, length(inpoly)))

    h_unit = SVector(1.0, 0.0)
    v_unit = SVector(0.0, 1.0)

    function quadrant(dot_v, dot_h)
        (dot_v >= 0.0 && dot_h > 0.0) && return 0
        (dot_v > 0.0 && dot_h <= 0.0) && return 1
        (dot_v <= 0.0 && dot_h < 0.0) && return 2
        (dot_v < 0.0 && dot_h >= 0.0) && return 3
    end

    @threaded points for query_point in axes(points, 2)
        winding_number = 0
        v_query = point_position(points, shape, query_point)

        for edge in eachface(shape)
            v1 = edge_vertices[edge][1]
            v2 = edge_vertices[edge][2]

            direction1 = v1 - v_query
            direction2 = v2 - v_query

            dot_v1 = dot(direction1, v_unit)
            dot_h1 = dot(direction1, h_unit)
            dot_v2 = dot(direction2, v_unit)
            dot_h2 = dot(direction2, h_unit)

            # because 0 <= `quadrant_numbers` <= 3 we know that -3 <= `quarter_angel` <= 3
            quarter_angel = quadrant(dot_v2, dot_h2) - quadrant(dot_v1, dot_h1)

            if quarter_angel == -3
                winding_number += 1
            elseif quarter_angel == 3
                winding_number -= 1
            elseif quarter_angel == -2 && positive_determinant(v1, v2, v_query)
                winding_number += 1
            elseif quarter_angel == 2 && !positive_determinant(v1, v2, v_query)
                winding_number -= 1
            end
        end

        store_winding_number && (winding_numbers[query_point] = winding_number)

        !(winding_number == 0) && (inpoly[query_point] = true)
    end

    return inpoly, winding_numbers
end

function positive_determinant(v1, v2, v_query)

    # Check the orientation of the triangle Δ(`v_query`, `v1`, `v2`) by finding the
    # sign of the determinant.
    # `+`: counter clockwise
    # `-`: clockwise
    positive_sign = (v1[1] - v_query[1]) * (v2[2] - v_query[2]) -
                    (v2[1] - v_query[1]) * (v1[2] - v_query[2]) > 0.0

    return positive_sign
end
