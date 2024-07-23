"""
    WindingNumberHorman()

Algorithm for inside-outside segmentation of a complex geometry proposed by [Horman et al. (2001)](@ref references_complex_shape).
It is only supported for 2D geometries.
[`WindingNumberHorman`](@ref) might handle edge cases a bit better, since the winding number is an integer value.
Also, it is faster than [`WindingNumberJacobson`](@ref) for 2D geometries with more than about 100 edges.


!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct WindingNumberHorman end

# Algorithm 2 from Horman et al. (2001) "The point in polygon problem for arbitrary polygons"
# https://doi.org/10.1016/S0925-7721(01)00012-8
function (point_in_poly::WindingNumberHorman)(geometry, points; store_winding_number=false)
    (; edge_vertices) = geometry

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
        v_query = point_position(points, geometry, query_point)

        for edge in eachface(geometry)
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
