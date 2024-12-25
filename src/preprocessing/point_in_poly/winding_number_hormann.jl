"""
    WindingNumberHormann()

Algorithm for inside-outside segmentation of a complex geometry proposed by [Hormann (2001)](@cite Hormann2001).
It is only supported for 2D geometries.
`WindingNumberHormann` might handle edge cases a bit better, since the winding number is an integer value.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct WindingNumberHormann end

# Algorithm 2 from Hormann et al. (2001) "The point in polygon problem for arbitrary polygons"
# https://doi.org/10.1016/S0925-7721(01)00012-8
function (point_in_poly::WindingNumberHormann)(geometry, points; store_winding_number=false)
    (; edge_vertices) = geometry

    # We cannot use a `BitVector` here, as writing to a `BitVector` is not thread-safe
    inpoly = fill(false, length(points))

    winding_numbers = Float64[]
    store_winding_number && (winding_numbers = resize!(winding_numbers, length(inpoly)))

    function quadrant(v, p)
        (v[1] > p[1] && v[2] >= p[2]) && return 0
        (v[1] <= p[1] && v[2] > p[2]) && return 1
        (v[1] < p[1] && v[2] <= p[2]) && return 2
        (v[1] >= p[1] && v[2] < p[2]) && return 3
    end

    @threaded points for query_point in eachindex(points)
        v_query = points[query_point]
        winding_number = 0

        for edge in eachface(geometry)
            v1 = edge_vertices[edge][1]
            v2 = edge_vertices[edge][2]

            # because 0 <= `quadrant_numbers` <= 3 we know that -3 <= `quarter_angle` <= 3
            quarter_angle = quadrant(v2, v_query) - quadrant(v1, v_query)

            direction_1 = v1 - v_query
            direction_2 = v2 - v_query

            # Check the orientation of the triangle Δ(`v_query`, `v1`, `v2`) by finding the
            # sign of the determinant.
            # `+`: counter clockwise
            # `-`: clockwise
            if quarter_angle == -3
                winding_number += 1
            elseif quarter_angle == 3
                winding_number -= 1
            elseif quarter_angle == -2 && det([direction_1 direction_2]) > 0
                winding_number += 1
            elseif quarter_angle == 2 && det([direction_1 direction_2]) < 0
                winding_number -= 1
            end
        end

        store_winding_number && (winding_numbers[query_point] = winding_number)

        winding_number != 0 && (inpoly[query_point] = true)
    end

    return inpoly, winding_numbers
end
