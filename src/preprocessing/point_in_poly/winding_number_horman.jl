# Kai Horman, Alexander Agathos
# The point in polygon problem for arbitrary polygons
# Computational Geometry 2001
# doi: 10.1016/s0925-7721(01)00012-8
struct WindingNumberHorman end

# Only for 2D yet.
function (point_in_poly::WindingNumberHorman)(shape, points)
    (; vertices, n_vertices) = shape
    quadrant_numbers = zeros(Int, n_vertices)

    inpoly = falses(size(points, 2))

    h_unit = SVector(1.0, 0.0)
    v_unit = SVector(0.0, 1.0)

    function quadrant(dot_v, dot_h)
        (dot_v >= 0.0 && dot_h > 0.0) && return 0
        (dot_v > 0.0 && dot_h <= 0.0) && return 1
        (dot_v <= 0.0 && dot_h < 0.0) && return 2
        (dot_v < 0.0 && dot_h >= 0.0) && return 3
    end

    @threaded for query_point in axes(points, 2)
        for vertex in eachvertices(shape)
            direction = point_position(vertices, shape, vertex) -
                        point_position(points, shape, query_point)

            dot_v = dot(direction, v_unit)
            dot_h = dot(direction, h_unit)

            quadrant_numbers[vertex] = quadrant(dot_v, dot_h)
        end

        # the last vertex is the same as the first one.
        quadrant_numbers[end] = quadrant_numbers[1]

        winding_number = 0
        for vertex in eachvertices(shape)
            v1 = point_position(vertices, shape, vertex)
            v2 = point_position(vertices, shape, vertex + 1)
            v_query = point_position(points, shape, query_point)

            # because 0 <= `quadrant_numbers` <= 3 we know that -3 <= `quarter_angel` <= 3
            quarter_angel = quadrant_numbers[vertex + 1] - quadrant_numbers[vertex]
            positive_det = positive_determinant(v1, v2, v_query)

            if quarter_angel == -3
                winding_number += 1
            elseif quarter_angel == 3
                winding_number -= 1
            elseif quarter_angel == -2 && positive_det
                winding_number += 1
            elseif quarter_angel == 2 && !positive_det
                winding_number -= 1
            end
        end

        !(winding_number == 0) && (inpoly[query_point] = true)
    end

    return inpoly
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
