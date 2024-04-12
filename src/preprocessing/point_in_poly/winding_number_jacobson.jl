# Alec Jacobson, Ladislav Kavan, and Olga Sorkine-Hornung. 2013.
# Robust inside-outside segmentation using generalized winding numbers.
# ACM Trans. Graph. 32, 4, Article 33 (July 2013), 12 pages.
# https://doi.org/10.1145/2461912.2461916

# ALterantive implementation: https://github.com/JuliaGeometry/Meshes.jl/blob/3e5272392ca917668c7ec3844a14b325c7568e31/src/winding.jl
struct WindingNumberJacobson{ELTYPE}
    winding_number_factor::ELTYPE

    function WindingNumberJacobson(; winding_number_factor=sqrt(eps()))
        ELTYPE = typeof(winding_number_factor)
        return new{ELTYPE}(winding_number_factor)
    end
end

function (point_in_poly::WindingNumberJacobson)(mesh, points)
    (; winding_number_factor) = point_in_poly
    (; faces) = mesh
    inpoly = falses(size(points, 2))

    @threaded for query_point in axes(points, 2)
        p = point_position(points, mesh, query_point)

        winding_number = sum(faces) do face
            return inverse_tangent_function(face, p)
        end

        winding_number /= pi_factor(Val(ndims(mesh)))

        # `(winding_number != 0.0)`
        if !(-winding_number_factor < winding_number < winding_number_factor)
            inpoly[query_point] = true
        end
    end

    return inpoly
end

# A. Van Oosterom 1983,
# The Solid Angle of a Plane Triangle (doi: 10.1109/TBME.1983.325207)
function inverse_tangent_function(face, p::SVector{3})
    v1, v2, v3 = face
    a = v1 - p
    b = v2 - p
    c = v3 - p
    a_ = norm(a)
    b_ = norm(b)
    c_ = norm(c)

    divisor = a_ * b_ * c_ + dot(a, b) * c_ + dot(b, c) * a_ + dot(c, a) * b_

    return 2atan(det([a b c]), divisor)
end

function inverse_tangent_function(edge, p::SVector{2})
    v1, v2 = edge
    a = v1 - p
    b = v2 - p

    return atan(det([a b]), (dot(a, b)))
end

@inline pi_factor(::Val{3}) = 4pi
@inline pi_factor(::Val{2}) = 2pi
