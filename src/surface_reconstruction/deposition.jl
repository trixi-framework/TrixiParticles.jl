# Bilinear/trilinear cloud-in-cell (CIC) deposition of particle measures.

# Keep each deposition kernel's arithmetic and boundary handling: planar accumulation
# uses area/spacing², while the 3D kernel multiplies by the inverse cell volume and can
# deposit a partial stencil when called directly. The public pipeline excludes partial
# stencils in both dimensions before calling these kernels.
function deposit_volume_cic!(field::AbstractMatrix, points, areas, origin, spacing;
                             backend=PolyesterBackend(), support=nothing)
    fill!(field, 0)
    inverse_spacing = inv(spacing)
    for particle in axes(points, 2)
        coordinate = (SVector{2, Float64}(points[:, particle]) - origin) * inverse_spacing
        lower = floor.(Int, coordinate)
        fraction = coordinate - lower
        all((0 .<= lower) .& (lower .< SVector(size(field)) .- 1)) ||
            throw(ArgumentError("particle stencil is outside the deposition grid"))
        for y_offset in 0:1, x_offset in 0:1
            weight_x = x_offset == 0 ? 1 - fraction[1] : fraction[1]
            weight_y = y_offset == 0 ? 1 - fraction[2] : fraction[2]
            field[lower[1] + x_offset + 1,
                  lower[2] + y_offset + 1] += Float32(areas[particle] / spacing^2 *
                                                      weight_x * weight_y)
        end
    end
    return (; deposited_volume=sum(Float64, field) * spacing^2,
            particle_volume=sum(areas))
end

# Voxel ranges that `deposit_volume_cic!` can write to: the CIC stencils of all
# particles, with the same index arithmetic, clamped to the grid
function deposition_support(points, origin, spacing, dimensions::NTuple{N, Int}) where {N}
    size(points, 2) == 0 && return ntuple(_ -> 1:0, Val(N))
    inverse_spacing = inv(spacing)
    lower = MVector{N, Int}(ntuple(_ -> typemax(Int), Val(N)))
    upper = MVector{N, Int}(ntuple(_ -> typemin(Int), Val(N)))
    @inbounds for particle in axes(points, 2), axis in 1:N
        cell = floor(Int, (points[axis, particle] - origin[axis]) * inverse_spacing)
        lower[axis] = min(lower[axis], cell)
        upper[axis] = max(upper[axis], cell)
    end
    # 0-based stencil cells `cell` and `cell + 1` are the 1-based voxels `cell + 1:cell + 2`
    return ntuple(axis -> max(lower[axis] + 1, 1):min(upper[axis] + 2, dimensions[axis]),
                  Val(N))
end

# With `support` (see `deposition_support`), `field` must be zero on entry, and the
# deposited volume is summed over `support` only. Otherwise, `field` is zeroed first.
function deposit_volume_cic!(field, points, volumes, origin, spacing;
                             backend=PolyesterBackend(), support=nothing)
    isnothing(support) && parallel_fill!(field, 0; backend)
    nx, ny, nz = size(field)
    inverse_spacing = inv(spacing)
    inverse_cell_volume = inv(spacing^3)
    @inbounds for particle in axes(points, 2)
        cx = (points[1, particle] - origin[1]) * inverse_spacing
        cy = (points[2, particle] - origin[2]) * inverse_spacing
        cz = (points[3, particle] - origin[3]) * inverse_spacing
        lower_x = floor(Int, cx)
        lower_y = floor(Int, cy)
        lower_z = floor(Int, cz)
        fx = cx - lower_x
        fy = cy - lower_y
        fz = cz - lower_z
        scaled_volume = volumes[particle] * inverse_cell_volume
        for z_offset in 0:1
            grid_z = lower_z + z_offset
            0 <= grid_z < nz || continue
            wz = z_offset == 0 ? 1 - fz : fz
            for y_offset in 0:1
                grid_y = lower_y + y_offset
                0 <= grid_y < ny || continue
                wy = y_offset == 0 ? 1 - fy : fy
                for x_offset in 0:1
                    grid_x = lower_x + x_offset
                    0 <= grid_x < nx || continue
                    wx = x_offset == 0 ? 1 - fx : fx
                    field[grid_x + 1, grid_y + 1,
                          grid_z + 1] += Float32(scaled_volume * wx * wy * wz)
                end
            end
        end
    end
    field_sum, _,
    _ = field_sum_extrema(field; backend,
                          region=something(support, axes(field)))
    return (deposited_volume=field_sum * spacing^3,
            particle_volume=sum(volumes))
end
