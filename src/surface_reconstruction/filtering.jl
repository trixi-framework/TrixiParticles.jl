# Separable Gaussian filter for the deposited volume field.
function gaussian_kernel(sigma_voxels)
    radius = floor(Int, GAUSSIAN_TRUNCATE * sigma_voxels + 0.5)
    weights = [exp(-0.5 * (offset / sigma_voxels)^2) for offset in (-radius):radius]
    weights ./= sum(weights)
    return Float32.(weights)
end

function convolve_x!(output, input, kernel, backend;
                     region=(axes(input, 1), axes(input, 2), axes(input, 3)),
                     zeroed=false)
    nx, ny, nz = size(input)
    radius = length(kernel) >>> 1
    xs, ys, zs = region
    region_is_grid = region == axes(input)
    region_is_grid || zeroed || parallel_fill!(output, 0.0f0; backend)
    @threaded backend for line in 0:(length(ys) * length(zs) - 1)
        y = rem(line, length(ys)) + first(ys)
        z = fld(line, length(ys)) + first(zs)
        @inbounds for x in xs
            value = 0.0f0
            first_offset = max(-radius, 1 - x)
            last_offset = min(radius, nx - x)
            @simd for offset in first_offset:last_offset
                value = muladd(kernel[offset + radius + 1], input[x + offset, y, z], value)
            end
            output[x, y, z] = value
        end
    end
    return output
end

function convolve_y!(output, input, kernel, backend;
                     region=(axes(input, 1), axes(input, 2), axes(input, 3)),
                     zeroed=false)
    nx, ny, nz = size(input)
    radius = length(kernel) >>> 1
    xs, ys, zs = region
    region_is_grid = region == axes(input)
    region_is_grid || zeroed || parallel_fill!(output, 0.0f0; backend)
    @threaded backend for line in 0:(length(xs) * length(zs) - 1)
        x = rem(line, length(xs)) + first(xs)
        z = fld(line, length(xs)) + first(zs)
        @inbounds for y in ys
            value = 0.0f0
            first_offset = max(-radius, 1 - y)
            last_offset = min(radius, ny - y)
            @simd for offset in first_offset:last_offset
                value = muladd(kernel[offset + radius + 1], input[x, y + offset, z], value)
            end
            output[x, y, z] = value
        end
    end
    return output
end

function convolve_z!(output, input, kernel, backend;
                     region=(axes(input, 1), axes(input, 2), axes(input, 3)),
                     zeroed=false)
    nx, ny, nz = size(input)
    radius = length(kernel) >>> 1
    xs, ys, zs = region
    region_is_grid = region == axes(input)
    region_is_grid || zeroed || parallel_fill!(output, 0.0f0; backend)
    @threaded backend for line in 0:(length(xs) * length(ys) - 1)
        x = rem(line, length(xs)) + first(xs)
        y = fld(line, length(xs)) + first(ys)
        @inbounds for z in zs
            value = 0.0f0
            first_offset = max(-radius, 1 - z)
            last_offset = min(radius, nz - z)
            @simd for offset in first_offset:last_offset
                value = muladd(kernel[offset + radius + 1], input[x, y, z + offset], value)
            end
            output[x, y, z] = value
        end
    end
    return output
end

# `support` optionally bounds the nonzero values of `field` (e.g. the CIC stencils of all
# particles). Outputs are then computed only where they can be nonzero — the region grows
# by the kernel radius along each pass's axis — and set to zero elsewhere. Every computed
# output uses the same arithmetic as on the full grid, and outputs outside the region are
# sums of exact zeros, so the result is bitwise identical.
# `zeroed=true` asserts that `temporary` and `scratch` are zero, which skips zeroing the
# outputs outside their regions (the regions cover `support`, so `field` is overwritten
# wherever it is nonzero).
function gaussian_filter!(field, temporary, scratch, kernel; backend=PolyesterBackend(),
                          support=nothing, zeroed=false)
    if support === nothing
        convolve_x!(temporary, field, kernel, backend)
        convolve_y!(scratch, temporary, kernel, backend)
        convolve_z!(field, scratch, kernel, backend)
        return field
    end

    radius = length(kernel) >>> 1
    expand(range, n) = max(first(range) - radius, 1):min(last(range) + radius, n)
    xs, ys, zs = support
    nx, ny, nz = size(field)
    xs_filtered, ys_filtered, zs_filtered = expand(xs, nx), expand(ys, ny), expand(zs, nz)
    convolve_x!(temporary, field, kernel, backend; region=(xs_filtered, ys, zs), zeroed)
    convolve_y!(scratch, temporary, kernel, backend;
                region=(xs_filtered, ys_filtered, zs), zeroed)
    convolve_z!(field, scratch, kernel, backend;
                region=(xs_filtered, ys_filtered, zs_filtered), zeroed)
    return field
end
