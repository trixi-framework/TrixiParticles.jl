using KernelAbstractions
using StaticArrays
using Adapt
using Metal
using Random: seed!

@inline function apply_ith_function(functions, index, args...)
    if index == 1
        # Found the function to apply, apply it and return
        return first(functions)(args...)
    end

    # Process remaining functions
    apply_ith_function(Base.tail(functions), index - 1, args...)
end

@kernel function mykernel(functions, a, b, t, indices)
    i = @index(Global)

    index = indices[i]
    a[i] = apply_ith_function(functions, index, b[i], t[i])
end

n = 10
backend = MetalBackend()

pressure_functions = (@inline((x, t)->x), @inline((x, t)->2 * x), @inline((x, t)->3 * x))
velocity_functions = (@inline((x, t)->x .^ 2 .+ t),
                      @inline((x, t)->SVector(2 * x[2], 3 * x[1]) .+ t),
                      @inline((x, t)->2 * x * t))

a = fill(SVector{2, Float32}(0.0, 0.0), n)
b = fill(SVector{2, Float32}(2.0, 2.0), n)
t = ones(Float32, n)

seed!(1)
indices = Int32.(rand(1:length(pressure_functions), n))

a_gpu = Adapt.adapt(backend, a)
b_gpu = Adapt.adapt(backend, b)
t_gpu = Adapt.adapt(backend, t)
indices_gpu = Adapt.adapt(backend, indices)

mykernel(backend)(pressure_functions, a_gpu, b_gpu, t_gpu, indices_gpu, ndrange=n)
a_gpu

using KernelAbstractions
using Atomix
using Adapt
using Metal

@kernel function mykernel(inc, input, values)
    i = @index(Global)

    j = Atomix.@atomic inc[1] += 1

    values[j] = input[i]
end

n = 10
backend = MetalBackend()

input = Adapt.adapt(backend, rand(Int32, n))
values = Adapt.adapt(backend, zeros(Int32, n))
inc = Adapt.adapt(backend, zeros(Int32, 1))

mykernel(backend)(inc, input, values, ndrange=n)

# ==========================================================================================

function map_sort!(a, b) # (Vector{Int}, Vector{Bool})
    map!((active, idx) -> active ? -1 : idx, a, b, eachindex(b))

    # TODO
    sort!(a, rev=true)

    return a # [idx_1, idx_2, ..., -1, -1]
end

Threads.@threads for particle in eachparticle
    if is_outside(particle)
        outside_particles[particle] = true
    end
end

map_sort!(candidates, outside_particles) # [candidate_1, candidate_2, ..., -1, -1]
map_sort!(available_particles, active_particles) # [inactive_1, inactive_2, ..., -1, -1]

next_particle[1] = 0
Threads.@threads for i in 1:count(outside_particles)
    particle = candidates[i]

    next_id = PointNeighbors.Atomix.@atomic next_particle[1] += 1
    new_particle = available_particles[next_id]

    copy_data(particle, new_particle)
end

using Metal
using Adapt

n_true = n_false = 100_000;
a_cpu = vcat(fill(UInt32(true), n_true), fill(UInt32(false), n_false));
a_gpu = Adapt.adapt(MetalBackend(), a_cpu);

@show length(findall(x -> x == true, a_cpu)) # 100_000
@show length(findall(x -> x == true, a_gpu)) # 100_000

n_true = n_false = 10_000_000;
a_cpu = vcat(fill(UInt32(true), n_true), fill(UInt32(false), n_false));
a_gpu = Adapt.adapt(MetalBackend(), a_cpu);

@show length(findall(x -> x == true, a_cpu)) # 1_000_000
@show length(findall(x -> x == true, a_gpu)) # 999_988

# 999_988
