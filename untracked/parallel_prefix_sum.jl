using KernelAbstractions
using Metal
using Adapt
using Random
using Polyester
using BenchmarkTools

@kernel function mykernel!(active, active_indices, prefix)
    i = @index(Global)
    if active[i] == true
        active_indices[prefix[i]] = i
    end
end

function my_findall(active_indices, active)
    prefix = cumsum(active)

    # Afterwards used to iterate only over computed indices: `active_indices[1:total]`
    total = Metal.@allowscalar prefix[end]

    mykernel!(backend)(active, active_indices, prefix, ndrange=length(active))

    return total
end

function my_findall_cpu(active_indices, active)
    prefix = cumsum(active)

    # Afterwards used to iterate only over computed indices: `active_indices[1:total]`
    total = prefix[end]
    Polyester.@batch for i in eachindex(active)
        if active[i] == true
            active_indices[prefix[i]] = i
        end
    end

    return total
end

n = 100_000_000
backend = MetalBackend()

active_indices_cpu = rand(Int, n)
active_indices = Adapt.adapt(backend, active_indices_cpu)

Random.seed!(1)
active_cpu = rand(UInt32.(0:1), n)
active = Adapt.adapt(backend, active_cpu)
active_bool = map(x -> x == true, active)

@benchmark my_findall($active_indices, $active)
@benchmark my_findall_cpu($active_indices_cpu, $active_cpu)

f(x) = x == true
@benchmark findall($f, $active)
@benchmark findall($f, $active_cpu)

@benchmark findall($active_bool)
