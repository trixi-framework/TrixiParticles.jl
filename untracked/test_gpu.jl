using KernelAbstractions
using Metal

@kernel function mykernel(a)
    i = @index(Global)

    # x = Ref(1)
    x = Ref(1)

    f() = x[] += 2
    f()

    a[i] = x[]
end

backend = MetalBackend()

a = MtlVector([1])

mykernel(backend)(a, ndrange=1)
