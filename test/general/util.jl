@testset verbose=true "foreach_noalloc" begin
    visited = Int[]
    returned = TrixiParticles.foreach_noalloc((1, 2, 3)) do element
        push!(visited, element)
        return nothing
    end
    @test returned === nothing
    @test visited == [1, 2, 3]

    visited_pairs = Tuple{Int, Symbol}[]
    returned = TrixiParticles.foreach_noalloc((1, 2), (:first, :second)) do pair
        push!(visited_pairs, pair)
        return nothing
    end
    @test returned === nothing
    @test visited_pairs == [(1, :first), (2, :second)]

    called = Ref(false)
    returned = TrixiParticles.foreach_noalloc(()) do element
        called[] = true
        return nothing
    end
    @test returned === nothing
    @test !called[]

    returned = TrixiParticles.foreach_noalloc((), ()) do pair
        called[] = true
        return nothing
    end
    @test returned === nothing
    @test !called[]
end

@testset verbose=true "ThreadedBroadcastArray" begin
    A = TrixiParticles.ThreadedBroadcastArray(ones(3, 3))
    B = ones(3, 3)

    # Test that all of these operations work
    @trixi_test_nowarn A .* 2
    @trixi_test_nowarn A .+ B
    @trixi_test_nowarn B .+ A
    @trixi_test_nowarn A .= 0
    @trixi_test_nowarn A .+= 0
    @trixi_test_nowarn A .= B
    @trixi_test_nowarn A .= A .+ B
    @trixi_test_nowarn A .= B .+ A
    @trixi_test_nowarn A .= A .* 2
    @trixi_test_nowarn A .= B .* 2
    @trixi_test_nowarn copyto!(A, B)
    @trixi_test_nowarn copyto!(A, TrixiParticles.ThreadedBroadcastArray(B))

    copyto!(A, TrixiParticles.ThreadedBroadcastArray(fill(2.0, 3, 3)))
    @test all(parent(A) .== 2)

    # Test that the resulting type of broadcasting is correct
    @test typeof(A .* 2) == typeof(A)
    @test typeof(A .+ B) == typeof(A)
    @test typeof(B .+ A) == typeof(A)

    # Test that the resulting type of `similar` is correct
    C = similar(A, Float64, (2, 2))
    @test typeof(C) == typeof(A)
    @test size(C) == (2, 2)
    @test typeof(similar(A, Float64)) == typeof(A)
    C = similar(A, (2, 2))
    @test typeof(C) == typeof(A)
    @test size(C) == (2, 2)
    @test typeof(similar(A)) == typeof(A)

    # Test that these operations all use the correct backend
    struct FailingBackend end

    # Define `@threaded` to fail with backend `FailingBackend`
    function TrixiParticles.PointNeighbors.parallel_foreach(f, iterator, ::FailingBackend)
        error("test1")
    end

    A2 = TrixiParticles.ThreadedBroadcastArray(ones(3, 3),
                                               parallelization_backend=FailingBackend())

    # Test that all of these operations fail (which means they are using `@threaded`)
    @test_throws "test1" A2 .* 2
    @test_throws "test1" A2 .+ B
    @test_throws "test1" B .+ A2
    @test_throws "test1" A2 .= 0
    @test_throws "test1" A2 .+= 0
    @test_throws "test1" A2 .= B
    @test_throws "test1" A2 .= A2 .+ B
    @test_throws "test1" A2 .= B .+ A2
    @test_throws "test1" A2 .= A2 .* 2
    @test_throws "test1" A2 .= B .* 2
    @test_throws "test1" copyto!(A2, B)
    @test_throws "test1" copyto!(A2, A)
end
