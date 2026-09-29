@testset verbose=true "foreach_noalloc" begin
    collection1 = (1, 2)
    visited = [0, 0]
    TrixiParticles.foreach_noalloc(collection1) do collection
        visited[collection] += 1
    end
    @test visited == [1, 1]

    collection2 = (3, 4)
    visited = []
    TrixiParticles.foreach_noalloc_zip(collection1, collection2) do (i, j)
        push!(visited, (i, j))
    end
    @test visited == [(1, 3), (2, 4)]
end

@testset verbose=true "copyto_threaded!" begin
    backend = SerialBackend()

    matrix = zeros(2, 3)
    TrixiParticles.copyto_threaded!(matrix, reshape(1:6, 2, 3), backend)
    @test matrix == reshape(1:6, 2, 3)
    @test_throws DimensionMismatch TrixiParticles.copyto_threaded!(matrix, zeros(2, 2),
                                                                   backend)

    vector = zeros(3)
    TrixiParticles.copyto_threaded!(vector, 1:3, backend)
    @test vector == 1:3
    @test_throws DimensionMismatch TrixiParticles.copyto_threaded!(vector, zeros(2),
                                                                   backend)
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
