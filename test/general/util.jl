@testset verbose=true "extract_smatrix" begin
    A = Float64.(collect(reshape(1:12, 2, 2, 3)))
    @test TrixiParticles.extract_smatrix(A, Val(2), 1) == A[:, :, 1]
    @test TrixiParticles.extract_smatrix(A, Val(2), 2) == A[:, :, 2]
    @test TrixiParticles.extract_smatrix(A, Val(2), 3) == A[:, :, 3]
    @test_throws "extract_smatrix only works" TrixiParticles.extract_smatrix(A, Val(1), 1)
    @test_throws "BoundsError" TrixiParticles.extract_smatrix(A, Val(3), 1)
end

@testset verbose=true "extract_svector" begin
    A = Float64.(collect(reshape(1:9, 3, 3)))
    @test TrixiParticles.extract_svector(A, Val(3), 1) == A[:, 1]
    @test TrixiParticles.extract_svector(A, Val(3), 2) == A[:, 2]
    @test TrixiParticles.extract_svector(A, Val(3), 3) == A[:, 3]
    @test TrixiParticles.extract_svector(A, Val(2), 1) == A[1:2, 1]
    @test TrixiParticles.extract_svector(A, Val(2), 2) == A[1:2, 2]
    @test TrixiParticles.extract_svector(A, Val(2), 3) == A[1:2, 3]
    @test_throws "BoundsError" TrixiParticles.extract_svector(A, Val(4), 1)
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
end
