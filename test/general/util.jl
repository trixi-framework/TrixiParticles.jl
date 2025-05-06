@testset verbose=true "ThreadedBroadcastArray" begin
    A = TrixiParticles.ThreadedBroadcastArray(ones(3, 3))
    B = ones(3, 3)

    # Test that all of these operations work
    @test_nowarn_mod A .* 2
    @test_nowarn_mod A .+ B
    @test_nowarn_mod B .+ A
    @test_nowarn_mod A .= 0
    @test_nowarn_mod A .+= 0
    @test_nowarn_mod A .= B
    @test_nowarn_mod A .= A .+ B
    @test_nowarn_mod A .= B .+ A
    @test_nowarn_mod A .= A .* 2
    @test_nowarn_mod A .= B .* 2

    # Redefine `@threaded` to fail with `Array`s
    default_backend = TrixiParticles.PointNeighbors.default_backend(B)
    TrixiParticles.PointNeighbors.default_backend(::Array) = Array
    function TrixiParticles.PointNeighbors.parallel_foreach(f, iterator, ::Type{Array})
        error("test1")
    end

    # Test that all of these operations fail (which means they are using `@threaded`)
    @test_throws "test1" A .* 2
    @test_throws "test1" A .+ B
    @test_throws "test1" B .+ A
    @test_throws "test1" A .= 0
    @test_throws "test1" A .+= 0
    @test_throws "test1" A .= B
    @test_throws "test1" A .= A .+ B
    @test_throws "test1" A .= B .+ A
    @test_throws "test1" A .= A .* 2
    @test_throws "test1" A .= B .* 2

    # Restore `@threaded` to work with `Array`s again
    TrixiParticles.PointNeighbors.default_backend(::Array) = default_backend
end
