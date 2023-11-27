@testset verbose=true "trixi_include" begin
    @trixi_testset "Basic" begin
        example = """
            x = 4
            """

        filename = tempname()
        try
            open(filename, "w") do file
                write(file, example)
            end

            # Use `@trixi_testset`, which wraps code in a temporary module, and call
            # `trixi_include` with `@__MODULE__` in order to isolate this test.
            @test_nowarn trixi_include(@__MODULE__, filename)
            @test @isdefined x
            @test x == 4

            @test_nowarn trixi_include(@__MODULE__, filename, x=7)
            @test x == 7

            @test_throws "assignment y not found in expression" trixi_include(@__MODULE__,
                                                                              filename, y=3)
        finally
            rm(filename, force=true)
        end
    end

    @trixi_testset "With `solve` Without `maxiters`" begin
        example = """
            solve() = 0
            x = solve()
            """

        filename = tempname()
        try
            open(filename, "w") do file
                write(file, example)
            end

            # Use `@trixi_testset`, which wraps code in a temporary module, and call
            # `trixi_include` with `@__MODULE__` in order to isolate this test.
            @test_throws "no method matching solve(; maxiters::Int64)" trixi_include(@__MODULE__,
                                                                                     filename)

            @test_throws "no method matching solve(; maxiters::Int64)" trixi_include(@__MODULE__,
                                                                                     filename,
                                                                                     maxiters=3)
        finally
            rm(filename, force=true)
        end
    end

    @trixi_testset "With `solve` with `maxiters`" begin
        # We need another example file that we include with `Base.include` first, in order to
        # define the `solve` method without `trixi_include` trying to insert `maxiters` kwargs.
        example1 = """
            solve(; maxiters=0) = maxiters
            """

        example2 = """
            x = solve()
            """

        filename1 = tempname()
        filename2 = tempname()
        try
            open(filename1, "w") do file
                write(file, example1)
            end
            open(filename2, "w") do file
                write(file, example2)
            end

            # Use `@trixi_testset`, which wraps code in a temporary module, and call
            # `Base.include` and `trixi_include` with `@__MODULE__` in order to isolate this test.
            Base.include(@__MODULE__, filename1)
            @test_nowarn trixi_include(@__MODULE__, filename2)
            @test @isdefined x
            @test x == 10^5

            @test_nowarn trixi_include(@__MODULE__, filename2, maxiters=7)
            @test x == 7
        finally
            rm(filename1, force=true)
            rm(filename2, force=true)
        end
    end
end
