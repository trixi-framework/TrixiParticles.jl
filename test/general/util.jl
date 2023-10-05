@testset verbose=true "trixi_include" begin
    example = """
        x = 4
        """

    filename = tempname()
    try
        open(filename, "w") do file
            write(file, example)
        end

        @test_nowarn trixi_include(filename)
        @test @isdefined x
        @test x == 4

        @test_nowarn trixi_include(filename, x=7)
        @test x == 7

        @test_throws "assignment y not found in expression" trixi_include(filename, y=3)
    finally
        rm(filename, force=true)
    end
end
