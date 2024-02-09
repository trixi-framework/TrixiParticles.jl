using TrixiParticles, Test, LinearAlgebra
@testset verbose=true "OpenBoundarySPHSystem" begin
    @testset verbose=true "Boundary Zone 2D" begin
        boundary_zones = [InFlow(), OutFlow()]
        particle_spacing = 0.2
        open_boundary_layers = 4

        point1s = [[0.0, 0.0], [0.5, -0.5], [1.0, 0.5]]
        point2s = [[0.0, 1.0], [0.2, 2.0], [2.3, 0.5]]

        @testset "Boundary Zone $boundary_zone" for boundary_zone in boundary_zones
            @testset "Points $(i)" for i in eachindex(point1s)
                plane_size = point2s[i] - point1s[i]
                flow_directions = [
                    normalize([-plane_size[2], plane_size[1]]),
                    -normalize([-plane_size[2], plane_size[1]]),
                ]

                plane_points = [point1s[i], point2s[i]]

                @testset "Flow Direction $(j)" for j in eachindex(flow_directions)
                    system = OpenBoundarySPHSystem(plane_points, boundary_zone, 1.0;
                                                   flow_direction=flow_directions[j],
                                                   particle_spacing, open_boundary_layers,
                                                   density=1.0)
                    zone_width = open_boundary_layers *
                                 system.initial_condition.particle_spacing
                    sign_ = (boundary_zone isa InFlow) ? -1 : 1

                    @test point1s[i] == system.zone_origin
                    @test point2s[i] - system.zone_origin == system.spanning_set[2]
                    @test sign_ * flow_directions[j] ≈ normalize(system.spanning_set[1])
                    @test zone_width ≈ norm(system.spanning_set[1])
                end
            end
        end
    end
end


points = (system.zone_origin,
system.zone_origin + system.spanning_set[2],
system.zone_origin + system.spanning_set[1])
