struct SeparateClusters

    function SeparateClusters(systems)
        for system in systems
            create_cluster(system)
        end
    end
end
