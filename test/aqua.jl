using Aqua
using TrixiParticles

const TEST_AMBIGUITIES = let
    value = lowercase(get(ENV, "TRIXIPARTICLES_TEST_AMBIGUITIES", "false"))
    value in ("1", "true")
end

Aqua.test_all(TrixiParticles; ambiguities=TEST_AMBIGUITIES)
