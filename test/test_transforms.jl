bs1 = SpinBasis(1 // 2)
bs2 = SpinBasis(2 // 2)

@test_throws IncompatibleBases StateTransform(spinup(bs1) => spinup(bs2))
@test_throws IncompatibleBases UnitaryTransform(spinup(bs1) => spinup(bs2))

t = UnitaryTransform(spinup(bs1) => spindown(bs1))
@test_throws IncompatibleBases global t += spindown(bs1) => spinup(bs2)
@test_throws ArgumentError UnitaryTransform(
    [spinup(bs1), spindown(bs1)],
    [[1.0, 2.0][3.0, 4.0]],
)

@test_throws ArgumentError UnitaryTransform(
    [spinup(bs1), spindown(bs1)],
    [[1.0 0.0]; [0.0 2.0]],
)
@test_throws DimensionMismatch UnitaryTransform(
    [spinup(bs1), spindown(bs1)],
    [[1.0 0.0 1.0]; [0.0 2.0 1.0]],
)
@test_throws IncompatibleBases UnitaryTransform(
    [spinup(bs1), spindown(bs2)],
    [[1.0 0.0]; [0.0 1.0]],
)
