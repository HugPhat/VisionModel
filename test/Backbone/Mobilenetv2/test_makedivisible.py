def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

testset = [16, 24, 32, 64, 96, 160, 320]

for item in testset:
    print(f'[{item}]<=> {_make_divisible(item, 8)}')