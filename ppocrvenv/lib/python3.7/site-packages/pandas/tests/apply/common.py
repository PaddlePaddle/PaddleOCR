from pandas.core.groupby.base import transformation_kernels

# tshift only works on time index and is deprecated
# There is no Series.cumcount or DataFrame.cumcount
series_transform_kernels = [
    x for x in sorted(transformation_kernels) if x not in ["tshift", "cumcount"]
]
frame_transform_kernels = [
    x for x in sorted(transformation_kernels) if x not in ["tshift", "cumcount"]
]
