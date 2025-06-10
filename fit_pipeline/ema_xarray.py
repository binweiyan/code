import numpy as np
import pandas as pd
import xarray as xr

# --- 0. demo 4-D array -------------------------------------------------------
S, D, E, V1 = 2, 20, 3, 500                 # arbitrary shapes
data = xr.DataArray(
    np.random.randn(S, D, E, V1),
    coords={"S": np.arange(S),
            "D": np.arange(D),
            "E": np.arange(E),
            "V1": np.arange(V1)},
    dims=("S", "D", "E", "V1"),
    name="price",
)

# --- 1. 1-D helper (unchanged) ----------------------------------------------
def _ema_1d(arr, hl):
    return (
        pd.Series(arr)
        .ewm(halflife=hl, adjust=False)
        .mean()
        .to_numpy()
    )

def ema_da(da, *, hl, dim="D"):
    return xr.apply_ufunc(
        _ema_1d,
        da,
        kwargs={"hl": hl},
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,            # loops over S, E, V1 automatically
        dask="parallelized",
        output_dtypes=[da.dtype],
    )

# --- 2. compute the four EMAs -----------------------------------------------
hls = [1, 3, 5, 10]

# Option A – put them side-by-side under a new HL coordinate
ema_stack = xr.concat(
    [ema_da(data, hl=h) for h in hls],
    dim=xr.IndexVariable("HL", hls)
)
# dims: ("HL", "S", "D", "E", "V1")

# Option B – keep separate variables in a Dataset
ds = data.to_dataset(name="price")
for h in hls:
    ds[f"price_ema_hl{h}"] = ema_da(data, hl=h)

#given a piece of data, hls, check if the V dimension is bigger than 100
#if so, split the data into slices of pieces with V1 size 100 and compute the ema for each slice and store them in seperate files
#if not, compute the ema for the whole data

#return the ema data
def ema_small_data(data, hls, output_path):
    ema_data = xr.concat(
        [ema_da(data, hl=h) for h in hls],
        dim=xr.IndexVariable("HL", hls)
    )
    ema_data.to_netcdf(f"{output_path}/ema_data.nc")



def ema_data(data, hls, output_path):
    if data.V1.size > 100:
        for i in range(0, data.V1.size, 100):
            #split the data into slices of pieces with V1 size 100
            data_slice = data.isel(V1=slice(i, i+100))
            #save the data slice to a netcdf file
            data_slice.to_netcdf(f"{output_path}/data_slice_{i}.nc")
    else:
        #compute the ema for the whole data
        ema_small_data(data, hls, output_path)







    