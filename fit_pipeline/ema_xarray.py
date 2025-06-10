#generate ema on an xarray object on dimension D with hl 1, 3, 5, 10
import numpy as np
import pandas as pd
import xarray as xr

# --- 1. toy data -------------------------------------------------------------
N = 20                           # length of the demo series
da = xr.DataArray(
    np.random.randn(N),
    coords={"D": np.arange(N)},
    dims="D",
    name="price",
)
# da.shape -> (20,)  with dim "D"

# --- 2. helper to compute EMA on one 1-D slice -------------------------------
def _ema_1d(arr, hl):
    """Return EMA of a 1-D NumPy array using pandas’ ewm."""
    return (
        pd.Series(arr)
        .ewm(halflife=hl, adjust=False)
        .mean()
        .to_numpy()
    )

def ema_da(da, *, hl, dim="D"):
    """Vectorised EMA for an xarray DataArray along *dim*."""
    return xr.apply_ufunc(
        _ema_1d,                       # the function to apply
        da,                            # first (only) argument
        kwargs={"hl": hl},             # pass half-life
        input_core_dims=[[dim]],       # treat *dim* as a 1-D core
        output_core_dims=[[dim]],      # same shape back
        vectorize=True,                # loop over all other dims
        dask="parallelized",           # works on dask-backed arrays too
        output_dtypes=[da.dtype],      # preserve dtype
    )

# --- 3. add EMA variables with the requested half-lives ----------------------
for hl in (1, 3, 5, 10):
    da[f"ema_hl{hl}"] = ema_da(da, hl=hl)

# --- 4. view result ----------------------------------------------------------
print(da)
