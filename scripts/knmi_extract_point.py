from pathlib import Path
import math

import xarray as xr


GRIB_PATH = Path(
    "data/raw/knmi/harmonie_arome_cy43_p1/extracted/"
    "HA43_N20_202605150400_00000_GB"
)

# Valkenburgse meer lan/lon
SITE_LAT = 52.1603
SITE_LON = 4.44197

def open_param(parameter: int, level: int):
    return xr.open_dataset(
        GRIB_PATH,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "indicatorOfParameter": parameter,
                "level": level,
            },
            "indexpath": "",
        },
    )


def nearest_value(ds):
    # Coordinate names should be latitude/longitude for this regular_ll file.
    point = ds.sel(latitude=SITE_LAT, longitude=SITE_LON, method="nearest")

    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise RuntimeError(f"Expected one data variable, found: {data_vars}")

    var = data_vars[0]
    return var, float(point[var].values), float(point.latitude), float(point.longitude)


def wind_speed_direction(u: float, v: float):
    speed = math.sqrt(u**2 + v**2)

    # Meteorological wind direction: direction from which wind blows.
    direction = (270 - math.degrees(math.atan2(v, u))) % 360

    return speed, direction


def main():
    for level in [10, 50, 100, 200, 300]:
        u_ds = open_param(parameter=33, level=level)
        v_ds = open_param(parameter=34, level=level)

        _, u, lat, lon = nearest_value(u_ds)
        _, v, _, _ = nearest_value(v_ds)

        speed, direction = wind_speed_direction(u, v)

        print(
            f"{level:>3} m | "
            f"grid=({lat:.5f}, {lon:.5f}) | "
            f"u={u: .3f} v={v: .3f} | "
            f"speed={speed:.3f} m/s | "
            f"direction={direction:.1f}°"
        )


if __name__ == "__main__":
    main()

