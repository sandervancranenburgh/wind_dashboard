import math
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import xarray as xr


TAR_PATH = Path("data/raw/knmi/harmonie_arome_cy43_p1/HARM43_V1_P1_2026051504.tar")

# Use your actual spot/sensor coordinates.
SITE_LAT = 52.168
SITE_LON = 4.437

LEVELS = [10, 50, 100, 200, 300]


def parse_run_and_horizon(filename: str):
    # Example: HA43_N20_202605150400_00600_GB
    parts = filename.split("_")
    run_str = parts[2]          # 202605150400
    horizon_str = parts[3]      # 00600

    run_ts = pd.to_datetime(run_str, format="%Y%m%d%H%M", utc=True)
    horizon_hr = int(horizon_str[:3])
    target_ts = run_ts + pd.Timedelta(hours=horizon_hr)

    return run_ts, horizon_hr, target_ts


def open_param(grib_path: Path, parameter: int, level: int):
    return xr.open_dataset(
        grib_path,
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
    point = ds.sel(latitude=SITE_LAT, longitude=SITE_LON, method="nearest")

    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise RuntimeError(f"Expected one data variable, found: {data_vars}")

    var = data_vars[0]
    return float(point[var].values), float(point.latitude), float(point.longitude)


def wind_speed_direction(u: float, v: float):
    speed = math.sqrt(u**2 + v**2)

    # Meteorological direction: direction from which the wind blows.
    direction = (270 - math.degrees(math.atan2(v, u))) % 360

    return speed, direction


def circular_difference_degrees(a: float, b: float) -> float:
    """Smallest signed difference a - b in degrees."""
    return (a - b + 180) % 360 - 180


def extract_one_grib(grib_path: Path):
    run_ts, horizon_hr, target_ts = parse_run_and_horizon(grib_path.name)

    row = {
        "source": "knmi_harmonie_p1",
        "run_ts": run_ts.isoformat(),
        "target_ts": target_ts.isoformat(),
        "horizon_hr": horizon_hr,
        "site_lat": SITE_LAT,
        "site_lon": SITE_LON,
    }

    grid_lat = None
    grid_lon = None

    for level in LEVELS:
        u_ds = open_param(grib_path, parameter=33, level=level)
        v_ds = open_param(grib_path, parameter=34, level=level)

        u, lat, lon = nearest_value(u_ds)
        v, _, _ = nearest_value(v_ds)

        speed, direction = wind_speed_direction(u, v)

        row[f"u_{level}m"] = u
        row[f"v_{level}m"] = v
        row[f"wind_speed_{level}m"] = speed
        row[f"wind_dir_{level}m"] = direction
        row[f"wind_dir_sin_{level}m"] = math.sin(math.radians(direction))
        row[f"wind_dir_cos_{level}m"] = math.cos(math.radians(direction))

        grid_lat = lat
        grid_lon = lon

    row["grid_lat"] = grid_lat
    row["grid_lon"] = grid_lon

    # Derived vertical-structure features.
    row["speed_shear_10m_50m"] = row["wind_speed_50m"] - row["wind_speed_10m"]
    row["speed_shear_10m_100m"] = row["wind_speed_100m"] - row["wind_speed_10m"]
    row["speed_shear_10m_300m"] = row["wind_speed_300m"] - row["wind_speed_10m"]

    row["speed_ratio_10m_50m"] = row["wind_speed_10m"] / row["wind_speed_50m"]
    row["speed_ratio_10m_100m"] = row["wind_speed_10m"] / row["wind_speed_100m"]
    row["speed_ratio_10m_300m"] = row["wind_speed_10m"] / row["wind_speed_300m"]

    row["dir_shear_10m_100m"] = circular_difference_degrees(
        row["wind_dir_100m"],
        row["wind_dir_10m"],
    )

    row["dir_shear_10m_300m"] = circular_difference_degrees(
        row["wind_dir_300m"],
        row["wind_dir_10m"],
    )

    return row


def main():
    out_dir = Path("data/processed/knmi")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    with tarfile.open(TAR_PATH, "r") as tar:
        members = sorted( (m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name,)

        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            for i, member in enumerate(members, start=1):
                print(f"[{i:02d}/{len(members)}] Extracting {member.name}")

                tar.extract(member, path=tmp_dir)
                grib_path = tmp_dir / member.name

                rows.append(extract_one_grib(grib_path))

                grib_path.unlink()

    df = pd.DataFrame(rows).sort_values("horizon_hr")

    out_path = out_dir / "harmonie_p1_wind_features_2026051504.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(df[[
        "horizon_hr",
        "target_ts",
        "wind_speed_10m",
        "wind_speed_100m",
        "speed_shear_10m_100m",
        "wind_dir_10m",
    ]].head())


if __name__ == "__main__":
    main()
