"""ECMWF Open Data provider for the provider-neutral forecast boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from next_day_wind_model.forecast_provider import (
    ForecastBatch,
    ForecastPoint,
    ForecastRun,
    ForecastSite,
    ForecastValue,
    ForecastValueBatch,
    utc_datetime,
)


PROVIDER = "ECMWF"
MODEL = "IFS"
MODEL_VERSION = "50r1"
SOURCE = "ecmwf_open_data"
RESOLUTION = "0.25 degrees"
STREAM = "oper"
FORECAST_TYPE = "fc"
WIND_PARAMETERS = ("10u", "10v")
GUST_PARAMETER = "10fg"
GUST_THREE_HOUR_PARAMETER = "10fg3"


def gust_parameters_for_leads(lead_hours: Sequence[int]) -> tuple[str, ...]:
    """Return public IFS gust identifiers for the requested native steps."""
    steps = {int(step) for step in lead_hours}
    parameters: list[str] = []
    if any(step <= 90 or step >= 150 for step in steps):
        parameters.append(GUST_PARAMETER)
    if any(93 <= step <= 144 for step in steps):
        parameters.append(GUST_THREE_HOUR_PARAMETER)
    return tuple(parameters)


class EcmwfOpenDataError(RuntimeError):
    pass


def _official_client(source: str, model: str, resolution: str) -> Any:
    try:
        from ecmwf.opendata import Client
    except ModuleNotFoundError as exc:
        raise EcmwfOpenDataError(
            "Could not import the official ecmwf-opendata client or one of its dependencies. "
            "Install next_day_wind_model/requirements-ecmwf.txt in the development environment."
        ) from exc
    return Client(source=source, model=model, resol=resolution)


def _as_datetime(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise EcmwfOpenDataError(f"Invalid ECMWF timestamp: {value!r}")
    parsed = timestamp.to_pydatetime()
    return utc_datetime(parsed)


def _normalised_name(value: Any) -> str:
    return str(value or "").lower().replace("_", "").replace("-", "")


def _matches_parameter(variable_name: str, attrs: dict[str, Any], aliases: Iterable[str]) -> bool:
    candidates = {
        _normalised_name(variable_name),
        _normalised_name(attrs.get("GRIB_shortName")),
        _normalised_name(attrs.get("shortName")),
    }
    wanted = {_normalised_name(alias) for alias in aliases}
    return bool(candidates & wanted)


def _target_longitude(longitudes: Any, longitude: float) -> float:
    try:
        minimum = float(longitudes.min())
        maximum = float(longitudes.max())
    except Exception:
        return float(longitude)
    if minimum >= 0.0 and maximum > 180.0:
        return float(longitude) % 360.0
    return ((float(longitude) + 180.0) % 360.0) - 180.0


def _parameter_series_from_datasets(
    datasets: Sequence[Any],
    site: ForecastSite,
    aliases: Iterable[str],
) -> dict[int, dict[str, Any]]:
    """Extract one parameter at the nearest grid point, keyed by lead hour."""
    values: dict[int, dict[str, Any]] = {}
    for dataset in datasets:
        for variable_name, data_array in dataset.data_vars.items():
            if not _matches_parameter(variable_name, dict(data_array.attrs), aliases):
                continue
            if "latitude" not in data_array.coords or "longitude" not in data_array.coords:
                raise EcmwfOpenDataError(f"ECMWF variable {variable_name!r} has no latitude/longitude coordinates")
            longitude = _target_longitude(data_array.coords["longitude"], site.longitude)
            point = data_array.sel(latitude=site.latitude, longitude=longitude, method="nearest")

            if "step" in point.dims:
                selections = (point.isel(step=index) for index in range(int(point.sizes["step"])))
            else:
                selections = (point,)
            for selected in selections:
                if "step" in selected.coords:
                    lead_hours = int(round(pd.Timedelta(selected.coords["step"].item()).total_seconds() / 3600.0))
                elif "valid_time" in selected.coords and "time" in selected.coords:
                    delta = _as_datetime(selected.coords["valid_time"].item()) - _as_datetime(selected.coords["time"].item())
                    lead_hours = int(round(delta.total_seconds() / 3600.0))
                else:
                    raise EcmwfOpenDataError(f"ECMWF variable {variable_name!r} has no forecast step metadata")

                run_time = _as_datetime(selected.coords["time"].item())
                valid_time = (
                    _as_datetime(selected.coords["valid_time"].item())
                    if "valid_time" in selected.coords
                    else run_time + pd.Timedelta(hours=lead_hours).to_pytimedelta()
                )
                values[lead_hours] = {
                    "value": float(selected.item()),
                    "run_time": run_time,
                    "valid_time": valid_time,
                    "grid_latitude": float(selected.coords["latitude"].item()),
                    "grid_longitude": float(selected.coords["longitude"].item()),
                    "units": data_array.attrs.get("GRIB_units") or data_array.attrs.get("units"),
                    "short_name": data_array.attrs.get("GRIB_shortName") or variable_name,
                    "parameter_id": data_array.attrs.get("GRIB_paramId"),
                    "grid_type": data_array.attrs.get("GRIB_gridType"),
                    "edition": data_array.attrs.get("GRIB_edition"),
                }
    return values


def _open_grib_datasets(path: Path) -> list[Any]:
    try:
        import cfgrib
    except ModuleNotFoundError as exc:
        raise EcmwfOpenDataError(
            "cfgrib and ecCodes are required. Install next_day_wind_model/requirements.txt."
        ) from exc
    try:
        return list(cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}))
    except Exception as exc:
        raise EcmwfOpenDataError(f"Could not decode ECMWF GRIB2 file {path}") from exc


def extract_ecmwf_point_forecasts(
    wind_grib: Path,
    site: ForecastSite,
    *,
    requested_lead_hours: Sequence[int] | None = None,
    gust_grib: Path | None = None,
    fetched_time: datetime | None = None,
) -> tuple[ForecastPoint, ...]:
    """Decode global GRIB fields and retain only the grid point nearest ``site``."""
    return extract_ecmwf_point_forecasts_for_sites(
        wind_grib,
        (site,),
        requested_lead_hours=requested_lead_hours,
        gust_grib=gust_grib,
        fetched_time=fetched_time,
    )


def extract_ecmwf_point_forecasts_for_sites(
    wind_grib: Path,
    sites: Sequence[ForecastSite],
    *,
    requested_lead_hours: Sequence[int] | None = None,
    gust_grib: Path | None = None,
    fetched_time: datetime | None = None,
) -> tuple[ForecastPoint, ...]:
    """Decode each GRIB once and extract the nearest point for every site."""
    unique_sites = tuple(sites)
    if not unique_sites:
        raise ValueError("sites cannot be empty")
    if len({site.name for site in unique_sites}) != len(unique_sites):
        raise ValueError("site names must be unique")

    datasets = _open_grib_datasets(wind_grib)
    gust_datasets = _open_grib_datasets(gust_grib) if gust_grib is not None else []
    try:
        fetched = fetched_time or datetime.now(timezone.utc)
        points: list[ForecastPoint] = []
        for site in unique_sites:
            points.extend(
                _extract_ecmwf_point_forecasts_from_datasets(
                    datasets,
                    site,
                    requested_lead_hours=requested_lead_hours,
                    gust_datasets=gust_datasets,
                    fetched_time=fetched,
                )
            )
        return tuple(points)
    finally:
        for dataset in (*datasets, *gust_datasets):
            dataset.close()


def _extract_ecmwf_point_forecasts_from_datasets(
    datasets: Sequence[Any],
    site: ForecastSite,
    *,
    requested_lead_hours: Sequence[int] | None,
    gust_datasets: Sequence[Any],
    fetched_time: datetime,
) -> tuple[ForecastPoint, ...]:
    u_values = _parameter_series_from_datasets(datasets, site, ("10u", "u10"))
    v_values = _parameter_series_from_datasets(datasets, site, ("10v", "v10"))
    gust_values = (
        _parameter_series_from_datasets(gust_datasets, site, ("10fg", "10fg3", "fg10", "i10fg"))
        if gust_datasets
        else {}
    )
    required = set(int(step) for step in requested_lead_hours) if requested_lead_hours is not None else None
    common_steps = sorted(set(u_values) & set(v_values))
    if required is not None:
        common_steps = [step for step in common_steps if step in required]
    if not common_steps:
        raise EcmwfOpenDataError("No matching 10 m U/V forecast steps were decoded from ECMWF GRIB2")

    points: list[ForecastPoint] = []
    for lead_hours in common_steps:
        u_row = u_values[lead_hours]
        v_row = v_values[lead_hours]
        if u_row["run_time"] != v_row["run_time"] or u_row["valid_time"] != v_row["valid_time"]:
            raise EcmwfOpenDataError(f"10 m U/V metadata disagree at lead hour {lead_hours}")
        gust = gust_values.get(lead_hours, {}).get("value")
        points.append(
            ForecastPoint.from_uv(
                provider=PROVIDER,
                model=MODEL,
                run_time=u_row["run_time"],
                valid_time=u_row["valid_time"],
                lead_time_hours=lead_hours,
                site=site,
                grid_latitude=u_row["grid_latitude"],
                grid_longitude=u_row["grid_longitude"],
                u10_mps=u_row["value"],
                v10_mps=v_row["value"],
                wind_gust_mps=gust,
                source=SOURCE,
                fetched_time=fetched_time,
                model_version=MODEL_VERSION,
                resolution=RESOLUTION,
                metadata={
                    "stream": STREAM,
                    "type": FORECAST_TYPE,
                    "u_units": u_row["units"],
                    "v_units": v_row["units"],
                    "u_short_name": u_row["short_name"],
                    "v_short_name": v_row["short_name"],
                },
            )
        )
    return tuple(points)


def extract_ecmwf_scalar_values(
    grib_path: Path,
    site: ForecastSite,
    *,
    variable: str,
    aliases: Sequence[str],
    requested_lead_hours: Sequence[int] | None = None,
    fetched_time: datetime | None = None,
) -> tuple[ForecastValue, ...]:
    """Decode one optional ECMWF scalar field independently from U/V."""
    return extract_ecmwf_scalar_values_for_sites(
        grib_path,
        (site,),
        variable=variable,
        aliases=aliases,
        requested_lead_hours=requested_lead_hours,
        fetched_time=fetched_time,
    )


def extract_ecmwf_scalar_values_for_sites(
    grib_path: Path,
    sites: Sequence[ForecastSite],
    *,
    variable: str,
    aliases: Sequence[str],
    requested_lead_hours: Sequence[int] | None = None,
    fetched_time: datetime | None = None,
) -> tuple[ForecastValue, ...]:
    """Decode one scalar GRIB once and extract it for every requested site."""
    unique_sites = tuple(sites)
    if not unique_sites:
        raise ValueError("sites cannot be empty")
    if len({site.name for site in unique_sites}) != len(unique_sites):
        raise ValueError("site names must be unique")
    datasets = _open_grib_datasets(grib_path)
    try:
        fetched = fetched_time or datetime.now(timezone.utc)
        values: list[ForecastValue] = []
        for site in unique_sites:
            values.extend(
                _extract_ecmwf_scalar_values_from_datasets(
                    datasets,
                    site,
                    variable=variable,
                    aliases=aliases,
                    requested_lead_hours=requested_lead_hours,
                    fetched_time=fetched,
                )
            )
        return tuple(values)
    finally:
        for dataset in datasets:
            dataset.close()


def _extract_ecmwf_scalar_values_from_datasets(
    datasets: Sequence[Any],
    site: ForecastSite,
    *,
    variable: str,
    aliases: Sequence[str],
    requested_lead_hours: Sequence[int] | None,
    fetched_time: datetime,
) -> tuple[ForecastValue, ...]:
    rows = _parameter_series_from_datasets(datasets, site, aliases)
    required = set(int(step) for step in requested_lead_hours) if requested_lead_hours is not None else None
    steps = sorted(rows)
    if required is not None:
        steps = [step for step in steps if step in required]

    values: list[ForecastValue] = []
    for lead_hours in steps:
        row = rows[lead_hours]
        values.append(
            ForecastValue(
                provider=PROVIDER,
                model=MODEL,
                run_time=row["run_time"],
                valid_time=row["valid_time"],
                lead_time_hours=lead_hours,
                site=site,
                grid_latitude=row["grid_latitude"],
                grid_longitude=row["grid_longitude"],
                variable=variable,
                value=row["value"],
                unit=str(row["units"] or "m s**-1"),
                source=SOURCE,
                fetched_time=fetched_time,
                model_version=MODEL_VERSION,
                resolution=RESOLUTION,
                metadata={
                    "stream": STREAM,
                    "type": FORECAST_TYPE,
                    "short_name": row["short_name"],
                    "parameter_id": row["parameter_id"],
                    "grid_type": row["grid_type"],
                    "grib_edition": row["edition"],
                },
            )
        )
    return tuple(values)


class EcmwfOpenDataProvider:
    """Retrieve deterministic IFS surface fields through ECMWF's official client."""

    provider_name = PROVIDER
    model_name = MODEL

    def __init__(
        self,
        *,
        source: str = "ecmwf",
        include_gust: bool = False,
        client: Any | None = None,
    ) -> None:
        self.source = source
        self.include_gust = include_gust
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _official_client(self.source, model="ifs", resolution="0p25")
        return self._client

    def _request(self, lead_hours: Sequence[int], *, include_parameters: bool = True) -> dict[str, Any]:
        request: dict[str, Any] = {
            "stream": STREAM,
            "type": FORECAST_TYPE,
            "step": list(lead_hours),
        }
        if include_parameters:
            request["param"] = list(WIND_PARAMETERS)
        return request

    def latest_run(self, required_lead_hours: Sequence[int]) -> ForecastRun:
        steps = tuple(sorted({int(step) for step in required_lead_hours}))
        if not steps:
            raise ValueError("required_lead_hours cannot be empty")
        latest = self.client.latest(**self._request((max(steps),)))
        run_time = _as_datetime(latest)
        return ForecastRun(
            provider=PROVIDER,
            model=MODEL,
            run_time=run_time,
            source=SOURCE,
            model_version=MODEL_VERSION,
            resolution=RESOLUTION,
            provider_reference=f"{run_time:%Y%m%d%H}",
        )

    def _retrieve(self, run: ForecastRun, lead_hours: Sequence[int], parameters: Sequence[str], target: Path) -> Any:
        """Download to a temporary sibling and publish the GRIB atomically."""
        partial = target.with_suffix(target.suffix + ".part")
        partial.unlink(missing_ok=True)
        request = self._request(lead_hours)
        request.update(
            {
                "date": run.run_time.strftime("%Y%m%d"),
                "time": run.run_time.hour,
                "param": list(parameters),
                "target": str(partial),
            }
        )
        try:
            result = self.client.retrieve(**request)
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise EcmwfOpenDataError(f"ECMWF retrieval produced no data for {target.name}")
            partial.replace(target)
            return result
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    def fetch_wind(
        self,
        run: ForecastRun,
        site: ForecastSite,
        lead_hours: Sequence[int],
        work_dir: Path,
    ) -> ForecastBatch:
        """Fetch only mandatory U/V fields for independently durable storage."""
        return self.fetch_wind_sites(run, (site,), lead_hours, work_dir)

    def fetch_wind_sites(
        self,
        run: ForecastRun,
        sites: Sequence[ForecastSite],
        lead_hours: Sequence[int],
        work_dir: Path,
    ) -> ForecastBatch:
        """Fetch global U/V fields once and extract all requested sites."""
        if run.provider != PROVIDER or run.model != MODEL:
            raise ValueError("run does not belong to ECMWF IFS")
        unique_sites = tuple(sites)
        if not unique_sites:
            raise ValueError("sites cannot be empty")
        steps = tuple(sorted({int(step) for step in lead_hours}))
        if not steps:
            raise ValueError("lead_hours cannot be empty")
        work_dir.mkdir(parents=True, exist_ok=True)
        run_id = run.run_time.strftime("%Y%m%d%H")
        wind_path = work_dir / f"ecmwf_ifs_{run_id}_{steps[0]}-{steps[-1]}h_10uv.grib2"
        self._retrieve(run, steps, WIND_PARAMETERS, wind_path)
        fetched_time = datetime.now(timezone.utc)
        points = extract_ecmwf_point_forecasts_for_sites(
            wind_path,
            unique_sites,
            requested_lead_hours=steps,
            fetched_time=fetched_time,
        )
        if any(point.run_time != run.run_time for point in points):
            raise EcmwfOpenDataError("downloaded ECMWF GRIB belongs to a different run than requested")
        return ForecastBatch(run=run, points=points, artifacts=(wind_path,))

    def fetch_gust_values(
        self,
        run: ForecastRun,
        site: ForecastSite,
        lead_hours: Sequence[int],
        work_dir: Path,
    ) -> ForecastValueBatch:
        """Fetch gust alone, allowing recovery without re-downloading U/V."""
        return self.fetch_gust_values_sites(run, (site,), lead_hours, work_dir)

    def fetch_gust_values_sites(
        self,
        run: ForecastRun,
        sites: Sequence[ForecastSite],
        lead_hours: Sequence[int],
        work_dir: Path,
    ) -> ForecastValueBatch:
        """Fetch global gust fields once and extract all requested sites."""
        if run.provider != PROVIDER or run.model != MODEL:
            raise ValueError("run does not belong to ECMWF IFS")
        unique_sites = tuple(sites)
        if not unique_sites:
            raise ValueError("sites cannot be empty")
        steps = tuple(sorted({int(step) for step in lead_hours}))
        if not steps:
            raise ValueError("lead_hours cannot be empty")
        work_dir.mkdir(parents=True, exist_ok=True)
        run_id = run.run_time.strftime("%Y%m%d%H")
        gust_path = work_dir / f"ecmwf_ifs_{run_id}_{steps[0]}-{steps[-1]}h_gust.grib2"
        self._retrieve(run, steps, gust_parameters_for_leads(steps), gust_path)
        fetched_time = datetime.now(timezone.utc)
        values = extract_ecmwf_scalar_values_for_sites(
            gust_path,
            unique_sites,
            variable="wind_gust_10m",
            aliases=("10fg", "10fg3", "fg10", "i10fg"),
            requested_lead_hours=steps,
            fetched_time=fetched_time,
        )
        if any(value.run_time != run.run_time for value in values):
            raise EcmwfOpenDataError("downloaded ECMWF gust GRIB belongs to a different run than requested")
        return ForecastValueBatch(run=run, values=values, artifacts=(gust_path,))

    def fetch(
        self,
        run: ForecastRun,
        site: ForecastSite,
        lead_hours: Sequence[int],
        work_dir: Path,
    ) -> ForecastBatch:
        wind_batch = self.fetch_wind(run, site, lead_hours, work_dir)
        if not self.include_gust:
            return wind_batch

        gust_steps = tuple(step for step in lead_hours if int(step) > 0)
        if not gust_steps:
            return wind_batch
        gust_batch = self.fetch_gust_values(run, site, gust_steps, work_dir)
        gust_by_lead = {value.lead_time_hours: value.value for value in gust_batch.values}
        points = tuple(
            replace(point, wind_gust_mps=gust_by_lead.get(point.lead_time_hours))
            for point in wind_batch.points
        )
        return ForecastBatch(
            run=run,
            points=points,
            artifacts=wind_batch.artifacts + gust_batch.artifacts,
        )
