# %% [markdown]
# # NWM short-range forecasts for San Luis Obispo creeks and the Salinas River
#
# For a few reaches and a few forecast initialization times, pulls from the
# public NOAA bucket (https://noaa-nwm-pds.s3.amazonaws.com/index.html):
#
# - NWM short-range streamflow forecasts
# - the precipitation each forecast was driven with, averaged over the basin
#   upstream of each reach, and the NWM analysis precipitation for comparison
#
# and from USGS and NWS gauges, observed discharge and stage where available.
#
# Requires: `pip install xarray h5netcdf h5py s3fs pandas matplotlib requests
# pyproj shapely`

# %% Imports
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
import s3fs
import shapely
import shapely.geometry
import xarray as xr

# %% Settings
# NWM feature_id == NHDPlus COMID.
# NHDPlus names 8193611 (~35 km2 drainage) and 8193637 (~70 km2) "San Luis
# Obispo Creek" and 8212843 (~1020 km2) "Salinas River"; 8192679 (~24 km2) and
# 8192681 (~4 km2) have no name in NHDPlus.
REACHES = {
    8193611: "San Luis Obispo Creek (8193611)",
    8193637: "San Luis Obispo Creek (8193637)",
    8192679: "Stenner Creek (8192679)",
    8192681: "Old Garden Creek (8192681)",
    8212843: "Salinas River at Paso Robles (8212843)",
}

# Gauges on each reach: {reach: {gauge ID: name shown in the plots}}.
# USGS gauges report discharge and stage.
USGS_GAUGES = {8212843: {"11147500": "Salinas River at Paso Robles"}}
# NWS gauge IDs (stage only). These are the San Luis Obispo County sensors
# shown on wr.slocountywater.org; the NWS service keeps about 30 days of them.
# NPOC1 sits about 300 m below the end of reach 8192679, just after Old Garden
# Creek joins.
NWS_GAUGES = {
    8193611: {"ADXC1": "SLO Creek at Andrews St"},
    8193637: {"QMSC1": "SLO Creek at Marsh St", "SKSC1": "SLO Creek at Elks Lane"},
    8192679: {"NPOC1": "Stenner Creek at Nipomo St"},
}

LOCAL_TZ = "America/Los_Angeles"  # PST (UTC-8) in November
INIT_TIMES_LOCAL = [
    "2026-11-18 06:00",
    "2026-11-18 12:00",
    "2026-11-18 18:00",
    "2026-11-18 22:00",
]

LEAD_HOURS = range(1, 19)  # short_range runs hourly and goes out 18 h (CONUS)
VARIABLES = ["streamflow", "velocity"]  # m3/s and m/s

BUCKET = "noaa-nwm-pds"
OUT_DIR = Path("data")
EVENT = "slo_20261118"
INIT_CMAP = "YlOrRd"  # sequential colormap for successive initializations
CMS_TO_CFS = 35.3146667

# Lambert conformal grid of the NWM forcing files (1 km cells)
NWM_PROJ = (
    "+proj=lcc +a=6370000 +b=6370000 +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 "
    "+x_0=0 +y_0=0 +units=m +no_defs"
)

OUT_DIR.mkdir(exist_ok=True)
fs = s3fs.S3FileSystem(anon=True)  # public bucket, no AWS credentials needed


def to_local(utc: pd.Series) -> pd.Series:
    """Naive UTC timestamps -> timezone-aware local timestamps."""
    return utc.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)


# %% Build the lists of files to read
# Bucket layout (all times UTC):
#   nwm.YYYYMMDD/short_range/nwm.tHHz.short_range.channel_rt.fFFF.conus.nc
#   nwm.YYYYMMDD/forcing_short_range/nwm.tHHz.short_range.forcing.fFFF.conus.nc
#   nwm.YYYYMMDD/forcing_analysis_assim/nwm.tHHz.analysis_assim.forcing.tm02.conus.nc
init_times_utc = (
    pd.to_datetime(INIT_TIMES_LOCAL).tz_localize(LOCAL_TZ).tz_convert("UTC")
)
first_valid_utc = init_times_utc.min() + pd.Timedelta(hours=min(LEAD_HOURS))
last_valid_utc = init_times_utc.max() + pd.Timedelta(hours=max(LEAD_HOURS))


def streamflow_key(init_utc: pd.Timestamp, lead_hour: int) -> str:
    return (
        f"{BUCKET}/nwm.{init_utc:%Y%m%d}/short_range/"
        f"nwm.t{init_utc:%H}z.short_range.channel_rt.f{lead_hour:03d}.conus.nc"
    )


def forecast_rain_key(init_utc: pd.Timestamp, lead_hour: int) -> str:
    return (
        f"{BUCKET}/nwm.{init_utc:%Y%m%d}/forcing_short_range/"
        f"nwm.t{init_utc:%H}z.short_range.forcing.f{lead_hour:03d}.conus.nc"
    )


def analysis_rain_key(valid_utc: pd.Timestamp) -> str:
    # Each analysis cycle looks back 2 h. The "tm02" file (valid 2 h before the
    # cycle time) is the last and most complete analysis of that hour.
    cycle = valid_utc + pd.Timedelta(hours=2)
    return (
        f"{BUCKET}/nwm.{cycle:%Y%m%d}/forcing_analysis_assim/"
        f"nwm.t{cycle:%H}z.analysis_assim.forcing.tm02.conus.nc"
    )


streamflow_keys = [streamflow_key(t, h) for t in init_times_utc for h in LEAD_HOURS]
forecast_rain_keys = [
    forecast_rain_key(t, h) for t in init_times_utc for h in LEAD_HOURS
]
analysis_rain_keys = [
    analysis_rain_key(t)
    for t in pd.date_range(first_valid_utc, last_valid_utc, freq="h")
]

for local, utc in zip(INIT_TIMES_LOCAL, init_times_utc):
    print(f"{local} PST  ->  {utc:%Y-%m-%d %H}Z")

# %% Check that every file is in the bucket
all_keys = streamflow_keys + forecast_rain_keys + analysis_rain_keys
in_bucket = set()
for folder in {key.rsplit("/", 1)[0] for key in all_keys}:
    in_bucket.update(fs.ls(folder, detail=False))

missing = sorted(set(all_keys) - in_bucket)
if missing:
    raise FileNotFoundError(f"{len(missing)} files not in bucket, e.g. {missing[0]}")
print(f"All {len(all_keys)} files found.")

# %% Streamflow forecasts: pull the reaches out of each file
# Each file holds one valid time for all ~2.7 million CONUS reaches, so only
# the selected reaches are loaded. Takes about a minute.


def read_streamflow(key: str) -> pd.DataFrame:
    with fs.open(key, "rb") as f, xr.open_dataset(f, engine="h5netcdf") as ds:
        df = ds[VARIABLES].sel(feature_id=list(REACHES)).load().to_dataframe()
        df = df.reset_index()
        df["init_utc"] = ds["reference_time"].values[0]
        df["valid_utc"] = ds["time"].values[0]
    return df


with ThreadPoolExecutor(max_workers=8) as pool:
    forecasts = pd.concat(pool.map(read_streamflow, streamflow_keys), ignore_index=True)

forecasts["reach"] = forecasts["feature_id"].map(REACHES)
forecasts["lead_hour"] = (
    (forecasts["valid_utc"] - forecasts["init_utc"]) / pd.Timedelta(hours=1)
).astype(int)
forecasts["init_local"] = to_local(forecasts["init_utc"])
forecasts["valid_local"] = to_local(forecasts["valid_utc"])
forecasts = forecasts.rename(
    columns={"streamflow": "streamflow_cms", "velocity": "velocity_ms"}
)
forecasts["streamflow_cfs"] = forecasts["streamflow_cms"] * CMS_TO_CFS
forecasts = forecasts.sort_values(["feature_id", "init_utc", "lead_hour"])
forecasts = forecasts.reset_index(drop=True)

forecasts.to_csv(OUT_DIR / f"nwm_short_range_{EVENT}.csv", index=False)
print(f"{len(forecasts)} streamflow forecast rows")
forecasts.head()

# %% Basin upstream of each reach, on the NWM forcing grid
# Basin outlines come from the USGS NLDI service, in lon/lat.
to_nwm_grid = pyproj.Proj(NWM_PROJ)

basins = {}
for feature_id, name in REACHES.items():
    response = requests.get(
        f"https://api.water.usgs.gov/nldi/linked-data/comid/{feature_id}/basin",
        timeout=120,
    )
    response.raise_for_status()
    outline = shapely.geometry.shape(response.json()["features"][0]["geometry"])
    basins[feature_id] = shapely.transform(
        outline, lambda lonlat: np.column_stack(to_nwm_grid(lonlat[:, 0], lonlat[:, 1]))
    )
    print(f"{name}: {basins[feature_id].area / 1e6:,.0f} km2 upstream")

# Window of the grid that covers every basin, with a one-cell margin
basin_bounds = np.array([basin.bounds for basin in basins.values()])
x_min, y_min = basin_bounds[:, :2].min(axis=0) - 1000
x_max, y_max = basin_bounds[:, 2:].max(axis=0) + 1000

# %% Precipitation: basin averages from the forecast and analysis forcing files
# RAINRATE is the mean rate over the hour ending at the valid time. The files
# are ~140 MB each, but only the chunks covering the window are downloaded.
# Takes about three minutes.


def read_rain(key: str) -> xr.DataArray:
    with (
        fs.open(key, "rb", block_size=2**20, cache_type="blockcache") as f,
        xr.open_dataset(f, engine="h5netcdf") as ds,
    ):
        rain = ds["RAINRATE"].isel(time=0)
        rain = rain.sel(x=slice(x_min, x_max), y=slice(y_min, y_max)).load()
        rain = rain * 3600  # mm/s -> mm/h
        rain.attrs["init_utc"] = ds["reference_time"].values[0]
        rain.attrs["valid_utc"] = ds["time"].values[0]
    return rain


def basin_masks(grid: xr.DataArray) -> dict[int, np.ndarray]:
    """Grid cells whose centre falls inside each basin."""
    xx, yy = np.meshgrid(grid["x"], grid["y"])
    masks = {}
    for feature_id, basin in basins.items():
        mask = shapely.contains_xy(basin, xx, yy)
        if not mask.any():  # basin smaller than a cell: use the nearest cell
            distance = np.hypot(xx - basin.centroid.x, yy - basin.centroid.y)
            mask = distance == distance.min()
        masks[feature_id] = mask
    return masks


def basin_precip(keys: list[str]) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=8) as pool:
        grids = list(pool.map(read_rain, keys))
    masks = basin_masks(grids[0])
    return pd.DataFrame(
        {
            "feature_id": feature_id,
            "init_utc": grid.attrs["init_utc"],
            "valid_utc": grid.attrs["valid_utc"],
            "precip_mm_h": float(grid.values[mask].mean()),
        }
        for grid in grids
        for feature_id, mask in masks.items()
    )


precip_forecast = basin_precip(forecast_rain_keys)
precip_forecast["init_local"] = to_local(precip_forecast["init_utc"])
precip_forecast["valid_local"] = to_local(precip_forecast["valid_utc"])

precip_analysis = basin_precip(analysis_rain_keys).drop(columns="init_utc")
precip_analysis["valid_local"] = to_local(precip_analysis["valid_utc"])

precip = pd.concat(
    [
        precip_forecast.assign(source="forecast"),
        precip_analysis.assign(source="analysis"),
    ],
    ignore_index=True,
)
precip["reach"] = precip["feature_id"].map(REACHES)
precip.to_csv(OUT_DIR / f"nwm_basin_precip_{EVENT}.csv", index=False)
print(f"{len(precip)} basin precipitation rows")

# %% Observed discharge and stage from USGS and NWS gauges


def read_usgs(site: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    response = requests.get(
        "https://waterservices.usgs.gov/nwis/iv/",
        params={
            "format": "json",
            "sites": site,
            "parameterCd": "00060,00065",  # discharge (cfs), gage height (ft)
            "startDT": f"{start_utc:%Y-%m-%dT%H:%MZ}",
            "endDT": f"{end_utc:%Y-%m-%dT%H:%MZ}",
        },
        timeout=120,
    )
    response.raise_for_status()

    columns = {"00060": "discharge_cfs", "00065": "stage_ft"}
    series = {}
    for ts in response.json()["value"]["timeSeries"]:
        variable = ts["variable"]
        points = ts["values"][0]["value"]
        values = pd.Series(
            [float(p["value"]) for p in points],
            index=pd.to_datetime([p["dateTime"] for p in points], utc=True),
            dtype=float,
        )
        column = columns[variable["variableCode"][0]["value"]]
        series[column] = values[values != variable["noDataValue"]]

    df = pd.DataFrame(series).reindex(columns=list(columns.values()))
    return df.rename_axis("valid_utc").reset_index()


def read_nws(gauge: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    response = requests.get(
        f"https://api.water.noaa.gov/nwps/v1/gauges/{gauge}/stageflow/observed",
        timeout=120,
    )
    response.raise_for_status()
    raw = pd.DataFrame(
        response.json()["data"], columns=["validTime", "primary", "secondary"]
    )
    # primary is stage in ft, secondary is flow in kcfs; -999 marks missing
    df = pd.DataFrame(
        {
            "valid_utc": pd.to_datetime(raw["validTime"], utc=True),
            "discharge_cfs": raw["secondary"].where(raw["secondary"] > -900) * 1000,
            "stage_ft": raw["primary"].where(raw["primary"] > -900),
        }
    )
    return df[df["valid_utc"].between(start_utc, end_utc)]


observed = []
for network, read, gauges in [
    ("USGS", read_usgs, USGS_GAUGES),
    ("NWS", read_nws, NWS_GAUGES),
]:
    for feature_id, reach_gauges in gauges.items():
        for gauge_id, gauge_name in reach_gauges.items():
            df = read(gauge_id, init_times_utc.min(), last_valid_utc)
            df.insert(0, "feature_id", feature_id)
            df.insert(1, "gauge", f"{network} {gauge_id}")
            df.insert(2, "gauge_name", gauge_name)
            observed.append(df)

observed = pd.concat(observed, ignore_index=True)
observed["valid_local"] = observed["valid_utc"].dt.tz_convert(LOCAL_TZ)
observed["streamflow_cms"] = observed["discharge_cfs"] / CMS_TO_CFS
observed.to_csv(OUT_DIR / f"observed_{EVENT}.csv", index=False)
print(
    observed.groupby(["feature_id", "gauge", "gauge_name"])[
        ["streamflow_cms", "stage_ft"]
    ].count()
)

# %% Peak timing: each forecast against the analysis rain and the gauges


def peak(df: pd.DataFrame, column: str) -> tuple[float, str]:
    """Largest value in a column and the local time it occurs."""
    if df[column].notna().sum() == 0:
        return np.nan, ""
    row = df.loc[df[column].idxmax()]
    return row[column], f"{row['valid_local']:%b %d %H:%M}"


rows = []
for feature_id, name in REACHES.items():
    flow = forecasts[forecasts["feature_id"] == feature_id]
    rain = precip_forecast[precip_forecast["feature_id"] == feature_id]
    for init_local in sorted(flow["init_local"].unique()):
        rain_peak, rain_time = peak(rain[rain["init_local"] == init_local], "precip_mm_h")
        flow_peak, flow_time = peak(flow[flow["init_local"] == init_local], "streamflow_cfs")
        rows.append(
            {
                "reach": name,
                "source": f"forecast {init_local:%b %d %H:%M}",
                "peak_rain_mm_h": rain_peak,
                "peak_rain_time": rain_time,
                "peak_flow_cfs": flow_peak,
                "peak_flow_time": flow_time,
            }
        )

    rain_peak, rain_time = peak(
        precip_analysis[precip_analysis["feature_id"] == feature_id], "precip_mm_h"
    )
    rows.append(
        {
            "reach": name,
            "source": "NWM analysis rain",
            "peak_rain_mm_h": rain_peak,
            "peak_rain_time": rain_time,
        }
    )

    gauges = observed[observed["feature_id"] == feature_id]
    for (gauge, gauge_name), obs in gauges.groupby(["gauge", "gauge_name"], sort=False):
        flow_peak, flow_time = peak(obs, "discharge_cfs")
        stage_peak, stage_time = peak(obs, "stage_ft")
        rows.append(
            {
                "reach": name,
                "source": f"observed, {gauge_name} ({gauge})",
                "peak_flow_cfs": flow_peak,
                "peak_flow_time": flow_time,
                "peak_stage_ft": stage_peak,
                "peak_stage_time": stage_time,
            }
        )

summary = pd.DataFrame(rows)
print(summary.round(2).astype(object).fillna("").to_string(index=False))

# %% Plot: one figure per reach, precipitation above streamflow
# Times are plotted as local clock time (PST).


def clock(local: pd.Series) -> pd.Series:
    return local.dt.tz_localize(None)


def save_figure(fig: plt.Figure, path: Path, attempts: int = 5) -> None:
    """Save a figure, retrying when Windows briefly has the old file locked."""
    for attempt in range(attempts):
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1)


# Colour gradient over the initializations: earliest lightest, latest darkest
init_times_local = sorted(forecasts["init_local"].unique())
init_colors = dict(
    zip(
        init_times_local,
        plt.get_cmap(INIT_CMAP)(np.linspace(0.35, 0.95, len(init_times_local))),
    )
)

for feature_id, name in REACHES.items():
    fig, (ax_rain, ax_flow) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[1, 2]
    )

    # Precipitation: analysis as bars, each forecast as a stepped line
    analysis = precip_analysis[precip_analysis["feature_id"] == feature_id]
    ax_rain.bar(
        clock(analysis["valid_local"]),
        analysis["precip_mm_h"],
        width=-1 / 24,  # each value covers the hour ending at its valid time
        align="edge",
        color="0.8",
        label="NWM analysis",
    )
    rain = precip_forecast[precip_forecast["feature_id"] == feature_id]
    for init_local, run in rain.groupby("init_local"):
        hour_edges = [init_local.tz_localize(None), *clock(run["valid_local"])]
        ax_rain.stairs(
            run["precip_mm_h"],
            hour_edges,
            baseline=None,
            color=init_colors[init_local],
            linewidth=1.5,
        )
    ax_rain.set_ylabel("Basin-average\nprecipitation (mm/h)")
    ax_rain.legend(loc="upper right")
    ax_rain.grid(alpha=0.3)
    ax_rain.set_title(name)

    # Streamflow: each forecast, plus observed discharge and stage if gauged
    flow = forecasts[forecasts["feature_id"] == feature_id]
    for init_local, run in flow.groupby("init_local"):
        ax_flow.plot(
            clock(run["valid_local"]),
            run["streamflow_cfs"],
            marker="o",
            markersize=3,
            color=init_colors[init_local],
            label=f"NWM forecast, init {init_local:%b %d %I %p} PST",
        )
    ax_flow.set_ylabel("Streamflow (cfs)")
    ax_flow.grid(alpha=0.3)
    handles, labels = ax_flow.get_legend_handles_labels()

    # A gauge with discharge is drawn on the flow axis. A stage-only gauge is
    # drawn on a second axis as the rise above its lowest value in the window,
    # because each gauge has its own datum.
    gauges = observed[observed["feature_id"] == feature_id]
    ax_stage = None
    stage_styles = cycle([":", "--", "-."])
    for (gauge, gauge_name), obs in gauges.groupby(["gauge", "gauge_name"], sort=False):
        if obs["discharge_cfs"].notna().any():
            handles += ax_flow.plot(
                clock(obs["valid_local"]), obs["discharge_cfs"], color="black"
            )
            labels.append(f"Observed discharge, {gauge_name} ({gauge})")
        elif obs["stage_ft"].notna().any():
            if ax_stage is None:
                ax_stage = ax_flow.twinx()
                ax_stage.set_ylabel("Stage rise above lowest in window (ft)")
            handles += ax_stage.plot(
                clock(obs["valid_local"]),
                obs["stage_ft"] - obs["stage_ft"].min(),
                color="black",
                linestyle=next(stage_styles),
            )
            labels.append(f"Observed stage, {gauge_name} ({gauge})")

    ax_flow.set_xlabel("Valid time (PST)")
    ax_flow.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%I %p"))
    fig.tight_layout()
    # Legend sits below the figure so it never covers the hydrographs
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        fontsize="small",
    )
    save_figure(fig, OUT_DIR / f"nwm_short_range_{EVENT}_{feature_id}.png")

plt.show()

# %%
