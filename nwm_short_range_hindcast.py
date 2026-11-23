# %% [markdown]

# NWM short-range forecasts for San Luis Obispo Flood Event on November 18, 2026
# - Pulling data from NOAA bucket: https://noaa-nwm-pds.s3.amazonaws.com/index.html
# - Great summary of the NWM data access at CIROH hub: https://hub.ciroh.org/docs/products/data-management/dataaccess/
# - This code was written with Cursor and Claude; execution and syntax checked by Araki.

# %% Imports
import io
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path

import matplotlib.colors as mcolors
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

# %% Configurations

# Find your NWM feature_id (NHDPlus COMID) at https://water.noaa.gov/map
REACHES = {
    8193611: "San Luis Obispo Creek (ID:8193611)",
    8193637: "San Luis Obispo Creek (ID:8193637)",
    8192679: "Stenner Creek (ID:8192679)",
    8192681: "Old Garden Creek (ID:8192681)",
    8212843: "Salinas River at Paso Robles (ID:8212843)",
}

# USGS streamflow gauge
USGS_GAUGES = {8212843: {"11147500": "Salinas River at Paso Robles"}}

# NWS gauge IDs (stage only, San Luis Obispo County sensors)
# Original time series data available at wr.slocountywater.org
NWS_GAUGES = {
    8193611: {"ADXC1": "SLO Creek at Andrews St"},
    8193637: {"QMSC1": "SLO Creek at Marsh St", "SKSC1": "SLO Creek at Elks Lane"},
    8192679: {"NPOC1": "Stenner Creek at Nipomo St"},
}

# Local timezone is Pacific Standard Time (UTC-8) in November
LOCAL_TZ = "America/Los_Angeles"

# Forecast initialization times of interest (local time)
INIT_TIMES_LOCAL = [
    "2026-11-18 06:00",
    "2026-11-18 12:00",
    "2026-11-18 18:00",
    "2026-11-18 22:00",
]

# Start of the time axis on the forecast plots (local time)
PLOT_START_LOCAL = "2026-11-18 09:00"
PLOT_END_LOCAL = "2026-11-19 12:00"

# NWS alerts drawn as vertical lines on the forecast plots:
# the local time each took effect, and its line style (both lines are blue)
NWS_ALERTS = {
    "Flood Watch": ("2026-11-18 15:00", "--"),
    "Flash Flood Warning": ("2026-11-18 22:43", "-"),
}
ALERT_LINE = {"color": "blue", "linewidth": 3, "alpha": 0.4}

# Reaches whose plots show the alerts (Paso Robles is too far away)
ALERT_REACHES = [8193611, 8193637, 8192679, 8192681]

# Short-range forecast lead hours (CONUS, 18-hours, hourly)
LEAD_HOURS = range(1, 19)

# Variables of interest
VARIABLES = ["streamflow"]  # streamflow (m3/s)

# NOAA bucket
BUCKET = "noaa-nwm-pds"

# Output directory
OUT_DIR = Path("data")

# Event name
EVENT = "slo_20261118"

# Color map for initializations
INIT_CMAP = "YlOrRd"

# Conversion factor from cubic meters per second to cubic feet per second
CMS_TO_CFS = 35.3146667

# Lambert conformal grid of the NWM forcing files (1 km cells)
NWM_PROJ = (
    "+proj=lcc +a=6370000 +b=6370000 +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 "
    "+x_0=0 +y_0=0 +units=m +no_defs"
)

# Reaches drawn on the basin map (the Salinas River at Paso Robles is left out)
MAP_REACHES = [8193611, 8193637, 8192679, 8192681]

# Basin colours on the map: a gradient from the largest basin to the smallest
BASIN_COLORS = ("dodgerblue", "purple")

# Basemap for the basin map: USGS topographic tiles (public domain)
BASEMAP_URL = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/"
    "tile/{zoom}/{row}/{col}"
)
BASEMAP_ZOOM = 13  # one level up doubles the detail and quadruples the tiles

OUT_DIR.mkdir(exist_ok=True)
fs = s3fs.S3FileSystem(anon=True)  # public bucket, no AWS credentials needed


def to_local(utc: pd.Series) -> pd.Series:
    """Naive UTC timestamps -> timezone-aware local timestamps."""
    return utc.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)


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


# %% Build the lists of files to read
# NOAA bucket layout (all times UTC):

# Convert local times to UTC
init_times_utc = (
    pd.to_datetime(INIT_TIMES_LOCAL).tz_localize(LOCAL_TZ).tz_convert("UTC")
)
first_valid_utc = init_times_utc.min() + pd.Timedelta(hours=min(LEAD_HOURS))
last_valid_utc = init_times_utc.max() + pd.Timedelta(hours=max(LEAD_HOURS))

def streamflow_key(init_utc: pd.Timestamp, lead_hour: int) -> str:
    # Streamflow forecast file: nwm.YYYYMMDD/short_range/nwm.tHHz.short_range.channel_rt.fFFF.conus.nc
    return (
        f"{BUCKET}/nwm.{init_utc:%Y%m%d}/short_range/"
        f"nwm.t{init_utc:%H}z.short_range.channel_rt.f{lead_hour:03d}.conus.nc"
    )


def forecast_rain_key(init_utc: pd.Timestamp, lead_hour: int) -> str:
    # Rainfall forecast file: nwm.YYYYMMDD/forcing_short_range/nwm.tHHz.short_range.forcing.fFFF.conus.nc
    return (
        f"{BUCKET}/nwm.{init_utc:%Y%m%d}/forcing_short_range/"
        f"nwm.t{init_utc:%H}z.short_range.forcing.f{lead_hour:03d}.conus.nc"
    )


def analysis_rain_key(valid_utc: pd.Timestamp) -> str:
    # Analysis cycle looks back 2 hours. The "tm02" file (valid 2 hours before the cycle time) is the last and most complete analysis of that hour.
    # Analysis file: nwm.YYYYMMDD/forcing_analysis_assim/nwm.tHHz.analysis_assim.forcing.tm02.conus.nc
    cycle = valid_utc + pd.Timedelta(hours=2)
    return (
        f"{BUCKET}/nwm.{cycle:%Y%m%d}/forcing_analysis_assim/"
        f"nwm.t{cycle:%H}z.analysis_assim.forcing.tm02.conus.nc"
    )


# Build the lists of files to read
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

# %% Check that every file is in the NOAA bucket
all_keys = streamflow_keys + forecast_rain_keys + analysis_rain_keys
in_bucket = set()
for folder in {key.rsplit("/", 1)[0] for key in all_keys}:
    in_bucket.update(fs.ls(folder, detail=False))

missing = sorted(set(all_keys) - in_bucket)
if missing:
    raise FileNotFoundError(f"{len(missing)} files not in NOAA bucket, e.g. {missing[0]}")
print(f"All {len(all_keys)} files found in NOAA bucket.")

# %% Streamflow forecasts: pull the reaches of interest out of each file (takes about a while to run)

def read_streamflow(key: str) -> pd.DataFrame:
    with fs.open(key, "rb") as f, xr.open_dataset(f, engine="h5netcdf") as ds:
        df = ds[VARIABLES].sel(feature_id=list(REACHES)).load().to_dataframe()
        df = df.reset_index()
        df["init_utc"] = ds["reference_time"].values[0]
        df["valid_utc"] = ds["time"].values[0]
    return df

with ThreadPoolExecutor(max_workers=8) as pool:
    forecasts = pd.concat(pool.map(read_streamflow, streamflow_keys), ignore_index=True)

# Add the reach name and lead hour to the forecasts
forecasts["reach"] = forecasts["feature_id"].map(REACHES)
forecasts["lead_hour"] = (
    (forecasts["valid_utc"] - forecasts["init_utc"]) / pd.Timedelta(hours=1)
).astype(int)

# Convert the initialization and valid times to local time
forecasts["init_local"] = to_local(forecasts["init_utc"])
forecasts["valid_local"] = to_local(forecasts["valid_utc"])

# Rename the variables and convert to cubic feet per second
forecasts = forecasts.rename(columns={"streamflow": "streamflow_cms"})
forecasts["streamflow_cfs"] = forecasts["streamflow_cms"] * CMS_TO_CFS

# Sort the forecasts by feature_id, initialization time, and lead hour
forecasts = forecasts.sort_values(["feature_id", "init_utc", "lead_hour"])
forecasts = forecasts.reset_index(drop=True)

# Save the forecasts to a CSV file
forecasts.to_csv(OUT_DIR / f"nwm_short_range_{EVENT}.csv", index=False)
print(f"{len(forecasts)} streamflow forecast rows")
forecasts.head()

# %% Basin upstream of each reach, on the NWM forcing grid (takes about a minute to run)
# Basin outlines come from the USGS NLDI service, in lon/lat.
to_nwm_grid = pyproj.Proj(NWM_PROJ)

basins = {}  # on the NWM grid, for the basin-average precipitation
basins_lonlat = {}  # in lon/lat, for the map
for feature_id, name in REACHES.items():
    response = requests.get(
        f"https://api.water.usgs.gov/nldi/linked-data/comid/{feature_id}/basin",
        timeout=120,
    )
    response.raise_for_status()
    # Get the basin outline from the USGS NLDI service
    outline = shapely.geometry.shape(response.json()["features"][0]["geometry"])
    basins_lonlat[feature_id] = outline
    # Transform the outline to the NWM grid
    basins[feature_id] = shapely.transform(
        outline, lambda lonlat: np.column_stack(to_nwm_grid(lonlat[:, 0], lonlat[:, 1]))
    )
    # Print the area of the basin in square kilometers
    print(f"{name}: {basins[feature_id].area / 1e6:,.0f} km2 upstream")

# Window of the grid that covers every basin, with a one-cell margin
basin_bounds = np.array([basin.bounds for basin in basins.values()])
x_min, y_min = basin_bounds[:, :2].min(axis=0) - 1000
x_max, y_max = basin_bounds[:, 2:].max(axis=0) + 1000

# Save the basin bounds to a CSV file (rows are in the same order as `basins`)
basin_bounds_df = pd.DataFrame(
    basin_bounds, columns=["x_min", "y_min", "x_max", "y_max"]
)
basin_bounds_df.insert(0, "feature_id", list(basins))
basin_bounds_df.insert(1, "reach", basin_bounds_df["feature_id"].map(REACHES))
basin_bounds_df.to_csv(OUT_DIR / f"nwm_basin_bounds_{EVENT}.csv", index=False)

# %% Map: basins, NWM forecast points and NWS gauges of the San Luis Obispo reaches
# An NWM reach reports the flow leaving its downstream end, so that end is the
# point marked for each reach. The map is drawn in Web Mercator, the projection
# of the basemap tiles, with the axes labelled in degrees.
WEB_MERCATOR_WIDTH = 2 * np.pi * 6378137  # metres around the equator
to_web_mercator = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def lonlat_to_map(geometry: shapely.Geometry) -> shapely.Geometry:
    return shapely.transform(
        geometry,
        lambda lonlat: np.column_stack(
            to_web_mercator.transform(lonlat[:, 0], lonlat[:, 1])
        ),
    )


def read_flowline(feature_id: int) -> shapely.LineString:
    """Channel line of a reach from the USGS NLDI service, upstream to downstream."""
    response = requests.get(
        f"https://api.water.usgs.gov/nldi/linked-data/comid/{feature_id}",
        timeout=120,
    )
    response.raise_for_status()
    return shapely.geometry.shape(response.json()["features"][0]["geometry"])


def read_nws_location(gauge: str) -> tuple[float, float]:
    """Longitude and latitude of an NWS gauge from the NWPS service."""
    response = requests.get(
        f"https://api.water.noaa.gov/nwps/v1/gauges/{gauge}", timeout=120
    )
    response.raise_for_status()
    info = response.json()
    return info["longitude"], info["latitude"]


def read_basemap(
    west: float, south: float, east: float, north: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Basemap tiles covering a Web Mercator window: one image and its extent."""
    tile_width = WEB_MERCATOR_WIDTH / 2**BASEMAP_ZOOM
    half = WEB_MERCATOR_WIDTH / 2
    # Tile columns count east from 180 W, tile rows count south from the top
    cols = range(int((west + half) // tile_width), int((east + half) // tile_width) + 1)
    rows = range(int((half - north) // tile_width), int((half - south) // tile_width) + 1)

    def read_tile(row_col: tuple[int, int]) -> np.ndarray:
        row, col = row_col
        response = requests.get(
            BASEMAP_URL.format(zoom=BASEMAP_ZOOM, row=row, col=col), timeout=120
        )
        response.raise_for_status()
        return plt.imread(io.BytesIO(response.content), format="jpeg")

    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = list(pool.map(read_tile, [(row, col) for row in rows for col in cols]))
    image = np.vstack(
        [np.hstack(tiles[i : i + len(cols)]) for i in range(0, len(tiles), len(cols))]
    )
    extent = (
        cols[0] * tile_width - half,
        (cols[-1] + 1) * tile_width - half,
        half - (rows[-1] + 1) * tile_width,
        half - rows[0] * tile_width,
    )
    return image, extent


# Map window: every mapped basin plus a margin of 0.01 degrees (about 1 km)
map_bounds = np.array([basins_lonlat[feature_id].bounds for feature_id in MAP_REACHES])
lon_min, lat_min = map_bounds[:, :2].min(axis=0) - 0.01
lon_max, lat_max = map_bounds[:, 2:].max(axis=0) + 0.01
(west, east), (south, north) = to_web_mercator.transform(
    [lon_min, lon_max], [lat_min, lat_max]
)

fig, ax = plt.subplots(figsize=(8, 9))
basemap, basemap_extent = read_basemap(west, south, east, north)
ax.imshow(basemap, extent=basemap_extent)

# Largest basin first, so the basins nested inside it are drawn on top
largest_first = sorted(MAP_REACHES, key=lambda f: basins_lonlat[f].area, reverse=True)
basin_colors = np.linspace(
    mcolors.to_rgb(BASIN_COLORS[0]), mcolors.to_rgb(BASIN_COLORS[1]), len(largest_first)
)
for feature_id, color in zip(largest_first, basin_colors):
    basin = lonlat_to_map(basins_lonlat[feature_id])
    flowline = lonlat_to_map(read_flowline(feature_id))
    ax.fill(*basin.exterior.xy, color=color, alpha=0.12)
    ax.plot(*basin.exterior.xy, color=color, linewidth=2, label=REACHES[feature_id])
    ax.plot(*flowline.xy, color=color, linewidth=3.5)
    ax.plot(
        *flowline.coords[-1],  # downstream end of the reach
        marker="o",
        markersize=9,
        color=color,
        markeredgecolor="black",
    )

# NWS gauges on the mapped reaches, each labelled with its name and gauge ID
gauge_style = {"marker": "^", "markersize": 9, "color": "gold", "markeredgecolor": "black"}
for feature_id in MAP_REACHES:
    for gauge_id, gauge_name in NWS_GAUGES.get(feature_id, {}).items():
        gauge_x, gauge_y = to_web_mercator.transform(*read_nws_location(gauge_id))
        ax.plot(gauge_x, gauge_y, **gauge_style)
        ax.annotate(
            f"{gauge_name} ({gauge_id})",
            (gauge_x, gauge_y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize="small",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )

# Legend entries for the line styles, which are shared by every reach
ax.plot([], [], color="black", linewidth=2, label="Basin upstream of the reach")
ax.plot([], [], color="black", linewidth=3.5, label="NWM reach")
ax.plot(
    [], [], "o", markersize=9, color="white", markeredgecolor="black",
    label="NWM forecast point",
)
ax.plot([], [], linestyle="none", label="NWS gauge", **gauge_style)

# Axes in degrees: in Web Mercator x depends only on longitude, y only on latitude
lon_ticks = np.arange(np.ceil(lon_min / 0.05), np.floor(lon_max / 0.05) + 1) * 0.05
lat_ticks = np.arange(np.ceil(lat_min / 0.05), np.floor(lat_max / 0.05) + 1) * 0.05
ax.set_xticks(
    to_web_mercator.transform(lon_ticks, np.zeros_like(lon_ticks))[0],
    [f"{lon:.2f}" for lon in lon_ticks],
)
ax.set_yticks(
    to_web_mercator.transform(np.zeros_like(lat_ticks), lat_ticks)[1],
    [f"{lat:.2f}" for lat in lat_ticks],
)
ax.set_xlabel("Longitude (degrees east)")
ax.set_ylabel("Latitude (degrees north)")
ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.set_title("NWM reaches and their upstream basins, San Luis Obispo")
ax.annotate(
    "Basemap: USGS The National Map",
    xy=(1, 0),
    xycoords="axes fraction",
    xytext=(-4, 4),
    textcoords="offset points",
    ha="right",
    fontsize="x-small",
)
fig.tight_layout()
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize="small")
save_figure(fig, OUT_DIR / f"nwm_basin_map_{EVENT}.png")

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

# Read from past output files, so this cell and the plots can run without the
# download cells above. CSV files hold times as text, so the UTC columns are
# parsed back into timestamps and the local times are rebuilt from them.
forecasts = pd.read_csv(
    OUT_DIR / f"nwm_short_range_{EVENT}.csv", parse_dates=["init_utc", "valid_utc"]
)
forecasts["init_local"] = to_local(forecasts["init_utc"])
forecasts["valid_local"] = to_local(forecasts["valid_utc"])

observed = pd.read_csv(OUT_DIR / f"observed_{EVENT}.csv", parse_dates=["valid_utc"])
observed["valid_local"] = observed["valid_utc"].dt.tz_convert(LOCAL_TZ)

# Forecast and analysis rain share one file, told apart by the "source" column
precip = pd.read_csv(
    OUT_DIR / f"nwm_basin_precip_{EVENT}.csv", parse_dates=["init_utc", "valid_utc"]
)
precip["init_local"] = to_local(precip["init_utc"])
precip["valid_local"] = to_local(precip["valid_utc"])
precip_forecast = precip[precip["source"] == "forecast"]
precip_analysis = precip[precip["source"] == "analysis"]


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


plot_start = pd.Timestamp(PLOT_START_LOCAL)
plot_end = pd.Timestamp(PLOT_END_LOCAL)

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
        2, 1, figsize=(7, 5), sharex=True, height_ratios=[1, 2]
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
    ax_rain.set_ylabel("Precipitation (mm/h)")
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
            labels.append(f"Observed discharge,\n{gauge_name} ({gauge})")
        elif obs["stage_ft"].notna().any():
            if ax_stage is None:
                ax_stage = ax_flow.twinx()
                ax_stage.set_ylabel("Stage rise above lowest (ft)")
            # The window is the plotted one, so the lowest stage is taken from it
            plotted = clock(obs["valid_local"]) >= plot_start
            handles += ax_stage.plot(
                clock(obs["valid_local"]),
                obs["stage_ft"] - obs.loc[plotted, "stage_ft"].min(),
                color="black",
                linestyle=next(stage_styles),
            )
            labels.append(f"Observed stage,\n{gauge_name} ({gauge})")

    # NWS alerts: a vertical line on both panels at the time each took effect,
    # named beside the line at the top of the streamflow panel
    alerts = NWS_ALERTS if feature_id in ALERT_REACHES else {}
    for alert, (alert_start, linestyle) in alerts.items():
        alert_start = pd.Timestamp(alert_start)
        ax_rain.axvline(alert_start, linestyle=linestyle, **ALERT_LINE)
        ax_flow.axvline(alert_start, linestyle=linestyle, **ALERT_LINE)
        ax_flow.annotate(
            f"{alert}, {alert_start:%I:%M %p}",
            xy=(alert_start, 1),  # at the line, top of the panel
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            color=ALERT_LINE["color"],
            fontsize="small",
        )
    if alerts:
        # Headroom above the hydrographs, so the names never sit on the curves
        for ax in filter(None, (ax_flow, ax_stage)):
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, top + 0.1 * (top - bottom))

    ax_flow.set_xlabel("Local time (PST)")
    ax_flow.set_xlim(left=plot_start, right=plot_end)  # the rain panel shares this axis
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
