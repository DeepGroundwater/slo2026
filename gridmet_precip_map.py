# %% [markdown]

# gridMET 48-hour precipitation map of San Luis Obispo County for the Flood Event on November 18, 2026
# - gridMET is a daily gridded dataset with 1/24 degree (about 4 km) cells: https://www.climatologylab.org/gridmet.html
# - A gridMET day ends at about midnight Mountain Standard Time (11 PM PST), so the
#   two days Nov 18 and Nov 19 cover Nov 17 11 PM to Nov 19 11 PM PST.
# - The newest days in gridMET are preliminary ("early", then "provisional") and can
#   change in later updates. The script prints which kind it has read.
# - This script stands alone: it fetches the outlines and basemap itself, in the
#   same way as nwm_short_range_hindcast.py.

# %% Imports
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
import shapely
import shapely.geometry
import xarray as xr

# %% Configurations

# County that sets the map window: San Luis Obispo County
# (Census GEOID: state 06 followed by county 079)
COUNTY_GEOID = "06079"
COUNTY_NAME = "San Luis Obispo County"

# County outlines from the Census Bureau's TIGERweb service
COUNTY_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "State_County/MapServer/1/query"
)

# NWM reaches (NHDPlus COMID) whose upstream basins are drawn on the map
REACHES = {
    8193611: "San Luis Obispo Creek (ID:8193611)",
    8193637: "San Luis Obispo Creek (ID:8193637)",
    8192679: "Stenner Creek (ID:8192679)",
    8192681: "Old Garden Creek (ID:8192681)",
}

# gridMET daily precipitation, one file per year
GRIDMET_URL = "https://www.northwestknowledge.net/metdata/data/pr_2026.nc"

# First and last gridMET day to add up (two days = 48 hours)
FIRST_DAY = "2026-11-18"
LAST_DAY = "2026-11-19"

# A gridMET day ends at midnight Mountain Standard Time: 07 UTC the next day
DAY_END_UTC_HOUR = 7

# Half the width of a gridMET cell, in degrees
HALF_CELL = 1 / 48

# gridMET precipitation is in millimetres; the map is in inches
MM_PER_INCH = 25.4

# Local timezone is Pacific Standard Time (UTC-8) in November
LOCAL_TZ = "America/Los_Angeles"

# Output directory
OUT_DIR = Path("data")

# Event name
EVENT = "slo_20261118"

# Colour map for precipitation
PRECIP_CMAP = "YlOrRd"

# Basin colours on the map: a gradient from the largest basin to the smallest
BASIN_COLORS = ("dodgerblue", "purple")

# Spacing of the longitude and latitude ticks, in degrees
TICK_STEP = 0.25

# Basemap: USGS topographic tiles (public domain)
BASEMAP_URL = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/"
    "tile/{zoom}/{row}/{col}"
)
BASEMAP_ZOOM = 10  # one level up doubles the detail and quadruples the tiles

OUT_DIR.mkdir(exist_ok=True)


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


# %% Outlines in lon/lat: the county, and the basin upstream of each reach
response = requests.get(
    COUNTY_URL,
    params={
        "where": f"GEOID='{COUNTY_GEOID}'",
        "outSR": "4326",  # lon/lat
        "f": "geojson",
    },
    timeout=120,
)
response.raise_for_status()
county = shapely.geometry.shape(response.json()["features"][0]["geometry"])

# Basin outlines come from the USGS NLDI service
basins = {}
for feature_id in REACHES:
    response = requests.get(
        f"https://api.water.usgs.gov/nldi/linked-data/comid/{feature_id}/basin",
        timeout=120,
    )
    response.raise_for_status()
    basins[feature_id] = shapely.geometry.shape(
        response.json()["features"][0]["geometry"]
    )

# Box around the county (its outline includes the coastal waters)
lon_min, lat_min, lon_max, lat_max = county.bounds

# %% gridMET precipitation: every cell that touches the box, added up over the days
# The file is read over the internet, and only the pieces holding these cells
# are downloaded. Coordinates are cell centres; latitude runs north to south.
with (
    fsspec.open(GRIDMET_URL, "rb", block_size=2**20, cache_type="blockcache") as f,
    xr.open_dataset(f, engine="h5netcdf") as ds,
):
    daily = ds["precipitation_amount"].sel(
        day=slice(FIRST_DAY, LAST_DAY),
        lat=slice(lat_max + HALF_CELL, lat_min - HALF_CELL),
        lon=slice(lon_min - HALF_CELL, lon_max + HALF_CELL),
    )
    daily = daily.load()  # mm per day
    newest_day = pd.Timestamp(ds["day"].values[-1])
    # The file numbers its days from 1 and records (as text) where the final
    # and the provisional data end
    last_slice = int((ds["day"] <= np.datetime64(LAST_DAY)).sum())
    if last_slice <= int(ds.attrs["last_permanent_slice"]):
        status = "final"
    elif last_slice <= int(ds.attrs["last_provisional_slice"]):
        status = "provisional"
    else:
        status = "early"

n_days = (pd.Timestamp(LAST_DAY) - pd.Timestamp(FIRST_DAY)).days + 1
if daily.sizes["day"] != n_days:
    raise ValueError(
        f"gridMET has {daily.sizes['day']} of the {n_days} days asked for; "
        f"its newest day is {newest_day:%Y-%m-%d}"
    )

# Inches. Cells over the ocean have no data and stay empty.
total = daily.sum("day", skipna=False) / MM_PER_INCH

# The period the days cover, in local time
period_start = pd.Timestamp(FIRST_DAY, tz="UTC") + pd.Timedelta(hours=DAY_END_UTC_HOUR)
period_end = period_start + pd.Timedelta(days=n_days)
period_start, period_end = (t.tz_convert(LOCAL_TZ) for t in (period_start, period_end))
period_hours = (period_end - period_start) / pd.Timedelta(hours=1)

print(f"gridMET {FIRST_DAY} to {LAST_DAY} ({status} data)")
print(f"Covers {period_start:%b %d %I %p} to {period_end:%b %d %I %p} local time")
print(
    f"{total.sizes['lat']} x {total.sizes['lon']} cells, "
    f"{float(total.min()):.2f} to {float(total.max()):.2f} in"
)

# %% Map: precipitation cells and outlines on a USGS topographic basemap
# The map is drawn in Web Mercator, the projection of the basemap tiles, with the
# axes labelled in degrees. In Web Mercator x depends only on longitude and y
# only on latitude, so the gridMET cells stay rectangles.
WEB_MERCATOR_WIDTH = 2 * np.pi * 6378137  # metres around the equator
to_web_mercator = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def lon_to_x(lon: np.ndarray) -> np.ndarray:
    return to_web_mercator.transform(lon, np.zeros_like(lon))[0]


def lat_to_y(lat: np.ndarray) -> np.ndarray:
    return to_web_mercator.transform(np.zeros_like(lat), lat)[1]


def outline_xy(polygon: shapely.Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Outer ring of a lon/lat polygon, in map coordinates."""
    ring_lon, ring_lat = (np.asarray(c) for c in polygon.exterior.xy)
    return lon_to_x(ring_lon), lat_to_y(ring_lat)


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


# Cell edges, half a cell either side of the centres
lon = total["lon"].values
lat = total["lat"].values
lon_edges = np.append(lon - HALF_CELL, lon[-1] + HALF_CELL)
lat_edges = np.append(lat + HALF_CELL, lat[-1] - HALF_CELL)  # north to south
x_edges = lon_to_x(lon_edges)
y_edges = lat_to_y(lat_edges)

# Map window: the whole of every cell read
west, east = x_edges.min(), x_edges.max()
south, north = y_edges.min(), y_edges.max()

fig, ax = plt.subplots(figsize=(11, 6.5))
basemap, basemap_extent = read_basemap(west, south, east, north)
ax.imshow(basemap, extent=basemap_extent)

# Precipitation: see-through cells over the basemap
cells = ax.pcolormesh(
    x_edges, y_edges, total.values, cmap=PRECIP_CMAP, vmin=0, alpha=0.65
)
fig.colorbar(
    cells, ax=ax, shrink=0.8, label=f"{period_hours:.0f}-hour precipitation (in)"
)

# County outline (a county with islands has several parts)
for i, part in enumerate(shapely.get_parts(county)):
    ax.plot(
        *outline_xy(part),
        color="black",
        linewidth=1.2,
        label=f"{COUNTY_NAME} boundary" if i == 0 else None,
    )

# Basin outlines: largest first, so the basins nested inside it are drawn on top
largest_first = sorted(REACHES, key=lambda f: basins[f].area, reverse=True)
basin_colors = np.linspace(
    mcolors.to_rgb(BASIN_COLORS[0]), mcolors.to_rgb(BASIN_COLORS[1]), len(largest_first)
)
for feature_id, color in zip(largest_first, basin_colors):
    ax.plot(
        *outline_xy(basins[feature_id]),
        color=color,
        linewidth=1.5,
        label=f"Basin of {REACHES[feature_id]}",
    )

# Axes in degrees
lon_ticks = (
    np.arange(np.ceil(lon_edges.min() / TICK_STEP), np.floor(lon_edges.max() / TICK_STEP) + 1)
    * TICK_STEP
)
lat_ticks = (
    np.arange(np.ceil(lat_edges.min() / TICK_STEP), np.floor(lat_edges.max() / TICK_STEP) + 1)
    * TICK_STEP
)
ax.set_xticks(lon_to_x(lon_ticks), [f"{tick:.2f}" for tick in lon_ticks])
ax.set_yticks(lat_to_y(lat_ticks), [f"{tick:.2f}" for tick in lat_ticks])
ax.set_xlabel("Longitude (degrees east)")
ax.set_ylabel("Latitude (degrees north)")
ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.set_title(
    f"gridMET precipitation, {COUNTY_NAME}\n"
    f"{period_start:%b %d %I %p} to {period_end:%b %d %I %p} PST"
)
ax.annotate(
    f"Precipitation: gridMET ({status} data). Basemap: USGS The National Map",
    xy=(1, 0),
    xycoords="axes fraction",
    xytext=(-4, 4),
    textcoords="offset points",
    ha="right",
    fontsize="x-small",
)
fig.tight_layout()
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize="small")
save_figure(fig, OUT_DIR / f"gridmet_precip_{period_hours:.0f}h_{EVENT}.png")

plt.show()
