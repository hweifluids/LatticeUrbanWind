from pathlib import Path
from typing import Tuple, Optional, Union
import re


def get_utm_zone_from_lon(lon: float) -> int:
    """
    Compute UTM zone number from longitude in degrees.
    """
    lon_val = float(lon)
    zone = int((lon_val + 180.0) // 6.0) + 1
    if zone < 1:
        zone = 1
    if zone > 60:
        zone = 60
    return zone


def get_utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """
    Return EPSG code string for WGS84 UTM zone based on lon and lat.
    Northern hemisphere uses 326xx, southern uses 327xx.
    """
    lat_val = float(lat)
    zone = get_utm_zone_from_lon(lon)
    if lat_val >= 0.0:
        code = 32600 + zone
    else:
        code = 32700 + zone
    return f"EPSG:{code}"


def _parse_lonlat_pairs_from_text(txt: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Parse cut_lon_manual and cut_lat_manual from config text.
    Returns ((lon_min, lon_max), (lat_min, lat_max)) or None if not both found.
    """
    m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", txt)
    m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", txt)
    if not (m_lon and m_lat):
        return None

    def _parse_pair(m) -> Tuple[float, float]:
        arr = [s.strip() for s in m.group(1).split(",")]
        vals = [float(v) for v in arr if v]
        if not vals:
            raise ValueError("Empty lon or lat list in config")
        vmin = min(vals)
        vmax = max(vals)
        return vmin, vmax

    lon_min, lon_max = _parse_pair(m_lon)
    lat_min, lat_max = _parse_pair(m_lat)
    return (lon_min, lon_max), (lat_min, lat_max)


def get_utm_crs_from_bounds(lon_pair: Tuple[float, float], lat_pair: Tuple[float, float]) -> str:
    """
    Determine UTM CRS from lon and lat bounds.
    """
    lon_min, lon_max = lon_pair
    lat_min, lat_max = lat_pair
    lon_center = 0.5 * (lon_min + lon_max)
    lat_center = 0.5 * (lat_min + lat_max)
    return get_utm_epsg_from_lonlat(lon_center, lat_center)


def get_utm_crs_from_conf(conf_path: Union[str, Path], default_epsg: Optional[str] = None) -> str:
    """
    Determine UTM CRS from a deck file that contains cut_lon_manual and cut_lat_manual.
    If they are missing and default_epsg is provided, return default_epsg.
    """
    conf_file = Path(conf_path).expanduser().resolve()
    if not conf_file.exists():
        raise FileNotFoundError(f"Cannot find config file: {conf_file}")
    txt = conf_file.read_text(encoding="utf-8", errors="ignore")
    pairs = _parse_lonlat_pairs_from_text(txt)
    if pairs is None:
        if default_epsg is not None:
            print(f"[auto_UTM] cut_lon_manual or cut_lat_manual not found, use default {default_epsg}")
            return default_epsg
        raise ValueError("Config file does not contain cut_lon_manual and cut_lat_manual")
    lon_pair, lat_pair = pairs
    return get_utm_crs_from_bounds(lon_pair, lat_pair)


def get_utm_crs_from_conf_raw(conf_raw: str, default_epsg: Optional[str] = None) -> str:
    """
    Determine UTM CRS from raw config text.
    """
    pairs = _parse_lonlat_pairs_from_text(conf_raw)
    if pairs is None:
        if default_epsg is not None:
            print(f"[auto_UTM] cut_lon_manual or cut_lat_manual not found in raw text, use default {default_epsg}")
            return default_epsg
        raise ValueError("Config raw text does not contain cut_lon_manual and cut_lat_manual")
    lon_pair, lat_pair = pairs
    return get_utm_crs_from_bounds(lon_pair, lat_pair)


__all__ = [
    "get_utm_zone_from_lon",
    "get_utm_epsg_from_lonlat",
    "get_utm_crs_from_bounds",
    "get_utm_crs_from_conf",
    "get_utm_crs_from_conf_raw",
]
