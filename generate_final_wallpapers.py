#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add current dir to sys.path to import make_wallpaper
sys.path.append(str(Path(__file__).resolve().parent))

try:
    import make_wallpaper as wp
except ImportError as e:
    print(f"Error importing make_wallpaper: {e}")
    sys.exit(1)

def generate_orenburg():
    print("\n--- Generating Orenburg ---")
    city_raw = "Оренбург"
    city_lines = ["ОРЕНБУРГ"]
    region_code = "56"
    subtitles = ["СЕРДЦЕ ЕВРАЗИИ", "ОРЕНБУРГСКАЯ ОБЛАСТЬ · РОССИЯ"]
    lat, lon = 51.7681, 55.0970
    radius_km = 45 # A bit tighter for the city
    
    # Credentials from .env
    env = wp._load_env()
    client_id = env.get("SH_CLIENT_ID_SECOND_RESERVE") or env.get("SH_CLIENT_ID_RESERVE") or env.get("SH_CLIENT_ID", "")
    client_secret = env.get("SH_CLIENT_SECRET_SECOND_RESERVE") or env.get("SH_CLIENT_SECRET_RESERVE") or env.get("SH_CLIENT_SECRET", "")
    
    if not client_id:
        print("Missing SentinelHub credentials in .env")
        return

    slug = wp._slug(city_raw, region_code)
    tile_path = wp.TILES_DIR / f"{slug}.npz"
    
    if not tile_path.exists():
        print(f"Downloading tile for {city_raw}...")
        wp.download_tile(lat, lon, radius_km, client_id, client_secret, tile_path)

    t = wp.load_tile(tile_path, wp.WP_W, wp.WP_H, zoom=1.45, cx_frac=0.5, cy_frac=0.5)
    phi = wp._to_dms(lat, is_lat=True)
    lam = wp._to_dms(lon, is_lat=False)

    configs = [
        dict(render=wp.render_edge,     cmap=wp.EDGE_CMAP,     suffix="edge",
             active="edge", bar="Sentinel-2  ·  edge_composite  ·  нормализованная интенсивность"),
        dict(render=wp.render_ndvi,     cmap=wp.NDVI_CMAP,     suffix="ndvi",
             active="ndvi", bar="Sentinel-2  ·  NDVI max  ·  индекс растительности"),
        dict(render=wp.render_ndvi_var, cmap=wp.NDVI_VAR_CMAP, suffix="ndvi_var",
             active="var",  bar="Sentinel-2  ·  NDVI STD  ·  межсезонная изменчивость"),
    ]

    for cfg in configs:
        layer = cfg["render"](t, cfg["cmap"])
        out_path = wp.OUT_DIR / f"wallpaper_{slug}_{cfg['suffix']}.png"
        wp.build_wallpaper(
            layer=layer, cmap=cfg["cmap"], out_path=out_path,
            city_lines=city_lines, region_code=region_code, subtitles=subtitles,
            phi=phi, lam=lam, bar_label=cfg["bar"],
            active_ch=cfg["active"], digit_dx=0
        )

def update_spb():
    print("\n--- Updating Saint Petersburg ---")
    city_raw = "spb" # Keep original slug to overwrite
    city_lines = ["САНКТ-", "ПЕТЕРБУРГ"]
    region_code = "78"
    subtitles = ["КУЛЬТУРНАЯ СТОЛИЦА", "СЕВЕРНАЯ ПАЛЬМИРА · РОССИЯ"]
    
    # New coordinates requested by user
    lat, lon = 59.929571, 30.296643
    radius_km = 40
    
    # User specifically asked for 2 decimal places precision for the coordinate display
    # We will format phi/lam as decimal degrees to satisfy "точность до двух знаков"
    # Actually, let's see if we should stick to DMS but with precise center, 
    # OR change the display to decimal as requested.
    # "переделай отображения точки на 59.929571, 30.296643 (ну с точностью до двух знаков после запятой)"
    # This strongly implies decimal display. 
    phi = f"φ  {lat:.2f}° N"
    lam = f"λ  {lon:.2f}° E"

    env = wp._load_env()
    client_id = env.get("SH_CLIENT_ID_SECOND_RESERVE") or env.get("SH_CLIENT_ID_RESERVE") or env.get("SH_CLIENT_ID", "")
    client_secret = env.get("SH_CLIENT_SECRET_SECOND_RESERVE") or env.get("SH_CLIENT_SECRET_RESERVE") or env.get("SH_CLIENT_SECRET", "")
    
    slug = wp._slug(city_raw, region_code)
    tile_path = wp.TILES_DIR / f"{slug}_updated.npz" # Use distinct tile for updated coords
    
    if not tile_path.exists():
        print(f"Downloading updated tile for SPB...")
        wp.download_tile(lat, lon, radius_km, client_id, client_secret, tile_path)

    t = wp.load_tile(tile_path, wp.WP_W, wp.WP_H, zoom=1.38, cx_frac=0.5, cy_frac=0.5)

    configs = [
        dict(render=wp.render_edge,     cmap=wp.EDGE_CMAP,     suffix="edge",
             active="edge", bar="Sentinel-2  ·  edge_composite  ·  нормализованная интенсивность"),
        dict(render=wp.render_ndvi,     cmap=wp.NDVI_CMAP,     suffix="ndvi",
             active="ndvi", bar="Sentinel-2  ·  NDVI max  ·  индекс растительности"),
        dict(render=wp.render_ndvi_var, cmap=wp.NDVI_VAR_CMAP, suffix="ndvi_var",
             active="var",  bar="Sentinel-2  ·  NDVI STD  ·  межсезонная изменчивость"),
    ]

    for cfg in configs:
        layer = cfg["render"](t, cfg["cmap"])
        out_path = wp.OUT_DIR / f"wallpaper_{slug}_{cfg['suffix']}.png"
        wp.build_wallpaper(
            layer=layer, cmap=cfg["cmap"], out_path=out_path,
            city_lines=city_lines, region_code=region_code, subtitles=subtitles,
            phi=phi, lam=lam, bar_label=cfg["bar"],
            active_ch=cfg["active"], digit_dx=0
        )

if __name__ == "__main__":
    generate_orenburg()
    update_spb()
    print("\nAll tasks completed.")
