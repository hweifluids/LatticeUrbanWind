#!/usr/bin/env python3
"""
Visualize DEM (Digital Elevation Model) TIFF file
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import sys

def visualize_dem(tif_path):
    """
    Visualize a DEM TIFF file with multiple views

    Args:
        tif_path: Path to the TIFF file
    """
    print(f"Loading DEM file: {tif_path}")

    # Read the DEM data
    with rasterio.open(tif_path) as src:
        dem_data = src.read(1)  # Read first band

        # Get original metadata
        print(f"\nOriginal DEM Metadata:")
        print(f"  Shape: {dem_data.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Resolution: {src.res}")
        print(f"  NoData value: {src.nodata}")

        # Replace nodata values with NaN (handle -9999 as nodata)
        dem_data = np.where(dem_data == -9999, np.nan, dem_data)
        if src.nodata is not None:
            dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)

        # Reproject to EPSG:4326 (WGS84)
        print("\nReprojecting to EPSG:4326 (WGS84)...")
        dst_crs = CRS.from_epsg(4326)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        # Create output array
        dem_data_4326 = np.empty((dst_height, dst_width), dtype=np.float32)

        # Reproject
        reproject(
            source=dem_data,
            destination=dem_data_4326,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

        # Use reprojected data
        dem_data = dem_data_4326

        # Calculate bounds in lat/lon
        lon_min = dst_transform.c
        lat_max = dst_transform.f
        lon_max = dst_transform.c + dst_width * dst_transform.a
        lat_min = dst_transform.f + dst_height * dst_transform.e

        print(f"\nReprojected DEM Metadata:")
        print(f"  Shape: {dem_data.shape}")
        print(f"  CRS: EPSG:4326")
        print(f"  Lat range: [{lat_min:.6f}, {lat_max:.6f}]")
        print(f"  Lon range: [{lon_min:.6f}, {lon_max:.6f}]")

        # Get statistics
        valid_data = dem_data[~np.isnan(dem_data)]
        if len(valid_data) > 0:
            print(f"\nElevation Statistics (after filtering -9999):")
            print(f"  Min: {np.min(valid_data):.2f} m")
            print(f"  Max: {np.max(valid_data):.2f} m")
            print(f"  Mean: {np.mean(valid_data):.2f} m")
            print(f"  Std: {np.std(valid_data):.2f} m")
            print(f"  Valid pixels: {len(valid_data)} / {dem_data.size} ({100*len(valid_data)/dem_data.size:.1f}%)")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))

        # Create extent for imshow (in lat/lon)
        extent = [lon_min, lon_max, lat_min, lat_max]

        # 1. Basic elevation map
        ax1 = plt.subplot(2, 2, 1)
        im1 = ax1.imshow(dem_data, cmap='terrain', interpolation='bilinear',
                         extent=extent, aspect='auto')
        ax1.set_title('Elevation Map (EPSG:4326)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Longitude (°)')
        ax1.set_ylabel('Latitude (°)')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Elevation (m)', rotation=270, labelpad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. Hillshade
        ax2 = plt.subplot(2, 2, 2)
        ls = LightSource(azdeg=315, altdeg=45)
        # Use approximate resolution in degrees
        dx_deg = abs(dst_transform.a)
        dy_deg = abs(dst_transform.e)
        hillshade = ls.hillshade(dem_data, vert_exag=2, dx=dx_deg, dy=dy_deg)
        ax2.imshow(hillshade, cmap='gray', interpolation='bilinear',
                   extent=extent, aspect='auto')
        ax2.set_title('Hillshade (Shaded Relief)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Longitude (°)')
        ax2.set_ylabel('Latitude (°)')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 3. Elevation with hillshade overlay
        ax3 = plt.subplot(2, 2, 3)
        ax3.imshow(hillshade, cmap='gray', alpha=0.5, interpolation='bilinear',
                   extent=extent, aspect='auto')
        im3 = ax3.imshow(dem_data, cmap='terrain', alpha=0.6, interpolation='bilinear',
                         extent=extent, aspect='auto')
        ax3.set_title('Elevation + Hillshade Overlay', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Longitude (°)')
        ax3.set_ylabel('Latitude (°)')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('Elevation (m)', rotation=270, labelpad=20)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # 4. Elevation histogram
        ax4 = plt.subplot(2, 2, 4)
        if len(valid_data) > 0:
            ax4.hist(valid_data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax4.set_title('Elevation Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Elevation (m)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Min: {np.min(valid_data):.1f} m\n'
            stats_text += f'Max: {np.max(valid_data):.1f} m\n'
            stats_text += f'Mean: {np.mean(valid_data):.1f} m\n'
            stats_text += f'Std: {np.std(valid_data):.1f} m'
            ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = tif_path.replace('.tif', '_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
        # Create a 3D surface plot
        print("\nGenerating 3D surface plot...")
        fig_3d = plt.figure(figsize=(14, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Downsample for 3D plot if data is too large
        step = max(1, min(dem_data.shape) // 200)

        # Create coordinate arrays in lat/lon
        lon_array = np.linspace(lon_min, lon_max, dem_data.shape[1])
        lat_array = np.linspace(lat_max, lat_min, dem_data.shape[0])  # Note: reversed for image coordinates

        lon_sampled = lon_array[::step]
        lat_sampled = lat_array[::step]

        X, Y = np.meshgrid(lon_sampled, lat_sampled)
        Z = dem_data[::step, ::step]

        # Plot surface
        surf = ax_3d.plot_surface(X, Y, Z, cmap='terrain',
                                   linewidth=0, antialiased=True, alpha=0.8)

        ax_3d.set_title('3D Terrain Surface (EPSG:4326)', fontsize=14, fontweight='bold', pad=20)
        ax_3d.set_xlabel('Longitude (°)', labelpad=10)
        ax_3d.set_ylabel('Latitude (°)', labelpad=10)
        ax_3d.set_zlabel('Elevation (m)', labelpad=10)

        # Add colorbar
        cbar_3d = fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5)
        cbar_3d.set_label('Elevation (m)', rotation=270, labelpad=20)

        # Save 3D plot
        output_3d_path = tif_path.replace('.tif', '_3d_visualization.png')
        plt.savefig(output_3d_path, dpi=300, bbox_inches='tight')
        print(f"3D visualization saved to: {output_3d_path}")

        plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tif_file = sys.argv[1]
    else:
        tif_file = "AST14DEM_00408232025023230_20251011022044.tif"
    
    visualize_dem(tif_file)

