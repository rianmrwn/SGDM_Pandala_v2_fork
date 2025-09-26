import os
import math
import time
import requests
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
import argparse
from tqdm import tqdm
from io import BytesIO
import pyproj

class GoogleSatelliteDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.tile_size = 256  # 256x256 tiles
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # Set user agent to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def lat_lon_to_tile(self, lat, lon, zoom):
        """Convert latitude, longitude to tile coordinates"""
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
        return x, y
    
    def calculate_zoom_from_resolution(self, target_resolution_meters, latitude):
        """
        Calculate optimal zoom level to achieve target resolution in meters/pixel
        Accounts for Web Mercator distortion at different latitudes
        """
        # Web Mercator ground resolution at equator for zoom 0: ~156543 meters/pixel
        equator_resolution = 156543.03392804097
        
        # Adjust for latitude (Web Mercator distortion)
        lat_factor = math.cos(math.radians(abs(latitude)))
        
        # Calculate theoretical zoom level for exact resolution
        theoretical_zoom = math.log2(equator_resolution * lat_factor / target_resolution_meters)
        
        # For target resolution, we need to ensure we're using a zoom level that gives ≤ target_resolution m/px
        zoom_level = math.ceil(theoretical_zoom)
        
        # Cap at maximum available zoom (Google typically goes up to 20-21)
        zoom_level = min(21, max(1, zoom_level))
        
        # Calculate the actual resolution we'll get with this zoom level
        actual_resolution = equator_resolution * lat_factor / (2 ** zoom_level)
        
        print(f"Target resolution: {target_resolution_meters} meters/pixel")
        print(f"Calculated zoom level: {zoom_level}")
        print(f"Actual resolution at zoom {zoom_level}: {actual_resolution:.2f} meters/pixel")
        
        return zoom_level, actual_resolution

    def get_tile_url(self, x, y, zoom):
        """Generate URL for a tile based on its coordinates"""
        # Using Google Satellite API
        return f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"

    def download_tile(self, x, y, zoom):
        """Download a single tile from Google Satellite"""
        url = self.get_tile_url(x, y, zoom)
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content))
                else:
                    print(f"Failed to download tile ({x}, {y}, {zoom}), status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading tile ({x}, {y}, {zoom}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        # Return a blank tile if all attempts fail
        return Image.new('RGB', (self.tile_size, self.tile_size), (255, 255, 255))

    def download_area(self, center_lat, center_lon, width_meters, height_meters, resolution_meters, output_path):
        """
        Download satellite imagery for a rectangular area
        
        Parameters:
        - center_lat, center_lon: Center coordinates of the area
        - width_meters, height_meters: Width and height of the area in meters
        - resolution_meters: Target resolution in meters per pixel
        - output_path: Path to save the output GeoTIFF
        """
        # Ensure minimum resolution is 0.75 m/px as requested
        resolution_meters = max(0.75, resolution_meters)
        
        # Calculate optimal zoom level for requested resolution
        zoom_level, actual_resolution = self.calculate_zoom_from_resolution(resolution_meters, center_lat)
        
        # Calculate image dimensions based on actual resolution
        width_pixels = int(width_meters / actual_resolution)
        height_pixels = int(height_meters / actual_resolution)
        
        # Ensure dimensions are even for better tile alignment
        width_pixels = width_pixels if width_pixels % 2 == 0 else width_pixels + 1
        height_pixels = height_pixels if height_pixels % 2 == 0 else height_pixels + 1
        
        print(f"Area dimensions: {width_meters}m x {height_meters}m")
        print(f"Image dimensions: {width_pixels}px x {height_pixels}px")
        
        # Calculate the distance in degrees
        meters_per_degree_lat = 111320  # Approximate meters per degree of latitude
        meters_per_degree_lon = 111320 * math.cos(math.radians(center_lat))
        
        # Calculate bounding box
        lat_offset = (height_meters / 2) / meters_per_degree_lat
        lon_offset = (width_meters / 2) / meters_per_degree_lon
        
        min_lat = center_lat - lat_offset
        max_lat = center_lat + lat_offset
        min_lon = center_lon - lon_offset
        max_lon = center_lon + lon_offset
        
        # Convert to tile coordinates
        min_tile_x, max_tile_y = self.lat_lon_to_tile(min_lat, min_lon, zoom_level)
        max_tile_x, min_tile_y = self.lat_lon_to_tile(max_lat, max_lon, zoom_level)
        
        # Ensure we have enough tiles to cover the area
        tiles_x = max_tile_x - min_tile_x + 1
        tiles_y = max_tile_y - min_tile_y + 1
        
        print(f"Downloading {tiles_x}x{tiles_y} tiles at zoom level {zoom_level}")
        
        # Create a blank image to hold all the tiles
        total_width = tiles_x * self.tile_size
        total_height = tiles_y * self.tile_size
        combined_image = Image.new('RGB', (total_width, total_height))
        
        # Download and combine tiles
        for y in tqdm(range(min_tile_y, max_tile_y + 1), desc="Downloading rows"):
            for x in range(min_tile_x, max_tile_x + 1):
                tile = self.download_tile(x, y, zoom_level)
                
                # Calculate position in the combined image
                pos_x = (x - min_tile_x) * self.tile_size
                pos_y = (y - min_tile_y) * self.tile_size
                
                # Paste the tile into the combined image
                combined_image.paste(tile, (pos_x, pos_y))
                
                # Add a small delay to prevent rate limiting
                time.sleep(0.1)
        
        # Calculate the exact pixel coordinates of the requested area within the combined image
        world_size = 2 ** zoom_level * self.tile_size
        center_x_pixel = (center_lon + 180) / 360 * world_size
        center_y_pixel = (1 - math.log(math.tan(math.radians(center_lat)) + 1 / math.cos(math.radians(center_lat))) / math.pi) / 2 * world_size
        
        # Calculate the offset from the top-left corner of our combined image
        offset_x = center_x_pixel - (min_tile_x * self.tile_size)
        offset_y = center_y_pixel - (min_tile_y * self.tile_size)
        
        # Calculate the crop box to extract exactly the requested area
        left = int(offset_x - width_pixels / 2)
        top = int(offset_y - height_pixels / 2)
        right = int(offset_x + width_pixels / 2)
        bottom = int(offset_y + height_pixels / 2)
        
        # Ensure crop box is within the combined image
        left = max(0, left)
        top = max(0, top)
        right = min(total_width, right)
        bottom = min(total_height, bottom)
        
        # Crop the image to the requested area
        cropped_image = combined_image.crop((left, top, right, bottom))
        
        # Save as GeoTIFF with proper georeference
        array = np.array(cropped_image)
        
        # Determine UTM zone from longitude
        utm_zone = int((center_lon + 180) / 6) + 1
        hemisphere = 'north' if center_lat >= 0 else 'south'
        
        # Create UTM CRS string
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        
        # Convert center coordinates to UTM
        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        center_x_utm, center_y_utm = transformer.transform(center_lon, center_lat)
        
        # Calculate the top-left corner in UTM coordinates
        left_utm = center_x_utm - (width_meters / 2)
        top_utm = center_y_utm + (height_meters / 2)
        
        # Create GeoTIFF transform (from the top-left corner)
        pixel_width = width_meters / array.shape[1]
        pixel_height = height_meters / array.shape[0]
        transform = from_origin(left_utm, top_utm, pixel_width, pixel_height)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=array.shape[0],
            width=array.shape[1],
            count=3,
            dtype=array.dtype,
            crs=utm_crs,
            transform=transform,
        ) as dst:
            # Write each channel separately
            for i in range(3):
                dst.write(array[:, :, i], i + 1)
        
        print(f"Saved GeoTIFF to {output_path}")
        print(f"Final image size: {array.shape}")
        print(f"Final output resolution: {pixel_width:.2f} meters/pixel")
        print(f"UTM Zone: {utm_zone} {hemisphere}")
        
        return array, pixel_width

    def download_from_reference_tif(self, reference_tif_path, output_path, resolution_meters=None):
        """
        Download imagery based on the extent of a reference GeoTIFF
        
        Parameters:
        - reference_tif_path: Path to reference GeoTIFF
        - output_path: Path to save the output GeoTIFF
        - resolution_meters: Target resolution in meters per pixel (optional)
        """
        with rasterio.open(reference_tif_path) as src:
            # Get the bounds of the reference GeoTIFF
            bounds = src.bounds
            
            # Check if the reference is already in a UTM projection
            is_utm = False
            if src.crs:
                is_utm = '+proj=utm' in src.crs.to_proj4()
            
            if is_utm:
                print("Reference GeoTIFF is already in UTM projection")
                # Get the center in UTM coordinates
                center_x_utm = (bounds.left + bounds.right) / 2
                center_y_utm = (bounds.bottom + bounds.top) / 2
                
                # Calculate width and height in meters directly
                width_meters = bounds.right - bounds.left
                height_meters = bounds.top - bounds.bottom
                
                # Convert center to lat/lon for tile download
                transformer = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                center_lon, center_lat = transformer.transform(center_x_utm, center_y_utm)
            else:
                print("Reference GeoTIFF is in a non-UTM projection, assuming WGS84")
                # Calculate center coordinates
                center_lon = (bounds.left + bounds.right) / 2
                center_lat = (bounds.bottom + bounds.top) / 2
                
                # Calculate width and height in meters
                meters_per_degree_lat = 111320
                meters_per_degree_lon = 111320 * math.cos(math.radians(center_lat))
                
                width_meters = (bounds.right - bounds.left) * meters_per_degree_lon
                height_meters = (bounds.top - bounds.bottom) * meters_per_degree_lat
            
            # Use the reference resolution if no target resolution is specified
            if resolution_meters is None:
                if is_utm:
                    # Calculate resolution directly from UTM units
                    resolution_meters = max(
                        (bounds.right - bounds.left) / src.width,
                        (bounds.top - bounds.bottom) / src.height
                    )
                else:
                    # Calculate approximate resolution from the reference GeoTIFF
                    resolution_meters = max(
                        (bounds.right - bounds.left) / src.width * meters_per_degree_lon,
                        (bounds.top - bounds.bottom) / src.height * meters_per_degree_lat
                    )
            
            # Ensure minimum resolution is 0.75 m/px as requested
            resolution_meters = max(0.75, resolution_meters)
            
            print(f"Reference GeoTIFF bounds: {bounds}")
            print(f"Center coordinates: {center_lat}, {center_lon}")
            print(f"Area dimensions: {width_meters:.2f}m x {height_meters:.2f}m")
            print(f"Target resolution: {resolution_meters} meters/pixel")
            
            # Download the area
            return self.download_area(
                center_lat, center_lon,
                width_meters, height_meters,
                resolution_meters, output_path
            )

def main():
    parser = argparse.ArgumentParser(description='Download high-resolution satellite imagery from Google Satellite')
    parser.add_argument('--center_lat', type=float, help='Center latitude')
    parser.add_argument('--center_lon', type=float, help='Center longitude')
    parser.add_argument('--width', type=float, help='Width in meters')
    parser.add_argument('--height', type=float, help='Height in meters')
    parser.add_argument('--resolution', type=float, default=0.75, help='Resolution in meters per pixel (default: 0.75)')
    parser.add_argument('--output', type=str, default='/data_img/data_prep/HR.tif', help='Output GeoTIFF path')
    parser.add_argument('--input_tif_reference', type=str, help='Reference GeoTIFF for area and resolution')
    
    args = parser.parse_args()
    
    downloader = GoogleSatelliteDownloader()
    
    if args.input_tif_reference:
        # Download based on reference GeoTIFF
        downloader.download_from_reference_tif(
            args.input_tif_reference,
            args.output,
            args.resolution
        )
    elif args.center_lat and args.center_lon and args.width and args.height:
        # Download based on explicit parameters
        downloader.download_area(
            args.center_lat, args.center_lon,
            args.width, args.height,
            args.resolution, args.output
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

