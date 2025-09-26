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


class VoyagerMapDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.tile_size = 512  # retina tiles are 512x512 pixels
        self.tile_scale = self.tile_size / 256
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.subdomains = ['a', 'b', 'c', 'd']

    def lat_lon_to_tile(self, lat, lon, zoom):
        """Convert latitude, longitude to tile coordinates"""
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
        return x, y

    def calculate_zoom_from_resolution(self, target_resolution_meters, latitude):
        """Estimate zoom level for desired ground resolution"""
        equator_resolution = 156543.03392804097  # meters/pixel at zoom 0 with 256px tiles
        lat_factor = math.cos(math.radians(abs(latitude)))
        theoretical_zoom = math.log2(
            equator_resolution * lat_factor / (target_resolution_meters * self.tile_scale)
        )
        zoom_level = math.ceil(theoretical_zoom)
        zoom_level = min(20, max(0, zoom_level))  # Carto Voyager tops out around z=20
        actual_resolution = equator_resolution * lat_factor / (2 ** zoom_level * self.tile_scale)
        print(f"Target resolution: {target_resolution_meters} meters/pixel")
        print(f"Calculated zoom level: {zoom_level}")
        print(f"Actual resolution at zoom {zoom_level}: {actual_resolution:.2f} meters/pixel")
        return zoom_level, actual_resolution

    def get_tile_url(self, x, y, zoom):
        """Build URL for the Carto Voyager No Labels retina tile"""
        subdomain = self.subdomains[(x + y) % len(self.subdomains)]
        return f"https://{subdomain}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{zoom}/{x}/{y}@2x.png"

    def download_tile(self, x, y, zoom):
        """Download a single tile and ensure an RGB image"""
        url = self.get_tile_url(x, y, zoom)
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    tile = Image.open(BytesIO(response.content)).convert('RGBA')
                    rgb_tile = Image.alpha_composite(Image.new('RGBA', tile.size, (255, 255, 255, 255)), tile)
                    return rgb_tile.convert('RGB')
                else:
                    print(f"Failed to download tile, status code: {response.status_code}")
            except Exception as exc:
                print(f"Error downloading tile {zoom}/{x}/{y}: {exc}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        return Image.new('RGB', (self.tile_size, self.tile_size), (255, 255, 255))

    def download_area(self, center_lat, center_lon, width_meters, height_meters, resolution_meters, output_path):
        """Download imagery for a rectangular area centred on the provided coordinates"""
        resolution_meters = max(0.75, resolution_meters)
        zoom_level, actual_resolution = self.calculate_zoom_from_resolution(resolution_meters, center_lat)
        width_pixels = int(width_meters / actual_resolution)
        height_pixels = int(height_meters / actual_resolution)
        width_pixels += width_pixels % 2
        height_pixels += height_pixels % 2
        print(f"Area dimensions: {width_meters}m x {height_meters}m")
        print(f"Image dimensions: {width_pixels}px x {height_pixels}px")

        meters_per_degree_lat = 111320
        meters_per_degree_lon = 111320 * math.cos(math.radians(center_lat))
        lat_offset = (height_meters / 2) / meters_per_degree_lat
        lon_offset = (width_meters / 2) / meters_per_degree_lon
        min_lat = center_lat - lat_offset
        max_lat = center_lat + lat_offset
        min_lon = center_lon - lon_offset
        max_lon = center_lon + lon_offset

        min_tile_x, max_tile_y = self.lat_lon_to_tile(min_lat, min_lon, zoom_level)
        max_tile_x, min_tile_y = self.lat_lon_to_tile(max_lat, max_lon, zoom_level)
        tiles_x = max_tile_x - min_tile_x + 1
        tiles_y = max_tile_y - min_tile_y + 1
        print(f"Downloading {tiles_x}x{tiles_y} tiles at zoom level {zoom_level}")

        total_width = tiles_x * self.tile_size
        total_height = tiles_y * self.tile_size
        combined_image = Image.new('RGB', (total_width, total_height))

        for y in tqdm(range(min_tile_y, max_tile_y + 1), desc="Downloading rows"):
            for x in range(min_tile_x, max_tile_x + 1):
                tile = self.download_tile(x, y, zoom_level)
                pos_x = (x - min_tile_x) * self.tile_size
                pos_y = (y - min_tile_y) * self.tile_size
                combined_image.paste(tile, (pos_x, pos_y))
                time.sleep(0.05)

        world_size = 2 ** zoom_level * self.tile_size
        center_x_pixel = (center_lon + 180) / 360 * world_size
        center_y_pixel = (1 - math.log(math.tan(math.radians(center_lat)) + 1 / math.cos(math.radians(center_lat))) / math.pi) / 2 * world_size
        offset_x = center_x_pixel - (min_tile_x * self.tile_size)
        offset_y = center_y_pixel - (min_tile_y * self.tile_size)
        left = int(offset_x - width_pixels / 2)
        top = int(offset_y - height_pixels / 2)
        right = int(offset_x + width_pixels / 2)
        bottom = int(offset_y + height_pixels / 2)
        left = max(0, left)
        top = max(0, top)
        right = min(total_width, right)
        bottom = min(total_height, bottom)
        cropped_image = combined_image.crop((left, top, right, bottom))

        array = np.array(cropped_image)
        utm_zone = int((center_lon + 180) / 6) + 1
        hemisphere = 'north' if center_lat >= 0 else 'south'
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        center_x_utm, center_y_utm = transformer.transform(center_lon, center_lat)
        left_utm = center_x_utm - (width_meters / 2)
        top_utm = center_y_utm + (height_meters / 2)
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
            for i in range(3):
                dst.write(array[:, :, i], i + 1)

        print(f"Saved GeoTIFF to {output_path}")
        print(f"Final image size: {array.shape}")
        print(f"Final output resolution: {pixel_width:.2f} meters/pixel")
        print(f"UTM Zone: {utm_zone} {hemisphere}")
        return array, pixel_width

    def download_from_reference_tif(self, reference_tif_path, output_path, resolution_meters=None):
        """Download imagery based on a reference GeoTIFF extent"""
        with rasterio.open(reference_tif_path) as src:
            bounds = src.bounds
            is_utm = False
            if src.crs:
                try:
                    is_utm = '+proj=utm' in src.crs.to_proj4()
                except Exception:
                    is_utm = False

            if is_utm:
                print("Reference GeoTIFF is already in UTM projection")
                center_x_utm = (bounds.left + bounds.right) / 2
                center_y_utm = (bounds.bottom + bounds.top) / 2
                width_meters = bounds.right - bounds.left
                height_meters = bounds.top - bounds.bottom
                transformer = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                center_lon, center_lat = transformer.transform(center_x_utm, center_y_utm)
            else:
                print("Reference GeoTIFF is in a non-UTM projection, assuming WGS84")
                center_lon = (bounds.left + bounds.right) / 2
                center_lat = (bounds.bottom + bounds.top) / 2
                meters_per_degree_lat = 111320
                meters_per_degree_lon = 111320 * math.cos(math.radians(center_lat))
                width_meters = (bounds.right - bounds.left) * meters_per_degree_lon
                height_meters = (bounds.top - bounds.bottom) * meters_per_degree_lat

            if resolution_meters is None:
                if is_utm:
                    resolution_meters = max(
                        (bounds.right - bounds.left) / src.width,
                        (bounds.top - bounds.bottom) / src.height
                    )
                else:
                    meters_per_degree_lat = 111320
                    meters_per_degree_lon = 111320 * math.cos(math.radians(center_lat))
                    resolution_meters = max(
                        (bounds.right - bounds.left) / src.width * meters_per_degree_lon,
                        (bounds.top - bounds.bottom) / src.height * meters_per_degree_lat
                    )

            resolution_meters = max(0.75, resolution_meters)
            print(f"Reference GeoTIFF bounds: {bounds}")
            print(f"Center coordinates: {center_lat}, {center_lon}")
            print(f"Area dimensions: {width_meters:.2f}m x {height_meters:.2f}m")
            print(f"Target resolution: {resolution_meters} meters/pixel")

            return self.download_area(
                center_lat, center_lon,
                width_meters, height_meters,
                resolution_meters, output_path
            )


def main():
    parser = argparse.ArgumentParser(description='Download Voyager basemap imagery')
    parser.add_argument('--center_lat', type=float, help='Center latitude')
    parser.add_argument('--center_lon', type=float, help='Center longitude')
    parser.add_argument('--width', type=float, help='Width in meters')
    parser.add_argument('--height', type=float, help='Height in meters')
    parser.add_argument('--resolution', type=float, default=0.75, help='Resolution in meters per pixel (default: 0.75)')
    parser.add_argument('--output', type=str, default='voyager_ref.tif', help='Output GeoTIFF path')
    parser.add_argument('--input_tif_reference', type=str, help='Reference GeoTIFF for area and resolution')

    args = parser.parse_args()

    downloader = VoyagerMapDownloader()

    if args.input_tif_reference:
        downloader.download_from_reference_tif(
            args.input_tif_reference,
            args.output,
            args.resolution
        )
    elif args.center_lat and args.center_lon and args.width and args.height:
        downloader.download_area(
            args.center_lat, args.center_lon,
            args.width, args.height,
            args.resolution, args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()