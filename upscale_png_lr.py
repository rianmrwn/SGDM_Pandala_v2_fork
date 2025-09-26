import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import cv2

class BatchUpscaleProcessor:
    """
    Batch processor for upscaling PNG images
    """
    
    def __init__(self, input_folder, output_folder=None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder) if output_folder else self.input_folder / "processed"
        self.results = []
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def get_file_list(self, pattern="*.png"):
        """
        Get list of PNG files in the input folder
        """
        file_patterns = [pattern, "*.PNG"]
        files = []
        
        for pattern in file_patterns:
            files.extend(list(self.input_folder.glob(pattern)))
            
        return sorted(files)
    
    def get_image_info(self, file_path):
        """
        Get basic information about a PNG file
        """
        try:
            with Image.open(str(file_path)) as img:
                width, height = img.size
                mode = img.mode
                
                file_info = {
                    'filename': file_path.name,
                    'filepath': str(file_path),
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                }
                
                return file_info, None
            
        except Exception as e:
            return None, f"Error processing {file_path}: {str(e)}"
    
    def batch_upscale(self, target_size=(800, 800), interpolation='bicubic'):
        """
        Batch upscale images
        """
        files = self.get_file_list()
        print(f"Found {len(files)} files to process in {self.input_folder}")
        
        upscale_results = []
        
        # Map interpolation method string to cv2 interpolation
        interpolation_methods = {
            'bicubic': cv2.INTER_CUBIC,
            'bilinear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'lanczos': cv2.INTER_LANCZOS4
        }
        interp_method = interpolation_methods.get(interpolation, cv2.INTER_CUBIC)
        
        for i, file_path in enumerate(files, 1):
            print(f"\nProcessing {i}/{len(files)}: {file_path.name}")
            
            try:
                # Get original file info
                original_info, error = self.get_image_info(file_path)
                if error:
                    print(f"  Error: {error}")
                    continue
                
                # Parse tile coordinates from filename if it follows the pattern "tile_X_Y.png"
                filename = file_path.name
                tile_coords = None
                
                if filename.startswith("tile_") and filename.endswith(".png"):
                    try:
                        # Extract coordinates from filename (tile_X_Y.png)
                        parts = filename.replace(".png", "").split("_")
                        if len(parts) >= 3:
                            tile_coords = (int(parts[1]), int(parts[2]))
                    except (ValueError, IndexError):
                        pass
                
                # Define output path using tile coordinates if available
                if tile_coords:
                    output_filename = f"tile_{tile_coords[0]}_{tile_coords[1]}.png"
                else:
                    # If the filename doesn't match the tile pattern, use a sequential index
                    output_filename = f"tile_{i-1}_0.png"
                
                output_path = self.output_folder / output_filename
                
                # Read image using OpenCV
                img = cv2.imread(str(file_path))
                if img is None:
                    raise ValueError(f"Could not read image: {file_path}")
                
                # Convert BGR to RGB if image is color
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Perform upscaling
                upscaled = cv2.resize(img, target_size, interpolation=interp_method)
                
                # Convert back to BGR if image is color
                if len(upscaled.shape) == 3:
                    upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
                
                # Save upscaled image
                cv2.imwrite(str(output_path), upscaled)
                
                # Get upscaled file info
                upscaled_info, error = self.get_image_info(output_path)
                
                if upscaled_info:
                    result_entry = {
                        'original_file': file_path.name,
                        'upscaled_file': output_path.name,
                        'original_size': f"{original_info['width']}x{original_info['height']}",
                        'upscaled_size': f"{upscaled_info['width']}x{upscaled_info['height']}",
                        'scale_factor_x': upscaled_info['width'] / original_info['width'],
                        'scale_factor_y': upscaled_info['height'] / original_info['height'],
                        'processing_status': 'Success'
                    }
                    
                    upscale_results.append(result_entry)
                    
                    print(f"  ✓ Original size: {original_info['width']}x{original_info['height']}")
                    print(f"  ✓ New size: {upscaled_info['width']}x{upscaled_info['height']}")
                    print(f"  ✓ Scale factor: {result_entry['scale_factor_x']:.2f}x")
                    print(f"  ✓ Saved to: {output_path.name}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {file_path.name}: {str(e)}")
                upscale_results.append({
                    'original_file': file_path.name,
                    'processing_status': f'Error: {str(e)}'
                })
        
        return upscale_results
    
    def generate_summary_report(self, upscale_results):
        """
        Generate comprehensive summary report
        """
        if not upscale_results:
            print("No results to analyze")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(upscale_results)
        
        print("\n" + "="*80)
        print("BATCH UPSCALING ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nFOLDER: {self.input_folder}")
        print(f"TOTAL FILES PROCESSED: {len(upscale_results)}")
        print(f"ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Count successful operations
        successful = len(df[df['processing_status'] == 'Success'])
        print(f"\nPROCESSING RESULTS:")
        print(f"  Successfully processed: {successful}/{len(upscale_results)} files")
        
        if successful > 0:
            success_df = df[df['processing_status'] == 'Success']
            
            # Extract numeric scale factors
            scale_factors = success_df['scale_factor_x'].mean()
            
            print(f"  Average scale factor: {scale_factors:.2f}x")
        
        # Save results to CSV
        csv_path = self.output_folder / f"upscaling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")

def process_batch(input_folder, output_folder=None, target_size=(800, 800), interpolation='bicubic'):
    """
    Main function to process a batch of images
    """
    processor = BatchUpscaleProcessor(input_folder, output_folder)
    
    print("Starting batch processing of images...")
    print(f"Input folder: {input_folder}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Interpolation method: {interpolation}")
    
    # Perform batch upscaling
    print("\n--- BATCH UPSCALING ---")
    upscale_results = processor.batch_upscale(target_size, interpolation)
    
    # Generate report
    print("\n--- GENERATING REPORT ---")
    processor.generate_summary_report(upscale_results)
    
    return processor, upscale_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch upscale PNG images')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing PNG images')
    parser.add_argument('--output_folder', type=str, help='Output folder for upscaled images')
    parser.add_argument('--target_width', type=int, default=256, help='Target width for upscaled images')
    parser.add_argument('--target_height', type=int, default=256, help='Target height for upscaled images')
    parser.add_argument('--interpolation', type=str, default='bicubic', 
                        choices=['bicubic', 'bilinear', 'nearest', 'lanczos'],
                        help='Interpolation method for upscaling')
    
    args = parser.parse_args()
    
    # Process the batch
    processor, results = process_batch(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        target_size=(args.target_width, args.target_height),
        interpolation=args.interpolation
    )
    
    print("\nBatch processing completed!")