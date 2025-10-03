#!/usr/bin/env python3
import os
import sys
import struct
from argparse import ArgumentParser

def preview_bif(bif_path):
    if not os.path.isfile(bif_path):
        print(f"Error: File not found: {bif_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.splitext(os.path.basename(bif_path))[0] + "_preview"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(bif_path, 'rb') as f:
        magic = f.read(8)
        if magic != b'\x89BIF\r\n\x1a\n':
            print("Error: Not a valid BIF file.", file=sys.stderr)
            sys.exit(1)

        version = struct.unpack('<I', f.read(4))[0]
        num_images = struct.unpack('<I', f.read(4))[0]
        interval_ms = struct.unpack('<I', f.read(4))[0]
        f.read(44)  # Skip reserved area

        print(f"BIF Version: {version}")
        print(f"Number of images: {num_images}")
        print(f"Interval (ms): {interval_ms}")

        image_offsets = []
        for _ in range(num_images):
            timestamp_index = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<I', f.read(4))[0]
            image_offsets.append(offset)

        # Read the end-of-table marker
        f.read(8)

        for i, offset in enumerate(image_offsets):
            f.seek(offset)
            # The next offset determines the size of the current image
            next_offset = image_offsets[i + 1] if i + 1 < len(image_offsets) else os.fstat(f.fileno()).st_size
            image_size = next_offset - offset
            image_data = f.read(image_size)
            
            output_filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            with open(output_filename, 'wb') as img_f:
                img_f.write(image_data)

    print(f"Successfully extracted {num_images} frames to: {output_dir}")

def main():
    parser = ArgumentParser(description='Extract frames from a Roku BIF file for preview.')
    parser.add_argument('bif_file', metavar='BIF_FILE', type=str, help='Path to the BIF file to preview')
    args = parser.parse_args()
    preview_bif(args.bif_file)

if __name__ == '__main__':
    main()
