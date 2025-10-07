#!/usr/bin/env python3
import os
import sys
import struct
import cv2
import numpy as np
import random
from argparse import ArgumentParser

def calculate_mse(image_a, image_b):
    """Calculates the Mean Squared Error between two images."""
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    return err

def validate_bif(bif_path, video_path, num_samples, mse_threshold, validate_all=False):
    """
    Validates a BIF file against its source video by comparing a sample of frames.
    """
    if not os.path.isfile(bif_path):
        print(f"Error: BIF file not found: {bif_path}", file=sys.stderr)
        return False

    if not os.path.isfile(video_path):
        print(f"Error: Source video not found: {video_path}", file=sys.stderr)
        return False

    try:
        with open(bif_path, 'rb') as f:
            magic = f.read(8)
            if magic != b'\x89BIF\r\n\x1a\n':
                print("Error: Not a valid BIF file.", file=sys.stderr)
                return False

            version = struct.unpack('<I', f.read(4))[0]
            num_images = struct.unpack('<I', f.read(4))[0]
            interval_ms = struct.unpack('<I', f.read(4))[0]
            f.read(44)  # Skip reserved area

            print(f"BIF Version: {version}, Images: {num_images}, Interval: {interval_ms}ms")

            if num_images == 0:
                print("Warning: BIF file contains no images to validate.")
                return True

            image_offsets = []
            for _ in range(num_images):
                f.read(4) # Skip timestamp index
                offset = struct.unpack('<I', f.read(4))[0]
                image_offsets.append(offset)
            
            # The last offset points to the end of the last image
            end_of_table = f.tell()
            f.seek(end_of_table + 4) # Skip 0xffffffff
            final_offset = struct.unpack('<I', f.read(4))[0]
            image_offsets.append(final_offset)


            # --- Validation ---
            mismatches = 0
            
            if validate_all:
                sample_indices = range(num_images)
                print(f"\nValidating all {num_images} frames...")
            else:
                sample_indices = random.sample(range(num_images), min(num_samples, num_images))
                print(f"\nValidating {len(sample_indices)} random frames...")


            for i, frame_index in enumerate(sample_indices):
                print(f"  - Processing Frame Index: {frame_index}... ", end='')
                
                # 1. Extract image from BIF
                offset = image_offsets[frame_index]
                next_offset = image_offsets[frame_index + 1]
                image_size = next_offset - offset
                f.seek(offset)
                bif_jpg_data = f.read(image_size)
                bif_image = cv2.imdecode(np.frombuffer(bif_jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                if bif_image is None:
                    print("FAILED (could not decode image from BIF)")
                    mismatches += 1
                    continue

                # 2. Extract frame from source video
                timestamp_ms = frame_index * interval_ms
                vcap = cv2.VideoCapture(video_path)
                vcap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
                success, video_frame = vcap.read()
                vcap.release()

                if not success:
                    print(f"FAILED (could not read frame at {timestamp_ms}ms from video)")
                    mismatches += 1
                    continue

                # 3. Normalize and Compare
                resized_video_frame = cv2.resize(video_frame, (bif_image.shape[1], bif_image.shape[0]), interpolation=cv2.INTER_AREA)
                mse = calculate_mse(bif_image, resized_video_frame)

                if mse < mse_threshold:
                    print(f"OK (MSE: {mse:.2f})")
                else:
                    print(f"MISMATCH (MSE: {mse:.2f})")
                    mismatches += 1

    except Exception as e:
        print(f"An error occurred during validation: {e}", file=sys.stderr)
        return False

    print("\n--- Validation Summary ---")
    if mismatches == 0:
        print(f"Success: All {len(sample_indices)} processed frames matched the source video.")
        return True
    else:
        print(f"Failure: {mismatches}/{len(sample_indices)} processed frames did not match.")
        return False


def main():
    parser = ArgumentParser(description='Validate a Roku BIF file against its source video.')
    parser.add_argument('bif_file', metavar='BIF_FILE', type=str, help='Path to the BIF file to validate')
    parser.add_argument('source_video', metavar='SOURCE_VIDEO', type=str, help='Path to the original source video file')
    parser.add_argument('-n', '--samples', type=int, default=5, help='Number of random frames to sample for validation (default: 5)')
    parser.add_argument('--all', action='store_true', help='Validate all frames in the BIF file (overrides --samples)')
    parser.add_argument('--mse-threshold', type=float, default=400.0, help='MSE threshold for a mismatch (default: 400.0)')
    args = parser.parse_args()

    if not validate_bif(args.bif_file, args.source_video, args.samples, args.mse_threshold, args.all):
        sys.exit(1)

if __name__ == '__main__':
    main()
