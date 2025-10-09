#!/usr/bin/env python3
import os
import sys
import struct
import array
import cv2
import multiprocessing
from argparse import ArgumentParser
from tqdm import tqdm

modes = {'sd': (240, 136), 'hd': (320, 180)}

def get_metadata(filepath, args):
    metadata = {}
    if os.path.isfile(filepath):
        if args.hwaccel == 'cuda':
            vcap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
        elif args.hwaccel == 'videotoolbox':
            vcap = cv2.VideoCapture(filepath, cv2.CAP_AVFOUNDATION)
        else:
            vcap = cv2.VideoCapture(filepath)

        if vcap.isOpened():
            metadata['width'] = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['aspect'] = float(metadata['width'] / metadata['height'])
            fps = vcap.get(cv2.CAP_PROP_FPS)
            frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['duration_ms'] = int(frame_count / fps) * 1000
            metadata['duration'] = int(frame_count / fps)
            vcap.release()
            return (True, metadata)
    return (False, metadata)

def process_frame(task):
    """
    Processes a single frame: opens the video, seeks, reads, resizes, and encodes.
    This function is designed to be self-contained for robust multiprocessing.
    """
    # Unpack all task parameters
    filepath, hwaccel, timestamp, target_size, preset, use_opencl = task

    # Map preset to interpolation algorithm and JPEG quality
    if preset == 'fast':
        interpolation = cv2.INTER_LINEAR
        jpeg_quality = 80
    elif preset == 'quality':
        interpolation = cv2.INTER_LANCZOS4
        jpeg_quality = 95
    else:  # medium (default)
        interpolation = cv2.INTER_AREA
        jpeg_quality = 90

    vcap = None
    try:
        # Create VideoCapture object inside the worker function for stability
        if hwaccel in ('videotoolbox', 'auto') and sys.platform == 'darwin':
            vcap = cv2.VideoCapture(filepath, cv2.CAP_AVFOUNDATION)
        elif hwaccel == 'cuda':
            vcap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
        else:
            vcap = cv2.VideoCapture(filepath)

        if not vcap.isOpened():
            print(f"Warning: Could not open video file in worker for timestamp {timestamp}s", file=sys.stderr)
            return None

        vcap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame_bgr = vcap.read()

        if not success:
            # Don't print warning if it's just the end of the video
            if timestamp < (vcap.get(cv2.CAP_PROP_FRAME_COUNT) / vcap.get(cv2.CAP_PROP_FPS)) - 1:
                 print(f"Warning: Could not read frame at {timestamp}s", file=sys.stderr)
            return None

        # Decide whether to use GPU (UMat) or CPU path for resizing
        if use_opencl:
            try:
                frame_umat = cv2.UMat(frame_bgr)
                resized_umat = cv2.resize(frame_umat, target_size, interpolation=interpolation)
                resized_frame = resized_umat.get()
            except cv2.error:
                # Fallback to CPU for this frame in case of a specific GPU error
                resized_frame = cv2.resize(frame_bgr, target_size, interpolation=interpolation)
        else:
            # CPU Path (original behavior)
            resized_frame = cv2.resize(frame_bgr, target_size, interpolation=interpolation)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        _success, encoded_image = cv2.imencode('.jpg', resized_frame, encode_params)
        return (timestamp, encoded_image.tobytes())

    finally:
        if vcap is not None:
            vcap.release()


def extract_images(metadata, args):
    frame_timestamps = range(args.offset, metadata['duration'], args.interval)

    # Check for OpenCL support once in the main process
    use_opencl = False
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.useOpenCL():
            use_opencl = True
            if not args.silent:
                print("OpenCL (GPU) acceleration for resizing is enabled.")

    # Each task now contains all the information needed to be self-contained
    tasks = [(args.filepath, args.hwaccel, ts, modes[args.mode], args.preset, use_opencl) for ts in frame_timestamps]

    images = []
    # The pool no longer needs an initializer, making it simpler and more robust.
    with multiprocessing.Pool(processes=args.jobs) as pool:
        # Calculate chunksize to balance parallel processing with sequential file access
        num_processes = pool._processes or 1
        chunksize, extra = divmod(len(tasks), num_processes * 4)
        if extra:
            chunksize += 1
        
        with tqdm(total=len(tasks), desc="Processing frames", unit="frame", disable=args.silent) as pbar:
            for result in pool.imap_unordered(process_frame, tasks, chunksize=chunksize):
                if result is not None:
                    images.append(result)
                pbar.update()

    images.sort(key=lambda x: x[0])
    return [img_data for _, img_data in images]


def assemble_bif(output_location, images, args):
    magic_number = [0x89, 0x42, 0x49, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]
    version = 0

    with open(output_location, 'wb') as f:
        array.array('B', magic_number).tofile(f)
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<I', len(images)))
        f.write(struct.pack('<I', 1000 * args.interval))
        array.array('B', [0x00] * 44).tofile(f)

        bif_table_size = 8 * (len(images) + 1)
        image_index_offset = 64 + bif_table_size

        image_offsets = []
        current_offset = image_index_offset
        for img_data in images:
            image_offsets.append(current_offset)
            current_offset += len(img_data)

        for i, offset in enumerate(image_offsets):
            f.write(struct.pack('<I', i))
            f.write(struct.pack('<I', offset))

        f.write(struct.pack('<I', 0xffffffff))
        f.write(struct.pack('<I', current_offset))

        for img_data in images:
            f.write(img_data)

    if not args.silent:
        print(f"Successfully created BIF file: {output_location}")

def main():
    parser = ArgumentParser(description='Generate BIF files for Roku trick-play thumbnails.')
    parser.add_argument('filepath', metavar='sourcevid', type=str, help='Video file to process')
    parser.add_argument('-i', '--interval', metavar='N', dest='interval', type=int, default=10,
                        help='Interval between images in seconds (default: 10)')
    parser.add_argument('-O', '--offset', metavar='N', dest='offset', type=int, default=0,
                        help='Offset to first image in seconds (default: 0)')
    parser.add_argument('-o', '--out', metavar='FILE', dest='output', type=str,
                        help='Destination path/file for the BIF result')
    parser.add_argument('--sd', dest='mode', action='store_const', const='sd', default='hd',
                        help='Generate SD resolution BIF file (default: HD)')
    parser.add_argument('-s', '--silent', dest='silent', action='store_true',
                        help='Suppress progress and diagnostic information')
    parser.add_argument('--hwaccel', type=str, default='auto', choices=['auto', 'cuda', 'videotoolbox', 'none'],
                        help='Hardware acceleration to use (default: auto)')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help=f'Number of parallel jobs to run (default: {os.cpu_count()})')
    parser.add_argument('--preset', type=str, default='medium', choices=['fast', 'medium', 'quality'],
                        help='Adjust for speed vs quality. (default: medium)')
    args = parser.parse_args()

    success, metadata = get_metadata(args.filepath, args)
    if not success:
        print('Error: Invalid or corrupt video file', file=sys.stderr)
        sys.exit(1)

    if not args.silent:
        print(f"Source: {os.path.basename(args.filepath)} ({metadata['width']}x{metadata['height']}, duration: {metadata['duration']}s)")

    width, height = modes[args.mode]
    if metadata['aspect'] > 1:
        width = int(height * metadata['aspect'])
    else:
        width = int(height * metadata['aspect'])
    modes[args.mode] = (width, height)

    images = extract_images(metadata, args)
    
    if not images:
        print("Error: No images were extracted. BIF file generation failed.", file=sys.stderr)
        sys.exit(1)

    destination = args.output if args.output is not None else \
        f'{os.path.splitext(os.path.basename(args.filepath))[0]}-{args.mode.upper()}.bif'
    
    assemble_bif(destination, images, args)

if __name__ == '__main__':
    main()