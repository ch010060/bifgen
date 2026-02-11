#!/usr/bin/env python3
import os
import sys
import struct
import array
import cv2
import multiprocessing
import shutil
import subprocess
import tempfile
import glob
import json
from argparse import ArgumentParser
from tqdm import tqdm

modes = {'sd': (240, 136), 'hd': (320, 180)}

def get_metadata(filepath, args):
    metadata = {}
    if os.path.isfile(filepath):
        # Try OpenCV first (preserves original behavior)
        if args.hwaccel == 'cuda':
            vcap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
        elif args.hwaccel == 'videotoolbox':
            vcap = cv2.VideoCapture(filepath, cv2.CAP_AVFOUNDATION)
        else:
            vcap = cv2.VideoCapture(filepath)

        if vcap.isOpened():
            metadata['width'] = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['aspect'] = float(metadata['width'] / metadata['height']) if metadata['height'] else 0.0
            fps = float(vcap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
            metadata['duration_ms'] = int(duration * 1000.0)
            metadata['duration'] = int(duration)
            vcap.release()
            return (True, metadata)

        # Fallback: probe via ffprobe if available to validate the file and get metadata
        ok, md = _ffprobe_metadata(filepath)
        if ok:
            return (True, md)
    return (False, metadata)

# Worker initializer to create a global VideoCapture object per worker process
vcap = None
def init_worker(filepath, hwaccel):
    global vcap
    if hwaccel == 'cuda':
        vcap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
    elif hwaccel == 'videotoolbox':
        vcap = cv2.VideoCapture(filepath, cv2.CAP_AVFOUNDATION)
    else:
        vcap = cv2.VideoCapture(filepath)

def process_frame(task):
    global vcap
    timestamp, target_size, preset = task

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

    vcap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame_bgr = vcap.read()

    if not success:
        print(f"Warning: Could not read frame at {timestamp}s", file=sys.stderr)
        return None

    resized_frame = cv2.resize(frame_bgr, target_size, interpolation=interpolation)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    _success, encoded_image = cv2.imencode('.jpg', resized_frame, encode_params)
    return (timestamp, encoded_image.tobytes())

def extract_images(metadata, args):
    # Quick pre-check: if OpenCV can't open with the selected backend, use FFmpeg fallback
    pre_vcap = None
    try:
        if args.hwaccel == 'cuda':
            pre_vcap = cv2.VideoCapture(args.filepath, cv2.CAP_FFMPEG)
        elif args.hwaccel == 'videotoolbox':
            pre_vcap = cv2.VideoCapture(args.filepath, cv2.CAP_AVFOUNDATION)
        else:
            pre_vcap = cv2.VideoCapture(args.filepath)
        if not pre_vcap.isOpened():
            if not args.silent:
                print('Primary backend failed to open. Falling back to FFmpeg extraction...', file=sys.stderr)
            return extract_images_ffmpeg(metadata, args)
    finally:
        if pre_vcap is not None:
            pre_vcap.release()

    frame_timestamps = range(args.offset, metadata['duration'], args.interval)
    tasks = [(ts, modes[args.mode], args.preset) for ts in frame_timestamps]

    images = []
    # initializer and initargs are used to create a per-process VideoCapture object
    # to avoid opening the file for every frame.
    with multiprocessing.Pool(processes=args.jobs, initializer=init_worker, initargs=(args.filepath, args.hwaccel)) as pool:
        # Calculate chunksize to balance parallel processing with sequential file access
        num_processes = pool._processes or 1  # Fallback for unusual pool configurations
        chunksize, extra = divmod(len(tasks), num_processes * 4)
        if extra:
            chunksize += 1
        
        with tqdm(total=len(tasks), desc="Processing frames", unit="frame", disable=args.silent) as pbar:
            for result in pool.imap_unordered(process_frame, tasks, chunksize=chunksize):
                if result is not None:
                    images.append(result)
                pbar.update()

    if not images:
        # If OpenCV path produced no images, fallback to FFmpeg extraction
        if not args.silent:
            print('No frames via OpenCV; falling back to FFmpeg extraction...', file=sys.stderr)
        return extract_images_ffmpeg(metadata, args)

    images.sort(key=lambda x: x[0])
    return [img_data for _, img_data in images]

def _ffprobe_metadata(filepath):
    if not shutil.which('ffprobe'):
        return (False, {})
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate',
            '-show_entries', 'format=duration',
            '-of', 'json',
            filepath
        ]
        out = subprocess.check_output(cmd)
        info = json.loads(out.decode('utf-8', errors='ignore'))
        streams = info.get('streams', [])
        fmt = info.get('format', {})
        if not streams:
            return (False, {})
        s = streams[0]
        width = int(s.get('width') or 0)
        height = int(s.get('height') or 0)
        fr_str = s.get('avg_frame_rate') or '0/1'
        try:
            num, den = fr_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except Exception:
            fps = 0.0
        try:
            duration = float(fmt.get('duration') or 0.0)
        except Exception:
            duration = 0.0
        if width <= 0 or height <= 0 or duration <= 0:
            return (False, {})
        metadata = {
            'width': width,
            'height': height,
            'aspect': float(width / height) if height else 0.0,
            'duration_ms': int(duration * 1000.0),
            'duration': int(duration),
            'fps': fps,
        }
        return (True, metadata)
    except Exception:
        return (False, {})

def _ffmpeg_qscale(preset: str):
    # Lower is better quality; keep filesize reasonable
    if preset == 'fast':
        return 6
    if preset == 'quality':
        return 2
    return 4

def extract_images_ffmpeg(metadata, args):
    if not shutil.which('ffmpeg'):
        if not args.silent:
            print('FFmpeg not found in PATH; cannot fallback to FFmpeg extraction.', file=sys.stderr)
        return []

    width, height = modes[args.mode]
    interval = max(1, int(args.interval))
    qscale = _ffmpeg_qscale(args.preset)

    images = []
    with tempfile.TemporaryDirectory(prefix='bif_frames_') as tmpdir:
        out_pattern = os.path.join(tmpdir, '%08d.jpg')
        cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'error', '-threads', '0']
        if args.offset:
            cmd += ['-ss', str(args.offset)]
        cmd += ['-i', args.filepath,
                '-map', '0:v:0', '-an', '-sn', '-dn',
                '-vf', f'fps=1/{interval}', '-s', f'{width}x{height}',
                '-qscale:v', str(qscale),
                # Request machine-readable progress and write sequential JPEGs
                '-progress', 'pipe:1',
                '-y', out_pattern]

        # Estimate expected images for a smoother progress bar
        expected = max(0, int(max(0, metadata.get('duration', 0) - int(args.offset)) / interval))

        try:
            last_count = 0
            with tqdm(total=expected, desc='Extracting', unit='img', disable=args.silent) as pbar:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
                if proc.stdout is not None:
                    for _ in proc.stdout:
                        try:
                            count = len(glob.glob(os.path.join(tmpdir, '*.jpg')))
                            if count > expected:
                                count = expected
                            if count > last_count:
                                pbar.update(count - last_count)
                                last_count = count
                        except Exception:
                            pass
                ret = proc.wait()
                # Final sync to actual files produced
                try:
                    final_count = len(glob.glob(os.path.join(tmpdir, '*.jpg')))
                    if final_count > expected:
                        final_count = expected
                    if final_count > last_count:
                        pbar.update(final_count - last_count)
                        last_count = final_count
                except Exception:
                    pass
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
        except subprocess.CalledProcessError:
            if not args.silent:
                print('FFmpeg extraction failed.', file=sys.stderr)
            return []

        # Collect frames in order
        for fname in sorted(glob.glob(os.path.join(tmpdir, '*.jpg'))):
            with open(fname, 'rb') as fp:
                images.append(fp.read())

    return images


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
