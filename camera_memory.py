import time
import argparse
import numpy as np
import pyrealsense2 as rs
import cv2
from multiprocessing import shared_memory, Lock, Process, Manager
import multiprocessing
import struct
import sys
import signal

# Shared memory names (choose unique names if you run multiple cameras)
COLOR_SHM_NAME = "realsense_color_shm_v1"
DEPTH_SHM_NAME = "realsense_depth_shm_v1"
META_NAME = "realsense_meta"  # Manager Namespace, not raw shm

def create_camera_pipeline(width=640, height=480, fps=60):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)
    # Optionally align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  #0.0010000000474974513
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])

    return pipeline, align

def create_shared_buffers(width, height, channels=3):
    color_nbytes = width * height * channels  # uint8
    depth_nbytes = width * height * 2         # uint16

    # Remove existing shared memory blocks if present
    # (attempt attach then unlink)
    for name in (COLOR_SHM_NAME, DEPTH_SHM_NAME):
        try:
            existing = shared_memory.SharedMemory(name=name)
            existing.close()
            existing.unlink()
            print(f"Removed existing shared memory {name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print("Warning removing shm:", e)

    color_shm = shared_memory.SharedMemory(create=True, size=color_nbytes, name=COLOR_SHM_NAME)
    depth_shm = shared_memory.SharedMemory(create=True, size=depth_nbytes, name=DEPTH_SHM_NAME)

    # Return SharedMemory objects
    return color_shm, depth_shm

def cleanup_shm(shm_list):
    """Clean up the shared memory segments."""
    for shm in shm_list:
        try:
            print(f"Cleaning up shared memory {shm.name}")
            shm.close()   # Close the shared memory
            shm.unlink()  # Unlink the shared memory (delete it from system)
            print(f"Shared memory {shm.name} cleaned up.")
        except Exception as e:
            print(f"Error cleaning up shared memory {shm.name}: {e}")

def producer_main(width=640, height=480, fps=30):
    manager = Manager()
    meta = manager.Namespace()
    lock = Lock()
    # breakpoint()
    # metadata fields: width,height,channels,color_dtype,depth_dtype,frame_id,timestamp
    meta.width = width
    meta.height = height
    meta.channels = 3
    meta.color_dtype = 'uint8'
    meta.depth_dtype = 'uint16'
    meta.frame_id = 0
    meta.timestamp = 0.0
    # breakpoint()
    color_shm, depth_shm = create_shared_buffers(width, height, channels=3)
    # breakpoint()
    # Map numpy arrays to the shared buffers
    color_buf = np.ndarray((height, width, 3), dtype=np.uint8, buffer=color_shm.buf)
    depth_buf = np.ndarray((height, width), dtype=np.uint16, buffer=depth_shm.buf)
    # breakpoint()
    pipeline, align = create_camera_pipeline(width, height, fps)
    print("Producer: RealSense pipeline started.")
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        while running:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data()) 
            depth = np.asanyarray(depth_frame.get_data())

            # Write into shared memory with lock
            lock.acquire()
            try:
                # copy into buffers (fast C-level copy)
                np.copyto(color_buf, color)
                np.copyto(depth_buf, depth)
                meta.frame_id += 1
                meta.timestamp = time.time()
            finally:
                lock.release()

            # Throttle a little if needed
            # print status occasionally
            if meta.frame_id % 30 == 0:
                print(f"Producer wrote frame {meta.frame_id} @ {meta.timestamp}")

    finally:
        pipeline.stop()
        cleanup_shm([color_shm, depth_shm])
        manager.shutdown()
        print("Producer exiting and cleaned shared memory.")

if __name__=="__main__":
    try:
        producer_main()
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # This block ensures cleanup happens in case of any unhandled exception
        print("Program finished. Attempting final cleanup.")
        # cleanup_shm([color_shm, depth_shm])