# Parse named command-line options and their default values.
import argparse
# Run capture saving and unlock submission in a background task without freezing the camera.
from concurrent.futures import ThreadPoolExecutor
# Generate an unpredictable temporary password for the local dashboard.
import secrets
# Exchange commands between threads using synchronized queues.
import queue
# Manage enrollment image directories with filesystem copy, move, and cleanup helpers.
import shutil
# Run background workers and protect state shared between threads.
import threading
# Read monotonic clocks and pace loops without depending on wall-clock changes.
import time
# Build filesystem paths relative to this script rather than the current working directory.
from pathlib import Path

# Read process environment settings and operating-system information.
import os
# Import the project-local database URI loader used before parsing command-line options.
from mongodb_config import load_mongodb_env

# Load the project MongoDB URI before command-line defaults are constructed.
load_mongodb_env()

# Reuse the shared UART sender that serializes command and servo-state writes.
from uart_commands import UARTCommands
# Import the controller that converts image-center error into pan and tilt commands.
from servo_tracking import ServoTracker
# Import the save-before-unlock workflow and its explicit database-save failure exception.
from unlock_action import CaptureSaveError, UnlockAction

# Create the command-line option parser for this program.
parser = argparse.ArgumentParser(description="Face recognition with UART approval output")
# Accept command-line option '--uart-port'; use '/dev/ttyAMA0' when it is omitted.
parser.add_argument("--uart-port", default="/dev/ttyAMA0", help="Pi UART device or USB serial device")
# Accept command-line option '--uart-baud'; use 115200 when it is omitted.
parser.add_argument("--uart-baud", type=int, default=115200, help="Must match the Pico baud rate")
# Accept command-line option '--no-uart'; use the parser action default when it is omitted.
parser.add_argument("--no-uart", action="store_true", help="Run recognition without sending commands")
# Accept command-line option '--mongo-uri'; use os.environ.get('MONGODB_URI') or 'mongodb://localhost:27017' when it is omitted.
parser.add_argument("--mongo-uri", default=os.environ.get("MONGODB_URI") or "mongodb://localhost:27017")
# Accept command-line option '--mongo-db'; use 'face_access' when it is omitted.
parser.add_argument("--mongo-db", default="face_access")
# Accept command-line option '--mongo-collection'; use 'unlock_events' when it is omitted.
parser.add_argument("--mongo-collection", default="unlock_events")
# Accept command-line option '--web-host'; use '0.0.0.0' when it is omitted.
parser.add_argument("--web-host", default="0.0.0.0", help="Local dashboard bind address")
# Accept command-line option '--web-port'; use 5000 when it is omitted.
parser.add_argument("--web-port", type=int, default=5000)
# Accept command-line option '--no-web'; use the parser action default when it is omitted.
parser.add_argument("--no-web", action="store_true", help="Disable the web dashboard")
# Accept command-line option '--headless'; use the parser action default when it is omitted.
parser.add_argument("--headless", action="store_true", help="Run without the desktop camera window")
# Accept command-line option '--pan-direction'; use -1 when it is omitted.
parser.add_argument("--pan-direction", type=int, choices=(-1, 1), default=-1)
# Accept command-line option '--tilt-direction'; use 1 when it is omitted.
parser.add_argument("--tilt-direction", type=int, choices=(-1, 1), default=1)
# Read command-line options and apply their defaults.
args = parser.parse_args()
# Continue with this branch only when not args.no_web.
if not args.no_web:
    # Attempt this operation and handle the expected failures below.
    try:
        # Import the thread-safe dashboard state and HTTP server only when web output is enabled.
        from web_server import Monitor, WebServer
    # Handle ImportError without losing required cleanup.
    except ImportError as error:
        # Report RuntimeError('Dashboard requires Flask: .venv/bin/python -m pip install flask').
        raise RuntimeError("Dashboard requires Flask: .venv/bin/python -m pip install flask") from error
# Set the CPU count available to this process for vision inference using len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1.
vision_cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)

# Access OpenCV camera capture, image drawing, face alignment, and JPEG encoding.
import cv2
# Use NumPy arrays and vector operations for boxes and face embeddings.
import numpy as np
# Configure the PyTorch CPU inference runtime used by the YOLO model.
import torch
# Load the Ultralytics YOLO interface for the project face-detector weights.
from ultralytics import YOLO

# Limit OpenCV worker threads to the CPUs available to this process.
cv2.setNumThreads(vision_cpu_count)


# Set the directory containing this script, independent of the working directory using Path(__file__).resolve().parent.
PROJECT_DIR = Path(__file__).resolve().parent
# Set the project-specific YOLO face detector weights path using PROJECT_DIR / 'model.pt'.
YOLO_MODEL_PATH = PROJECT_DIR / "model.pt"
# Set the directory holding the ONNX face models using PROJECT_DIR / 'models'.
MODEL_DIR = PROJECT_DIR / "models"
# Set the YuNet landmark detector model path using MODEL_DIR / 'face_detection_yunet_2023mar.onnx'.
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
# Set the SFace recognition model path using MODEL_DIR / 'face_recognition_sface_2021dec.onnx'.
SFACE_MODEL_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
# Set the directory containing the current owner enrollment photos using PROJECT_DIR / 'owner_faces'.
OWNER_DIR = PROJECT_DIR / "owner_faces"
# Set the temporary directory used until a new enrollment completes using PROJECT_DIR / '.owner_faces_pending'.
PENDING_DIR = PROJECT_DIR / ".owner_faces_pending"

# Set the number of owner photographs collected during enrollment using 60.
ENROLLMENT_SAMPLES = 60
# OpenCV's published SFace cosine threshold. Tune against real owner and
# non-owner samples before using this for any physical access control.
# Set the minimum SFace cosine similarity accepted as an owner match using 0.363.
COSINE_THRESHOLD = 0.363
# Set the minimum YOLO detection confidence using 0.5.
YOLO_CONFIDENCE = 0.5


# Stop with a clear error if a required model file is missing.
def require_file(path, description):
    # Continue with this branch only when not path.is_file().
    if not path.is_file():
        # Report FileNotFoundError(f'Missing {description}: {path}\nSee README.md for the model download instructions.').
        raise FileNotFoundError(
            # Continue the enclosing expression with this value or condition.
            f"Missing {description}: {path}\n"
            # Continue the enclosing expression with this value or condition.
            "See README.md for the model download instructions."
        # Finish the preceding expression, container, or function call.
        )


# Continue with this branch only when not hasattr(cv2, 'FaceDetectorYN_create') or not hasattr(cv2, 'FaceRecognizerSF_create').
if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
    # Report RuntimeError('This script needs OpenCV Python 4.10+ with FaceDetectorYN and FaceRecognizerSF. See README.md; no package was installed aut....
    raise RuntimeError(
        # Continue the enclosing expression with this value or condition.
        "This script needs OpenCV Python 4.10+ with FaceDetectorYN and "
        # Continue the enclosing expression with this value or condition.
        "FaceRecognizerSF. See README.md; no package was installed automatically."
    # Finish the preceding expression, container, or function call.
    )

# Call require_file with the arguments below.
require_file(YOLO_MODEL_PATH, "the original YOLO face detector model")
# Call require_file with the arguments below.
require_file(YUNET_MODEL_PATH, "the YuNet alignment model")
# Call require_file with the arguments below.
require_file(SFACE_MODEL_PATH, "the SFace recognition model")

# Keep YOLO from starving the camera display on the Pi 5.
# Set the CPU inference thread count used by PyTorch.
torch.set_num_threads(vision_cpu_count)
# Limit coordination between PyTorch operations to one thread.
torch.set_num_interop_threads(1)
# Load the project YOLO weights for CPU face detection.
model = YOLO(str(YOLO_MODEL_PATH))

# YuNet is used only to find landmarks inside YOLO's face crop so SFace can
# align it. YOLO remains the full-frame face detector.
# Create the YuNet landmark detector used to align each YOLO face crop.
landmark_detector = cv2.FaceDetectorYN_create(
    # Continue building landmark_detector from these values.
    str(YUNET_MODEL_PATH), "", (320, 320), 0.7, 0.3, 5000
# Finish the preceding expression, container, or function call.
)
# Create the SFace feature extractor used for owner recognition.
face_recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL_PATH), "")


# Align a face with YuNet landmarks and produce a normalized SFace feature vector.
def get_embedding(face_crop):
    # Set (height, width) from face_crop.shape[:2].
    height, width = face_crop.shape[:2]
    # Continue with this branch only when width < 20 or height < 20.
    if width < 20 or height < 20:
        # Return None to the caller.
        return None

    # Call landmark_detector.setInputSize with the arguments below.
    landmark_detector.setInputSize((width, height))
    # Set (_, faces) from landmark_detector.detect(face_crop).
    _, faces = landmark_detector.detect(face_crop)
    # Continue with this branch only when faces is None or len(faces) == 0.
    if faces is None or len(faces) == 0:
        # Return None to the caller.
        return None

    # Select the largest face in the YOLO crop, then align and embed it.
    # Set face from max(faces, key=lambda row: float(row[2] * row[3])).
    face = max(faces, key=lambda row: float(row[2] * row[3]))
    # Set aligned from face_recognizer.alignCrop(face_crop, face).
    aligned = face_recognizer.alignCrop(face_crop, face)
    # Set feature from face_recognizer.feature(aligned).reshape(-1).astype(np.float32).
    feature = face_recognizer.feature(aligned).reshape(-1).astype(np.float32)
    # Set norm from np.linalg.norm(feature).
    norm = np.linalg.norm(feature)
    # Continue with this branch only when norm == 0.
    if norm == 0:
        # Return None to the caller.
        return None
    # Return feature / norm to the caller.
    return feature / norm


# Rebuild the owner feature database from saved enrollment photographs.
def load_owner_embeddings():
    # Set embeddings from [].
    embeddings = []
    # Continue with this branch only when OWNER_DIR.is_dir().
    if OWNER_DIR.is_dir():
        # Process each image_path from sorted(OWNER_DIR.glob('*.jpg')).
        for image_path in sorted(OWNER_DIR.glob("*.jpg")):
            # Set image from cv2.imread(str(image_path)).
            image = cv2.imread(str(image_path))
            # Continue with this branch only when image is None.
            if image is None:
                # Skip the rest of this iteration and wait for the next input or frame.
                continue
            # Set the normalized SFace feature vector for the current face using get_embedding(image).
            embedding = get_embedding(image)
            # Continue with this branch only when embedding is not None.
            if embedding is not None:
                # Call embeddings.append with the arguments below.
                embeddings.append(embedding)
    # Return embeddings to the caller.
    return embeddings


# Set the normalized feature vectors representing the enrolled owner using load_owner_embeddings().
owner_embeddings = load_owner_embeddings()
# Set whether at least one valid owner feature vector is available using len(owner_embeddings) > 0.
registered = len(owner_embeddings) > 0

# Open the known-working USB camera. The capture thread continuously drains
# the camera; inference consumes only the newest frame and never queues lag.
# Open USB camera index 0 using the Linux V4L2 backend.
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# Continue with this branch only when not cap.isOpened().
if not cap.isOpened():
    # Report RuntimeError('Cannot open camera 0').
    raise RuntimeError("Cannot open camera 0")
# Request a USB-camera capture property; the driver may negotiate the actual value.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# Request a USB-camera capture property; the driver may negotiate the actual value.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# Request a USB-camera capture property; the driver may negotiate the actual value.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Request a USB-camera capture property; the driver may negotiate the actual value.
cap.set(cv2.CAP_PROP_FPS, 30)
# Request a USB-camera capture property; the driver may negotiate the actual value.
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Set the lock protecting camera and recognition state shared by worker threads using threading.Lock().
state_lock = threading.Lock()
# Set the newest camera frame shared with the detection and display loops using None.
latest_frame = None
# Set the camera frame sequence number using 0.
frame_number = 0
# Set the latest smoothed face boxes used for display and servo tracking using [].
latest_boxes = []
# Set the confidence values associated with the latest detections using [].
latest_confidences = []
# Set the original source image corresponding to the latest YOLO result using None.
latest_detection_frame = None
# Set the monotonic source timestamp used to reject stale detections using 0.0.
latest_detection_time = 0.0
# Set the unsmoothed YOLO boxes used for accurate recognition crops using [].
latest_recognition_boxes = []
# Set the frame identifier of the latest published detection using -1.
detection_number = -1
# Set the detected face index currently recognized as the owner using None.
authorized_box_index = None
# Set the unannotated source image eligible for an approved capture using None.
authorized_capture = None
# Set the timestamp used to enforce capture freshness using 0.0.
authorized_capture_time = 0.0
# Set the recognition label shared with the desktop and web dashboard using 'UNKNOWN' if registered else 'NOT REGISTERED'.
display_status = "UNKNOWN" if registered else "NOT REGISTERED"
# Set the number of enrollment images collected so far using 0.
enrollment_count = 0
# Set whether owner enrollment is currently running using False.
enrollment_active = False
# Set the fatal worker error reported during cleanup using None.
fatal_error = None
# Set the shared shutdown signal for background workers using threading.Event().
stop_event = threading.Event()
# Set the thread-safe queue carrying register and cancel commands using queue.Queue().
recognition_commands = queue.Queue()

# Set the old-box weight used to smooth the displayed bounding boxes using 0.25.
BOX_SMOOTHING = 0.25
# Set the minimum overlap used to match detections to previous display boxes using 0.3.
BOX_MATCH_IOU = 0.3


# Measure overlap between two boxes as intersection divided by union.
def box_iou(a, b):
    # Set left from max(a[0], b[0]).
    left = max(a[0], b[0])
    # Set top from max(a[1], b[1]).
    top = max(a[1], b[1])
    # Set right from min(a[2], b[2]).
    right = min(a[2], b[2])
    # Set bottom from min(a[3], b[3]).
    bottom = min(a[3], b[3])
    # Set intersection from max(0, right - left) * max(0, bottom - top).
    intersection = max(0, right - left) * max(0, bottom - top)
    # Set area_a from max(0, a[2] - a[0]) * max(0, a[3] - a[1]).
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    # Set area_b from max(0, b[2] - b[0]) * max(0, b[3] - b[1]).
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    # Return intersection / (area_a + area_b - intersection + 1e-06) to the caller.
    return intersection / (area_a + area_b - intersection + 1e-6)


# Continuously drain the USB camera and share only the newest frame.
def capture_frames():
    # Use shared module variables here: latest_frame, frame_number, fatal_error.
    global latest_frame, frame_number, fatal_error
    # Repeat this loop while not stop_event.is_set().
    while not stop_event.is_set():
        # Set (ok, frame) from cap.read().
        ok, frame = cap.read()
        # Continue with this branch only when not ok.
        if not ok:
            # Set the fatal worker error reported during cleanup using 'Cannot read camera frame'.
            fatal_error = "Cannot read camera frame"
            # Ask every background worker to finish.
            stop_event.set()
            # Leave this function without a result.
            return
        # Enter state_lock and release or restore it automatically on exit.
        with state_lock:
            # Set the newest camera frame shared with the detection and display loops using frame.
            latest_frame = frame
            # Update the camera frame sequence number using 1.
            frame_number += 1


# Run YOLO on fresh frames and publish smoothed display boxes and raw recognition boxes.
def detect_faces():
    # Use shared module variables here: latest_boxes, latest_confidences, latest_detection_frame.
    global latest_boxes, latest_confidences, latest_detection_frame
    # Use shared module variables here: latest_recognition_boxes, detection_number, latest_detection_time, fatal_error.
    global latest_recognition_boxes, detection_number, latest_detection_time, fatal_error
    # Set the most recent camera frame consumed by YOLO using -1.
    last_processed = -1
    # Set the previously smoothed boxes used to stabilize the display using [].
    tracked_boxes = []

    # Repeat this loop while not stop_event.is_set().
    while not stop_event.is_set():
        # Enter state_lock and release or restore it automatically on exit.
        with state_lock:
            # Continue with this branch only when latest_frame is None or frame_number == last_processed.
            if latest_frame is None or frame_number == last_processed:
                # Set image from None.
                image = None
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Set image from latest_frame.copy().
                image = latest_frame.copy()
                # Set current_number from frame_number.
                current_number = frame_number
                # Set the monotonic timestamp of the image submitted for inference using time.monotonic().
                source_time = time.monotonic()

        # Continue with this branch only when image is None.
        if image is None:
            # Yield briefly while waiting for new work instead of busy-spinning.
            time.sleep(0.002)
            # Skip the rest of this iteration and wait for the next input or frame.
            continue

        # Attempt this operation and handle the expected failures below.
        try:
            # Set result from model.predict(source=image, imgsz=320, conf=YOLO_CONFIDENCE, device='cpu', verbose=False)[0].
            result = model.predict(
                # Supply source as a named argument to the enclosing call.
                source=image,
                # Supply imgsz as a named argument to the enclosing call.
                imgsz=320,
                # Supply conf as a named argument to the enclosing call.
                conf=YOLO_CONFIDENCE,
                # Supply device as a named argument to the enclosing call.
                device="cpu",
                # Supply verbose as a named argument to the enclosing call.
                verbose=False,
            # Continue building result from these values.
            )[0]

            # Set the original YOLO bounding boxes before display smoothing using [].
            raw_boxes = []
            # Set the original YOLO confidence scores using [].
            raw_confidences = []
            # Continue with this branch only when result.boxes is not None and len(result.boxes) > 0.
            if result.boxes is not None and len(result.boxes) > 0:
                # Set the original YOLO bounding boxes before display smoothing using result.boxes.xyxy.cpu().numpy().astype(float).tolist().
                raw_boxes = result.boxes.xyxy.cpu().numpy().astype(float).tolist()
                # Set the original YOLO confidence scores using result.boxes.conf.cpu().numpy().astype(float).tolist().
                raw_confidences = result.boxes.conf.cpu().numpy().astype(float).tolist()

            # Smooth only the display boxes. Recognition receives the raw
            # YOLO boxes and matching source frame for an accurate face crop.
            # Set the previous-box indexes still available for overlap matching using set(range(len(tracked_boxes))).
            unmatched = set(range(len(tracked_boxes)))
            # Set the newly matched and smoothed display boxes using [].
            new_tracked = []
            # Process each (box, confidence) from zip(raw_boxes, raw_confidences).
            for box, confidence in zip(raw_boxes, raw_confidences):
                # Set box_array from np.asarray(box, dtype=float).
                box_array = np.asarray(box, dtype=float)
                # Set the previous box selected as the best overlap match using None.
                best_index = None
                # Set the current highest acceptable intersection-over-union score using BOX_MATCH_IOU.
                best_overlap = BOX_MATCH_IOU
                # Process each index from unmatched.
                for index in unmatched:
                    # Set overlap from box_iou(box_array, tracked_boxes[index]).
                    overlap = box_iou(box_array, tracked_boxes[index])
                    # Continue with this branch only when overlap > best_overlap.
                    if overlap > best_overlap:
                        # Set the previous box selected as the best overlap match using index.
                        best_index = index
                        # Set the current highest acceptable intersection-over-union score using overlap.
                        best_overlap = overlap
                # Continue with this branch only when best_index is not None.
                if best_index is not None:
                    # Set old_box from tracked_boxes[best_index].
                    old_box = tracked_boxes[best_index]
                    # Set box_array from BOX_SMOOTHING * old_box + (1 - BOX_SMOOTHING) * box_array.
                    box_array = BOX_SMOOTHING * old_box + (1 - BOX_SMOOTHING) * box_array
                    # Call unmatched.remove with the arguments below.
                    unmatched.remove(best_index)
                # Call new_tracked.append with the arguments below.
                new_tracked.append(box_array)
            # Set the previously smoothed boxes used to stabilize the display using new_tracked.
            tracked_boxes = new_tracked

            # Enter state_lock and release or restore it automatically on exit.
            with state_lock:
                # Set the latest smoothed face boxes used for display and servo tracking using [box.copy() for box in tracked_boxes].
                latest_boxes = [box.copy() for box in tracked_boxes]
                # Set the confidence values associated with the latest detections using raw_confidences.
                latest_confidences = raw_confidences
                # Set the original source image corresponding to the latest YOLO result using image.
                latest_detection_frame = image
                # Set the unsmoothed YOLO boxes used for accurate recognition crops using [np.asarray(box, dtype=float) for box in raw_boxes].
                latest_recognition_boxes = [np.asarray(box, dtype=float) for box in raw_boxes]
                # Set the frame identifier of the latest published detection using current_number.
                detection_number = current_number
                # Set the monotonic source timestamp used to reject stale detections using source_time.
                latest_detection_time = source_time
            # Set the most recent camera frame consumed by YOLO using current_number.
            last_processed = current_number
        # Handle Exception without losing required cleanup.
        except Exception as error:
            # Set the fatal worker error reported during cleanup using f'YOLO error: {error}'.
            fatal_error = f"YOLO error: {error}"
            # Ask every background worker to finish.
            stop_event.set()
            # Leave this function without a result.
            return


# Extract a face crop with a 20 percent border, clipped to the image boundaries.
def crop_face(image, box):
    # Set (x1, y1, x2, y2) from np.rint(box).astype(int).
    x1, y1, x2, y2 = np.rint(box).astype(int)
    # Set width from max(1, x2 - x1).
    width = max(1, x2 - x1)
    # Set height from max(1, y2 - y1).
    height = max(1, y2 - y1)
    # Set margin_x from int(width * 0.2).
    margin_x = int(width * 0.2)
    # Set margin_y from int(height * 0.2).
    margin_y = int(height * 0.2)
    # Set x1 from max(0, x1 - margin_x).
    x1 = max(0, x1 - margin_x)
    # Set y1 from max(0, y1 - margin_y).
    y1 = max(0, y1 - margin_y)
    # Set x2 from min(image.shape[1], x2 + margin_x).
    x2 = min(image.shape[1], x2 + margin_x)
    # Set y2 from min(image.shape[0], y2 + margin_y).
    y2 = min(image.shape[0], y2 + margin_y)
    # Return image[y1:y2, x1:x2].copy() to the caller.
    return image[y1:y2, x1:x2].copy()


# Publish recognition, enrollment, and authorized-capture state under the shared lock.
def set_ui_state(status, is_registering=False, count=0, box_index=None, capture=None, capture_time=0.0):
    # Use shared module variables here: display_status, enrollment_active, enrollment_count, authorized_box_index.
    global display_status, enrollment_active, enrollment_count, authorized_box_index
    # Use shared module variables here: authorized_capture, authorized_capture_time.
    global authorized_capture, authorized_capture_time
    # Enter state_lock and release or restore it automatically on exit.
    with state_lock:
        # Set the unannotated source image eligible for an approved capture using capture.
        authorized_capture = capture
        # Set the timestamp used to enforce capture freshness using capture_time.
        authorized_capture_time = capture_time
        # Set the recognition label shared with the desktop and web dashboard using status.
        display_status = status
        # Set whether owner enrollment is currently running using is_registering.
        enrollment_active = is_registering
        # Set the number of enrollment images collected so far using count.
        enrollment_count = count
        # Set the detected face index currently recognized as the owner using box_index.
        authorized_box_index = box_index


# Process enrollment commands and compare each single detected face with the owner.
def recognize_faces():
    # Use shared module variables here: owner_embeddings, registered, fatal_error.
    global owner_embeddings, registered, fatal_error
    # Set the last detection consumed by the recognition worker using -1.
    last_seen_detection = -1
    # Set whether this recognition worker is collecting enrollment samples using False.
    collecting = False
    # Set the enrollment photographs gathered for the pending owner using [].
    collected_images = []
    # Set the face feature vectors gathered during pending enrollment using [].
    collected_embeddings = []
    # Set the number of consecutive accepted owner matches using 0.
    recognized_streak = 0

    # Repeat this loop while not stop_event.is_set().
    while not stop_event.is_set():
        # Attempt this operation and handle the expected failures below.
        try:
            # Set command from recognition_commands.get(timeout=0.01).
            command = recognition_commands.get(timeout=0.01)
        # Handle queue.Empty without losing required cleanup.
        except queue.Empty:
            # Set command from None.
            command = None

        # Continue with this branch only when command == 'register'.
        if command == "register":
            # Set whether this recognition worker is collecting enrollment samples using True.
            collecting = True
            # Set the enrollment photographs gathered for the pending owner using [].
            collected_images = []
            # Set the face feature vectors gathered during pending enrollment using [].
            collected_embeddings = []
            # Continue with this branch only when PENDING_DIR.exists().
            if PENDING_DIR.exists():
                # Call shutil.rmtree with the arguments below.
                shutil.rmtree(PENDING_DIR)
            # Call PENDING_DIR.mkdir with the arguments below.
            PENDING_DIR.mkdir(parents=True, exist_ok=True)
            # Set the number of consecutive accepted owner matches using 0.
            recognized_streak = 0
            # Call set_ui_state with the arguments below.
            set_ui_state("REGISTERING 0/60", True, 0)
            # Print a diagnostic, status, or setup message to the console.
            print("Registration started. Keep one face in view; slowly turn left/right and vary the lighting.")
        # Continue with this branch only when command == 'cancel' and collecting.
        elif command == "cancel" and collecting:
            # Set whether this recognition worker is collecting enrollment samples using False.
            collecting = False
            # Call shutil.rmtree with the arguments below.
            shutil.rmtree(PENDING_DIR, ignore_errors=True)
            # Call set_ui_state with the arguments below.
            set_ui_state("UNKNOWN" if registered else "NOT REGISTERED")
            # Print a diagnostic, status, or setup message to the console.
            print("Registration cancelled; the previous owner registration was kept.")

        # Enter state_lock and release or restore it automatically on exit.
        with state_lock:
            # Continue with this branch only when detection_number == last_seen_detection or latest_detection_frame is None.
            if detection_number == last_seen_detection or latest_detection_frame is None:
                # Set detection_data from None.
                detection_data = None
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Set detection_data from (detection_number, latest_detection_time, latest_detection_frame.copy(), [box.copy() for box in latest_recognition_boxes]).
                detection_data = (
                    # Continue building detection_data from these values.
                    detection_number,
                    # Continue building detection_data from these values.
                    latest_detection_time,
                    # Continue building detection_data from these values.
                    latest_detection_frame.copy(),
                    # Continue building detection_data from these values.
                    [box.copy() for box in latest_recognition_boxes],
                # Finish the preceding expression, container, or function call.
                )

        # Continue with this branch only when detection_data is None.
        if detection_data is None:
            # Skip the rest of this iteration and wait for the next input or frame.
            continue

        # Set (current_detection, capture_time, image, boxes) from detection_data.
        current_detection, capture_time, image, boxes = detection_data
        # Set the last detection consumed by the recognition worker using current_detection.
        last_seen_detection = current_detection
        # Continue with this branch only when len(boxes) != 1.
        if len(boxes) != 1:
            # Set the number of consecutive accepted owner matches using 0.
            recognized_streak = 0
            # Continue with this branch only when collecting.
            if collecting:
                # Call set_ui_state with the arguments below.
                set_ui_state(
                    # Provide the next argument or expression for set_ui_state.
                    f"REGISTERING {len(collected_images)}/{ENROLLMENT_SAMPLES} - ONE FACE ONLY",
                    # Provide the next argument or expression for set_ui_state.
                    True,
                    # Provide the next argument or expression for set_ui_state.
                    len(collected_images),
                # Finish the preceding expression, container, or function call.
                )
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Call set_ui_state with the arguments below.
                set_ui_state("NOT REGISTERED" if not registered else "UNKNOWN")
            # Skip the rest of this iteration and wait for the next input or frame.
            continue

        # Set the cropped face image used for enrollment or recognition using crop_face(image, boxes[0]).
        face_image = crop_face(image, boxes[0])
        # Set the normalized SFace feature vector for the current face using get_embedding(face_image).
        embedding = get_embedding(face_image)
        # Continue with this branch only when embedding is None.
        if embedding is None:
            # Set the number of consecutive accepted owner matches using 0.
            recognized_streak = 0
            # Call set_ui_state with the arguments below.
            set_ui_state("UNKNOWN - FACE ALIGNMENT FAILED")
            # Skip the rest of this iteration and wait for the next input or frame.
            continue

        # Continue with this branch only when collecting.
        if collecting:
            # Set sample_index from len(collected_images).
            sample_index = len(collected_images)
            # Set sample_path from PENDING_DIR / f'owner_{sample_index:03d}.jpg'.
            sample_path = PENDING_DIR / f"owner_{sample_index:03d}.jpg"
            # Continue with this branch only when cv2.imwrite(str(sample_path), face_image).
            if cv2.imwrite(str(sample_path), face_image):
                # Call collected_images.append with the arguments below.
                collected_images.append(face_image)
                # Call collected_embeddings.append with the arguments below.
                collected_embeddings.append(embedding)

            # Set count from len(collected_images).
            count = len(collected_images)
            # Continue with this branch only when count in (20, 40).
            if count in (20, 40):
                # Print a diagnostic, status, or setup message to the console.
                print(f"Collected {count}/{ENROLLMENT_SAMPLES}. Change angle or lighting, then continue.")
            # Continue with this branch only when count >= ENROLLMENT_SAMPLES.
            if count >= ENROLLMENT_SAMPLES:
                # Continue with this branch only when OWNER_DIR.exists().
                if OWNER_DIR.exists():
                    # Call shutil.rmtree with the arguments below.
                    shutil.rmtree(OWNER_DIR)
                # Call PENDING_DIR.replace with the arguments below.
                PENDING_DIR.replace(OWNER_DIR)
                # Set the normalized feature vectors representing the enrolled owner using collected_embeddings.copy().
                owner_embeddings = collected_embeddings.copy()
                # Set whether at least one valid owner feature vector is available using True.
                registered = True
                # Set whether this recognition worker is collecting enrollment samples using False.
                collecting = False
                # Set the number of consecutive accepted owner matches using 0.
                recognized_streak = 0
                # Call set_ui_state with the arguments below.
                set_ui_state("UNKNOWN", False, 0)
                # Print a diagnostic, status, or setup message to the console.
                print(f"Registration complete: {len(owner_embeddings)} face samples saved in {OWNER_DIR}")
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Call set_ui_state with the arguments below.
                set_ui_state(
                    # Provide the next argument or expression for set_ui_state.
                    f"REGISTERING {count}/{ENROLLMENT_SAMPLES}",
                    # Provide the next argument or expression for set_ui_state.
                    True,
                    # Provide the next argument or expression for set_ui_state.
                    count,
                # Finish the preceding expression, container, or function call.
                )
            # Skip the rest of this iteration and wait for the next input or frame.
            continue

        # Continue with this branch only when not registered or not owner_embeddings.
        if not registered or not owner_embeddings:
            # Call set_ui_state with the arguments below.
            set_ui_state("NOT REGISTERED")
            # Skip the rest of this iteration and wait for the next input or frame.
            continue

        # Set the highest cosine similarity between this face and the stored owner vectors using max((float(np.dot(embedding, owner_feature)) for owner_feature in owner_embeddings)).
        score = max(float(np.dot(embedding, owner_feature)) for owner_feature in owner_embeddings)
        # Continue with this branch only when score >= COSINE_THRESHOLD.
        if score >= COSINE_THRESHOLD:
            # Update the number of consecutive accepted owner matches using 1.
            recognized_streak += 1
            # Continue with this branch only when recognized_streak >= 2.
            if recognized_streak >= 2:
                # Call set_ui_state with the arguments below.
                set_ui_state("AUTHORIZED", False, 0, 0, image, capture_time)
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Call set_ui_state with the arguments below.
                set_ui_state("UNKNOWN")
        # Handle the remaining case when the preceding conditions did not match.
        else:
            # Set the number of consecutive accepted owner matches using 0.
            recognized_streak = 0
            # Call set_ui_state with the arguments below.
            set_ui_state("UNKNOWN")


# Prepare a background worker without starting it yet.
capture_thread = threading.Thread(target=capture_frames, daemon=True)
# Prepare a background worker without starting it yet.
detection_thread = threading.Thread(target=detect_faces, daemon=True)
# Prepare a background worker without starting it yet.
recognition_thread = threading.Thread(target=recognize_faces, daemon=True)
# Set uart from None.
uart = None
# Set unlock_action from None.
unlock_action = None
# Prepare one background worker for database capture and unlock actions.
unlock_executor = ThreadPoolExecutor(max_workers=1)
# Set the pending background capture-and-unlock operation using None.
unlock_future = None
# Set the last capture or unlock result shown to the user using ''.
unlock_message = ""
# Set the time of the last accepted unlock action for the cooldown check using -10.0.
last_unlock_time = -10.0
# Set whether a desktop or web interaction has requested unlocking using False.
unlock_requested = False
# Set the desktop unlock button hit-test rectangle using (0, 0, 0, 0).
button_rect = (0, 0, 0, 0)
# Set the optional HTTP server sharing the same camera process using None.
web_server = None
# Set the thread-safe bridge between the camera loop and dashboard routes using None.
monitor = None
# Set the timestamp used to limit dashboard JPEG publishing to about 10 FPS using 0.0.
last_web_frame_time = 0.0
# Set the image-error controller that computes horizontal and vertical servo angles using ServoTracker(args.pan_direction, args.tilt_direction).
servo_tracker = ServoTracker(args.pan_direction, args.tilt_direction)
# Set the timestamp used to limit UART state updates to about 20 Hz using 0.0.
last_servo_send = 0.0


# Queue an unlock request when the desktop unlock button is clicked.
def mouse_click(event, x, y, flags, param):
    # Use shared module variables here: unlock_requested.
    global unlock_requested
    # Set (left, top, right, bottom) from button_rect.
    left, top, right, bottom = button_rect
    # Continue with this branch only when event == cv2.EVENT_LBUTTONUP and left <= x <= right and (top <= y <= bottom).
    if event == cv2.EVENT_LBUTTONUP and left <= x <= right and top <= y <= bottom:
        # Set whether a desktop or web interaction has requested unlocking using True.
        unlock_requested = True

# Attempt this operation and handle the expected failures below.
try:
    # Continue with this branch only when not args.no_uart.
    if not args.no_uart:
        # Set uart from UARTCommands(args.uart_port, args.uart_baud).
        uart = UARTCommands(args.uart_port, args.uart_baud)
        # Print a diagnostic, status, or setup message to the console.
        print(f"UART ready: {args.uart_port} at {args.uart_baud} baud")
    # Attempt this operation and handle the expected failures below.
    try:
        # Set unlock_action from UnlockAction(args.mongo_uri, args.mongo_db, args.mongo_collection, uart).
        unlock_action = UnlockAction(args.mongo_uri, args.mongo_db, args.mongo_collection, uart)
    # Handle CaptureSaveError without losing required cleanup.
    except CaptureSaveError as error:
        # Report SystemExit(str(error)).
        raise SystemExit(str(error)) from None
    # Continue with this branch only when not args.no_web.
    if not args.no_web:
        # Set the thread-safe bridge between the camera loop and dashboard routes using Monitor().
        monitor = Monitor()
        # Set password from os.environ.get('FACE_WEB_PASSWORD') or secrets.token_urlsafe(12).
        password = os.environ.get("FACE_WEB_PASSWORD") or secrets.token_urlsafe(12)
        # Set the optional HTTP server sharing the same camera process using WebServer(monitor, unlock_action, password, args.web_host, args.web_port).
        web_server = WebServer(monitor, unlock_action, password, args.web_host, args.web_port)
        # Start web_server in the background.
        web_server.start()
        # Print a diagnostic, status, or setup message to the console.
        print(f"Dashboard: http://<Pi-IP>:{args.web_port} (same Wi-Fi)", flush=True)
        # Print a diagnostic, status, or setup message to the console.
        print("Dashboard username: admin", flush=True)
        # Continue with this branch only when not os.environ.get('FACE_WEB_PASSWORD').
        if not os.environ.get("FACE_WEB_PASSWORD"):
            # Print a diagnostic, status, or setup message to the console.
            print(f"Temporary dashboard password: {password}", flush=True)
        # Handle the remaining case when the preceding conditions did not match.
        else:
            # Print a diagnostic, status, or setup message to the console.
            print("Dashboard password: from FACE_WEB_PASSWORD", flush=True)
    # Continue with this branch only when not args.headless.
    if not args.headless:
        # Create the desktop camera window.
        cv2.namedWindow("YOLO Face Recognition")
        # Attach the unlock-button click handler to the desktop window.
        cv2.setMouseCallback("YOLO Face Recognition", mouse_click)
    # Start capture_thread in the background.
    capture_thread.start()
    # Start detection_thread in the background.
    detection_thread.start()
    # Start recognition_thread in the background.
    recognition_thread.start()
    # Set last_displayed_frame from -1.
    last_displayed_frame = -1
    # Repeat this loop while not stop_event.is_set().
    while not stop_event.is_set():
        # Enter state_lock and release or restore it automatically on exit.
        with state_lock:
            # Set image from None if latest_frame is None else latest_frame.copy().
            image = None if latest_frame is None else latest_frame.copy()
            # Set current_number from frame_number.
            current_number = frame_number
            # Set boxes from [box.copy() for box in latest_boxes].
            boxes = [box.copy() for box in latest_boxes]
            # Set confidences from list(latest_confidences).
            confidences = list(latest_confidences)
            # Set status from display_status.
            status = display_status
            # Set registering from enrollment_active.
            registering = enrollment_active
            # Set count from enrollment_count.
            count = enrollment_count
            # Set green_box from authorized_box_index.
            green_box = authorized_box_index
            # Set a synchronized snapshot of the detection identifier for servo tracking using detection_number.
            servo_detection = detection_number
            # Set a synchronized snapshot of the detection timestamp for freshness checks using latest_detection_time.
            servo_capture_time = latest_detection_time

        # Continue with this branch only when image is None.
        if image is None:
            # Yield briefly while waiting for new work instead of busy-spinning.
            time.sleep(0.005)
            # Skip the rest of this iteration and wait for the next input or frame.
            continue

        # Set the monotonic clock reading used for this tracking update using time.monotonic().
        servo_now = time.monotonic()
        # Set (pan, tilt) from servo_tracker.update(boxes, image.shape[1], image.shape[0], servo_detection, servo_capture_time, servo_now).
        pan, tilt = servo_tracker.update(boxes, image.shape[1], image.shape[0],
                                         # Continue building (pan, tilt) from these values.
                                         servo_detection, servo_capture_time, servo_now)
        # Continue with this branch only when uart is not None and servo_now - last_servo_send >= 0.05.
        if uart is not None and servo_now - last_servo_send >= 0.05:
            # Transmit the current angles and approval flag as one complete STATE line.
            uart.send_state(pan, tilt)
            # Set the timestamp used to limit UART state updates to about 20 Hz using servo_now.
            last_servo_send = servo_now

        # Continue with this branch only when unlock_future is not None and unlock_future.done().
        if unlock_future is not None and unlock_future.done():
            # Attempt this operation and handle the expected failures below.
            try:
                # Set the last capture or unlock result shown to the user using unlock_future.result().
                unlock_message = unlock_future.result()
            # Handle Exception without losing required cleanup.
            except Exception as error:
                # Set the last capture or unlock result shown to the user using str(error) if isinstance(error, CaptureSaveError) else 'UNLOCK FAILED - CHECK TERMINAL'.
                unlock_message = str(error) if isinstance(error, CaptureSaveError) else "UNLOCK FAILED - CHECK TERMINAL"
                # Print a diagnostic, status, or setup message to the console.
                print(f"Unlock action failed: {error}", flush=True)
            # Set the pending background capture-and-unlock operation using None.
            unlock_future = None

        # Continue with this branch only when monitor is not None and monitor.consume_unlock().
        if monitor is not None and monitor.consume_unlock():
            # Set whether a desktop or web interaction has requested unlocking using True.
            unlock_requested = True

        # Enter state_lock and release or restore it automatically on exit.
        with state_lock:
            # Set can_unlock from display_status == 'AUTHORIZED' and (not enrollment_active) and (authorized_capture is not None) and (time.monotonic() - authorized_captur....
            can_unlock = (
                # Supply display_status as a named argument to the enclosing call.
                display_status == "AUTHORIZED" and not enrollment_active
                # Continue building can_unlock from these values.
                and authorized_capture is not None
                # Continue building can_unlock from these values.
                and time.monotonic() - authorized_capture_time < 1.0
                # Continue building can_unlock from these values.
                and len(latest_recognition_boxes) == 1
                # Continue building can_unlock from these values.
                and time.monotonic() - latest_detection_time < 1.0
                # Continue building can_unlock from these values.
                and unlock_future is None
                # Continue building can_unlock from these values.
                and time.monotonic() - last_unlock_time >= 3.0
            # Finish the preceding expression, container, or function call.
            )
            # Set capture from authorized_capture.copy() if can_unlock and unlock_requested else None.
            capture = authorized_capture.copy() if can_unlock and unlock_requested else None
        # Continue with this branch only when unlock_requested.
        if unlock_requested:
            # Set whether a desktop or web interaction has requested unlocking using False.
            unlock_requested = False
            # Continue with this branch only when capture is not None.
            if capture is not None:
                # Encode the image to JPEG and retain the success flag and compressed bytes.
                ok, encoded = cv2.imencode(".jpg", capture)
                # Continue with this branch only when ok.
                if ok:
                    # Set the time of the last accepted unlock action for the cooldown check using time.monotonic().
                    last_unlock_time = time.monotonic()
                    # Set the last capture or unlock result shown to the user using 'SAVING CAPTURE...'.
                    unlock_message = "SAVING CAPTURE..."
                    # Set the pending background capture-and-unlock operation using unlock_executor.submit(unlock_action.execute, encoded.tobytes()).
                    unlock_future = unlock_executor.submit(unlock_action.execute, encoded.tobytes())
                    # Set can_unlock from False.
                    can_unlock = False
                # Handle the remaining case when the preceding conditions did not match.
                else:
                    # Set the last capture or unlock result shown to the user using 'CAPTURE ENCODING FAILED'.
                    unlock_message = "CAPTURE ENCODING FAILED"
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Set the last capture or unlock result shown to the user using 'UNLOCK UNAVAILABLE - WAIT FOR OWNER MATCH / COOLDOWN'.
                unlock_message = "UNLOCK UNAVAILABLE - WAIT FOR OWNER MATCH / COOLDOWN"

        # Continue with this branch only when monitor is not None.
        if monitor is not None:
            # Call monitor.publish with the arguments below.
            monitor.publish({"status": display_status, "can_unlock": can_unlock,
                             # Include the UART operating state in this record.
                             "uart": "OFF (test mode)" if uart is None else "READY",
                             # Include the latest action result in this record.
                             "message": unlock_message})

        # Continue with this branch only when current_number == last_displayed_frame.
        if current_number == last_displayed_frame:
            # Continue with this branch only when args.headless.
            if args.headless:
                # Yield briefly while waiting for new work instead of busy-spinning.
                time.sleep(0.005)
                # Set key from -1.
                key = -1
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Set key from cv2.waitKey(1) & 255.
                key = cv2.waitKey(1) & 0xFF
        # Handle the remaining case when the preceding conditions did not match.
        else:
            # Set last_displayed_frame from current_number.
            last_displayed_frame = current_number
            # Process each (index, box) from enumerate(boxes).
            for index, box in enumerate(boxes):
                # Set (x1, y1, x2, y2) from np.rint(box).astype(int).
                x1, y1, x2, y2 = np.rint(box).astype(int)
                # Set authorized from status == 'AUTHORIZED' and index == green_box.
                authorized = status == "AUTHORIZED" and index == green_box
                # Set color from (0, 200, 0) if authorized else (0, 0, 255).
                color = (0, 200, 0) if authorized else (0, 0, 255)
                # Set label from 'AUTHORIZED' if authorized else 'UNKNOWN' if registered else 'NOT REGISTERED'.
                label = "AUTHORIZED" if authorized else ("UNKNOWN" if registered else "NOT REGISTERED")
                # Draw a face or button rectangle on the displayed image.
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                # Draw an informational label on the displayed image.
                cv2.putText(
                    # Provide the next argument or expression for cv2.putText.
                    image,
                    # Provide the next argument or expression for cv2.putText.
                    label,
                    # Provide the next argument or expression for cv2.putText.
                    (x1, max(22, y1 - 8)),
                    # Provide the next argument or expression for cv2.putText.
                    cv2.FONT_HERSHEY_SIMPLEX,
                    # Provide the next argument or expression for cv2.putText.
                    0.65,
                    # Provide the next argument or expression for cv2.putText.
                    color,
                    # Provide the next argument or expression for cv2.putText.
                    2,
                # Finish the preceding expression, container, or function call.
                )

            # Continue with this branch only when registering.
            if registering:
                # Continue with this branch only when count < 20.
                if count < 20:
                    # Set guidance from 'FACE FORWARD'.
                    guidance = "FACE FORWARD"
                # Continue with this branch only when count < 40.
                elif count < 40:
                    # Set guidance from 'TURN LEFT/RIGHT'.
                    guidance = "TURN LEFT/RIGHT"
                # Handle the remaining case when the preceding conditions did not match.
                else:
                    # Set guidance from 'VARY LIGHTING'.
                    guidance = "VARY LIGHTING"
                # Set status from f'REGISTERING {count}/{ENROLLMENT_SAMPLES}: {guidance} [Q CANCEL]'.
                status = f"REGISTERING {count}/{ENROLLMENT_SAMPLES}: {guidance} [Q CANCEL]"
            # Set status_color from (0, 200, 0) if status == 'AUTHORIZED' else (0, 0, 255).
            status_color = (0, 200, 0) if status == "AUTHORIZED" else (0, 0, 255)
            # Draw an informational label on the displayed image.
            cv2.putText(
                # Provide the next argument or expression for cv2.putText.
                image,
                # Provide the next argument or expression for cv2.putText.
                status,
                # Provide the next argument or expression for cv2.putText.
                (10, 28),
                # Provide the next argument or expression for cv2.putText.
                cv2.FONT_HERSHEY_SIMPLEX,
                # Provide the next argument or expression for cv2.putText.
                0.65,
                # Provide the next argument or expression for cv2.putText.
                status_color,
                # Provide the next argument or expression for cv2.putText.
                2,
            # Finish the preceding expression, container, or function call.
            )
            # Set (height, width) from image.shape[:2].
            height, width = image.shape[:2]
            # Set (cx, cy) from (width // 2, height // 2).
            cx, cy = width // 2, height // 2
            # Draw a cross at the image center as the servo tracking reference.
            cv2.drawMarker(image, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 20, 1)
            # Set the desktop unlock button hit-test rectangle using (width - 180, height - 55, width - 10, height - 15).
            button_rect = (width - 180, height - 55, width - 10, height - 15)
            # Set (left, top, right, bottom) from button_rect.
            left, top, right, bottom = button_rect
            # Draw a face or button rectangle on the displayed image.
            cv2.rectangle(image, (left, top), (right, bottom),
                          # Provide the next argument or expression for cv2.rectangle.
                          (0, 150, 0) if can_unlock else (80, 80, 80), -1)
            # Draw an informational label on the displayed image.
            cv2.putText(image, "UNLOCK [K]", (left + 18, top + 27),
                        # Provide the next argument or expression for cv2.putText.
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            # Draw an informational label on the displayed image.
            cv2.putText(image, unlock_message, (10, height - 68),
                        # Provide the next argument or expression for cv2.putText.
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)
            # Set uart_text from 'UART OFF' if uart is None else 'UART READY'.
            uart_text = "UART OFF" if uart is None else "UART READY"
            # Draw an informational label on the displayed image.
            cv2.putText(image, uart_text, (10, height - 12),
                        # Provide the next argument or expression for cv2.putText.
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            # Continue with this branch only when monitor is not None and time.monotonic() - last_web_frame_time >= 0.1.
            if monitor is not None and time.monotonic() - last_web_frame_time >= 0.1:
                # Encode the image to JPEG and retain the success flag and compressed bytes.
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                # Continue with this branch only when ok.
                if ok:
                    # Call monitor.publish with the arguments below.
                    monitor.publish({"status": display_status, "can_unlock": can_unlock,
                                     # Include the UART operating state in this record.
                                     "uart": "OFF (test mode)" if uart is None else "READY",
                                     # Include the latest action result in this record.
                                     "message": unlock_message}, encoded.tobytes())
                    # Set the timestamp used to limit dashboard JPEG publishing to about 10 FPS using time.monotonic().
                    last_web_frame_time = time.monotonic()
            # Continue with this branch only when args.headless.
            if args.headless:
                # Set key from -1.
                key = -1
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Display the latest annotated image in the desktop window.
                cv2.imshow("YOLO Face Recognition", image)
                # Set key from cv2.waitKey(1) & 255.
                key = cv2.waitKey(1) & 0xFF

        # Continue with this branch only when key in (ord('k'), ord('K')).
        if key in (ord("k"), ord("K")):
            # Set whether a desktop or web interaction has requested unlocking using True.
            unlock_requested = True
        # Continue with this branch only when key == ord('r') and (not registering).
        elif key == ord("r") and not registering:
            # Call recognition_commands.put with the arguments below.
            recognition_commands.put("register")
            # Call set_ui_state with the arguments below.
            set_ui_state("REGISTERING 0/60", True, 0)
        # Continue with this branch only when key == ord('q').
        elif key == ord("q"):
            # Enter state_lock and release or restore it automatically on exit.
            with state_lock:
                # Set registering_now from enrollment_active.
                registering_now = enrollment_active
            # Continue with this branch only when registering_now.
            if registering_now:
                # Call recognition_commands.put with the arguments below.
                recognition_commands.put("cancel")
            # Handle the remaining case when the preceding conditions did not match.
            else:
                # Leave this loop and continue with shutdown or the following operation.
                break
# Handle KeyboardInterrupt without losing required cleanup.
except KeyboardInterrupt:
    # Intentionally perform no additional action in this branch.
    pass
# Always release resources here, including after an exception or interruption.
finally:
    # Ask every background worker to finish.
    stop_event.set()
    # Continue with this branch only when web_server is not None.
    if web_server is not None:
        # Call web_server.close with the arguments below.
        web_server.close()
    # Call unlock_executor.shutdown with the arguments below.
    unlock_executor.shutdown(wait=True)
    # Continue with this branch only when unlock_action is not None.
    if unlock_action is not None:
        # Call unlock_action.close with the arguments below.
        unlock_action.close()
    # Continue with this branch only when uart is not None.
    if uart is not None:
        # Call uart.close with the arguments below.
        uart.close()
    # Process each worker from (capture_thread, detection_thread, recognition_thread).
    for worker in (capture_thread, detection_thread, recognition_thread):
        # Continue with this branch only when worker.ident is not None.
        if worker.ident is not None:
            # Wait briefly for this worker to finish during shutdown.
            worker.join(timeout=1)
    # Release the USB camera device during shutdown.
    cap.release()
    # Continue with this branch only when not args.headless.
    if not args.headless:
        # Close the OpenCV desktop windows during cleanup.
        cv2.destroyAllWindows()
    # Continue with this branch only when fatal_error.
    if fatal_error:
        # Print a diagnostic, status, or setup message to the console.
        print(fatal_error)
