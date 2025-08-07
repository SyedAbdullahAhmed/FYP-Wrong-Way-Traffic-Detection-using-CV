import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from code_function.vehicle_utils import save_vehicle_image, log_vehicle_data, ensure_browser_compatible_mp4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("custom_model.pt").to(device)
tracks = DeepSort(max_age=25, n_init=5, nn_budget=50,
                    max_cosine_distance=0.3,  # Tighter matching threshold
    max_iou_distance=0.7,
    gating_only_position=False)
location="nipa"
def detect_wrong_way_twolane(
    video_path,
    static_folder="static",
    collection=None,
    job_id=None,
    progress_dict={},
    stop_event=None,
    box1=None, direction_flag1=True,
    box2=None, direction_flag2=True,
    location="nipa"
):
    default_color = (255, 0, 0)
    correct_way_color = (0, 255, 0)
    wrong_way_color = (0, 0, 255)

    cap = cv2.VideoCapture(video_path)
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    if box1 is None or box2 is None:
        box_height = int(FRAME_HEIGHT * 0.18)
        box_width = int(FRAME_WIDTH * 0.45)
        gap = int(FRAME_WIDTH * 0.04)
        y_center = int(FRAME_HEIGHT * 0.7)
        y1 = y_center - box_height // 2
        y2 = y_center + box_height // 2
        x1_1 = 0
        x2_1 = box_width
        x1_2 = box_width + gap
        x2_2 = x1_2 + box_width
        print(f"Box1: ({x1_1}, {y1}, {x2_1}, {y2}), Box2: ({x1_2}, {y1}, {x2_2}, {y2})")
        box1 = (x1_1, y1, x2_1, y2)
        box2 = (x1_2, y1, x2_2, y2)
        # box1 = (0, 270, 445, 377)     # (x1, y1, x2, y2) for lane 1
        # box2=(500, 270, 700, 377)

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{video_filename}_output.mp4"
    browser_output_path = os.path.join(static_folder, 'outputtwolane.mp4')
    output_path = os.path.join(static_folder, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter with mp4v codec.")

    vehicle_info1 = {}
    vehicle_info2 = {}
    wrong_way_count1 = 0
    wrong_way_count2 = 0
    counted_wrong_way_vehicles1 = set()
    counted_wrong_way_vehicles2 = set()
    output_folder = os.path.join(static_folder, "wrong_way_vehicles")
    os.makedirs(output_folder, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened():
        if stop_event and stop_event.is_set():
            progress_dict[job_id] = {'progress': 0, 'output_video': None, 'cancelled': True}
            break
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Draw zones
        cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 255), 2)
        cv2.putText(frame, "Zone 1", (box1[0]+5, box1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), (255, 0, 255), 2)
        cv2.putText(frame, "Zone 2", (box2[0]+5, box2[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # YOLOv8 Tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640)[0]

        tracks = []
        if results.boxes.data is not None:
            for det in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, track_id, conf, cls_id = det[:7]
                tracks.append({
                    "id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": int(cls_id)
                })

        for track in tracks:
            track_id = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            color = default_color
            label = None

            with open(os.path.join(static_folder, "centroids.txt"), "a") as f:
                f.write(f"{track_id},{centroid[0]},{centroid[1]}\n")

            in_zone1 = (box1[0] <= centroid[0] <= box1[2]) and (box1[1] <= centroid[1] <= box1[3])
            in_zone2 = (box2[0] <= centroid[0] <= box2[2]) and (box2[1] <= centroid[1] <= box2[3])

            # Zone 1
            if in_zone1:
                vehicle_info1.setdefault(track_id, []).append(centroid)
                if len(vehicle_info1[track_id]) > 10:
                    vehicle_info1[track_id].pop(0)
                if len(vehicle_info1[track_id]) > 3:
                    y_diff = vehicle_info1[track_id][-1][1] - vehicle_info1[track_id][0][1]
                    movement_threshold = 20
                    if (direction_flag1 and y_diff >= movement_threshold) or (not direction_flag1 and y_diff <= -movement_threshold):
                        if track_id not in counted_wrong_way_vehicles1:
                            wrong_way_count1 += 1
                            counted_wrong_way_vehicles1.add(track_id)
                            image_path = save_vehicle_image(frame, track_id, x1, y1, x2, y2, output_folder)
                            log_vehicle_data(track_id, "vehicle", image_path, collection,location)
                        color = wrong_way_color
                        label = "Wrong Way"
                    else:
                        color = correct_way_color
                        label = "Right Way"
                if track_id in counted_wrong_way_vehicles1:
                    color = wrong_way_color
                    label = "Wrong Way"

            # Zone 2
            elif in_zone2:
                vehicle_info2.setdefault(track_id, []).append(centroid)
                if len(vehicle_info2[track_id]) > 10:
                    vehicle_info2[track_id].pop(0)
                if len(vehicle_info2[track_id]) > 3:
                    y_diff = vehicle_info2[track_id][-1][1] - vehicle_info2[track_id][0][1]
                    movement_threshold = 20
                    if (direction_flag2 and y_diff >= movement_threshold) or (not direction_flag2 and y_diff <= -movement_threshold):
                        if track_id not in counted_wrong_way_vehicles2:
                            wrong_way_count2 += 1
                            counted_wrong_way_vehicles2.add(track_id)
                            image_path = save_vehicle_image(frame, track_id, x1, y1, x2, y2, output_folder)
                            log_vehicle_data(track_id, "vehicle", image_path, collection,location)
                        color = wrong_way_color
                        label = "Wrong Way"
                    else:
                        color = correct_way_color
                        label = "Right Way"
                if track_id in counted_wrong_way_vehicles2:
                    color = wrong_way_color
                    label = "Wrong Way"

            # Draw label
            if label:
                cv2.putText(frame, f"{label} ID {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
         #   cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Overlay stats
        cv2.putText(frame, f"Wrong-Way Zone1: {wrong_way_count1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Wrong-Way Zone2: {wrong_way_count2}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        out.write(frame)
        progress = int((current_frame / total_frames) * 100)
        if progress >= 100:
            progress = 95
        progress_dict[job_id] = {
            'progress': progress,
            'output_video': os.path.basename(browser_output_path)
        }

    cap.release()
    out.release()
    success = ensure_browser_compatible_mp4(output_path, browser_output_path)
    progress_dict[job_id] = {'progress': 100, 'output_video': 'outputtwolane.mp4' if success else None}
    return browser_output_path if success else None
