import os
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from code_function.vehicle_utils import save_vehicle_image, log_vehicle_data, ensure_browser_compatible_mp4
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("custom_model.pt").to(device)
tracker = DeepSort(max_age=25, n_init=5, nn_budget=50,
                    max_cosine_distance=0.3,  # Tighter matching threshold
    max_iou_distance=0.7,
    gating_only_position=False)

# def iou(boxA, boxB):
#     # boxA and boxB format: [x1, y1, x2, y2]
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     if interArea == 0:
#         return 0.0

#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou
# def detect_wrong_way(video_path, static_folder="static", collection=None, job_id=None, progress_dict={}, stop_event=None):
   
    
#     #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #model = YOLO("custom_model.pt").to(device)

#     direction_flag = True  # Up-Down expected direction
#     cap = cv2.VideoCapture(video_path)
#     FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     FPS = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     video_filename = os.path.splitext(os.path.basename(video_path))[0]
#     output_filename = f"{video_filename}_output.mp4"
#     output_path = os.path.join(static_folder, output_filename)
#     browser_output_path = os.path.join(static_folder, 'output.mp4')

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

#     ROI_Y1 = int(FRAME_HEIGHT * 0.7)
#     ROI_Y2 = int(FRAME_HEIGHT)

#     buffer_size = 10
#     min_frames_for_decision = 5
#     movement_threshold = 15
#     match_distance_threshold = 30
#     min_detection_size = 1000

#     frame_buffer = []
#     counted_vehicles = []
#     wrong_way_count = 0
#     output_folder = os.path.join(static_folder, "wrong_way_vehicles")
#     os.makedirs(output_folder, exist_ok=True)

#     current_frame = 0

#     while cap.isOpened():
#         if stop_event is not None and stop_event.is_set():
#             progress_dict[job_id] = {'progress': 0, 'output_video': None, 'cancelled': True}
#             break

#         ret, frame = cap.read()
#         if not ret:
#             break

#         current_frame += 1
#         frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

#         cv2.rectangle(frame, (0, ROI_Y1), (FRAME_WIDTH, ROI_Y2), (0, 255, 255), 2)
#         cv2.putText(frame, "Detection Zone", (10, ROI_Y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         results = model(frame, device=device, verbose=False)
#         current_detections = []

#         for result in results:
#             for det in result.boxes.data.cpu().numpy():
#                 x1, y1, x2, y2, conf, cls = det[:6]
#                 bbox_area = (x2 - x1) * (y2 - y1)
#                 if conf >= 0.5 and bbox_area >= min_detection_size:
#                     centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
#                     current_detections.append({
#                         'bbox': (x1, y1, x2, y2),
#                         'centroid': centroid,
#                         'confidence': conf,
#                         'class': cls,
#                         'frame_num': len(frame_buffer)
#                     })

#         frame_buffer.append(current_detections)
#         if len(frame_buffer) > buffer_size:
#             frame_buffer.pop(0)

#         if len(frame_buffer) >= min_frames_for_decision:
#             for current_det in current_detections:
#                 cx, cy = current_det['centroid']

#                 if ROI_Y1 <= cy <= ROI_Y2:
#                     matched_history = []
#                     for i, past_dets in enumerate(frame_buffer[:-1]):
#                         best_match = None
#                         best_distance = float('inf')

#                         for det in past_dets:
#                             prev_cx, prev_cy = det['centroid']
#                             distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
#                             if distance < match_distance_threshold and distance < best_distance:
#                                 best_match = det
#                                 best_distance = distance

#                         if best_match is not None:
#                             matched_history.append(best_match)

#                     if len(matched_history) >= min_frames_for_decision - 1:
#                         y_movements = []
#                         frame_diffs = []
#                         for match in matched_history:
#                             y_diff = cy - match['centroid'][1]
#                             frame_diff = current_det['frame_num'] - match['frame_num']
#                             y_movements.append(y_diff)
#                             frame_diffs.append(frame_diff)

#                         y_velocities = [y / f for y, f in zip(y_movements, frame_diffs)]
#                         avg_y_velocity = np.mean(y_velocities)

#                         abs_velocity = abs(avg_y_velocity)
#                         label, color = "Unknown", (255, 255, 0)

#                         if abs_velocity >= movement_threshold / min_frames_for_decision:
#                             if direction_flag and avg_y_velocity > 0:
#                                 label, color = "Wrong Way", (0, 0, 255)
#                                 is_new_vehicle = True
#                                 for c in counted_vehicles:
#                                     if iou(current_det['bbox'], c['bbox']) > 0.5:
#                                         is_new_vehicle = False
#                                         break
#                                 if is_new_vehicle:
#                                     wrong_way_count += 1
#                                     counted_vehicles.append(current_det)
#                                     x1, y1, x2, y2 = map(int, current_det['bbox'])
#                                     image_path = save_vehicle_image(frame, wrong_way_count, x1, y1, x2, y2, output_folder)
#                                     log_vehicle_data(wrong_way_count, "vehicle", image_path, collection)
                                
#                                 # if not any(np.allclose(current_det['bbox'], c['bbox'], atol=50) for c in counted_vehicles):
#                                 #     wrong_way_count += 1
#                                 #     counted_vehicles.append(current_det)
#                                 #     x1, y1, x2, y2 = map(int, current_det['bbox'])
#                                 #     image_path = save_vehicle_image(frame, wrong_way_count, x1, y1, x2, y2, output_folder)
#                                 #     log_vehicle_data(wrong_way_count, "vehicle", image_path, collection)
#                             elif direction_flag and avg_y_velocity <= 0:
#                                 label, color = "Right Way", (0, 255, 0)
#                             elif not direction_flag and avg_y_velocity < 0:
#                                 label, color = "Wrong Way", (0, 0, 255)
#                                 for c in counted_vehicles:
#                                     if iou(current_det['bbox'], c['bbox']) > 0.5:
#                                         is_new_vehicle = False
#                                         break
#                                 if is_new_vehicle:
#                                         wrong_way_count += 1
#                                         counted_vehicles.append(current_det)
#                                         x1, y1, x2, y2 = map(int, current_det['bbox'])
#                                         image_path = save_vehicle_image(frame, wrong_way_count, x1, y1, x2, y2, output_folder)
#                                         log_vehicle_data(wrong_way_count, "vehicle", image_path, collection)
                                    
#                                 # if not any(np.allclose(current_det['bbox'], c['bbox'], atol=50) for c in counted_vehicles):
#                                 #     wrong_way_count += 1
#                                 #     counted_vehicles.append(current_det)
#                                 #     x1, y1, x2, y2 = map(int, current_det['bbox'])
#                                 #     image_path = save_vehicle_image(frame, wrong_way_count, x1, y1, x2, y2, output_folder)
#                                 #     log_vehicle_data(wrong_way_count, "vehicle", image_path, collection)
#                             else:
#                                 label, color = "Right Way", (0, 255, 0)

#                         x1, y1, x2, y2 = map(int, current_det['bbox'])
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                         cv2.putText(frame, f"{label} {abs_velocity:.1f}px/f", (x1, y1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         cv2.putText(frame, f"Wrong-Way Count: {wrong_way_count}", (30, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         out.write(frame)
#         progress_dict[job_id] = {
#             'progress': int((current_frame / total_frames) * 100),
#             'output_video': os.path.basename(browser_output_path)
#         }

#     cap.release()
#     out.release()

#     success = ensure_browser_compatible_mp4(output_path, browser_output_path)
#     if success:
#         progress_dict[job_id] = {'progress': 100, 'output_video': 'output.mp4'}
#         return browser_output_path
#     else:
#         progress_dict[job_id] = {'progress': 100, 'output_video': None}
#         return None
def detect_wrong_way(video_path, static_folder="static", collection=None, job_id=None, progress_dict={}, stop_event=None,location="nipa"):
    # Video Properties
    direction_flag = True   # True for Up-Down, False for Down-Up
    # Colors
    default_color = (255, 0, 0)  # Blue for detected vehicles
    correct_way_color = (0, 255, 0)  # Green for correct movement
    wrong_way_color = (0, 0, 255)  # Red for wrong-way vehicles

    cap = cv2.VideoCapture(video_path)
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    # Output Video Writer
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{video_filename}_output.mp4"
    browser_output_path = os.path.join(static_folder, 'output.mp4')
    output_path = os.path.join(static_folder, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter with mp4v codec.")

    # Detection Box (ROI)
    ROI_Y1 = int(FRAME_HEIGHT * 0.7)
    ROI_Y2 = int(FRAME_HEIGHT)
    ROI_X1 = 0 # Left boundary
    ROI_X2 = int(FRAME_WIDTH )  # Right boundary
    # Tracking Variables
    vehicle_info = {}
    wrong_way_count = 0
    counted_wrong_way_vehicles = set()
    output_folder = os.path.join(static_folder, "wrong_way_vehicles")
    os.makedirs(output_folder, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened():
        if stop_event is not None and stop_event.is_set():
            print("Detection cancelled!")
            progress_dict[job_id] = {'progress': 0, 'output_video': None, 'cancelled': True}
            break
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        # Resize frame ONCE, before detection/tracking/drawing
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
       
        # Draw ROI
        cv2.rectangle(frame, (0, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
        cv2.putText(frame, "Detection Zone", (10, ROI_Y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # YOLO Detection
        results = model(frame, device=device, verbose=False)
        detections = []
        for result in results:
            for det in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det[:6]
                if conf >= 0.5:
                    # Draw YOLO detection box (always correct size)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Light blue
                    detections.append(([x1, y1, x2, y2], conf, int(cls)))

        # DeepSORT Tracking
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            color = default_color
            label = "Not Checked"
           
            if (ROI_X1 <= centroid[0] <= ROI_X2) and (ROI_Y1 <= centroid[1] <= ROI_Y2):
                vehicle_info.setdefault(track_id, []).append(centroid)
                if len(vehicle_info[track_id]) > 10:
                    vehicle_info[track_id].pop(0)

                if len(vehicle_info[track_id]) > 3:
                    y_positions = [pos[1] for pos in vehicle_info[track_id]]
                    y_diff = y_positions[-1] - y_positions[0]
                    movement_threshold = 20

                    if direction_flag:  # Up-Down direction
                        if y_diff >= movement_threshold:
                            color = wrong_way_color
                            label = "Wrong Way"
                            if track_id not in counted_wrong_way_vehicles:
                                wrong_way_count += 1
                                counted_wrong_way_vehicles.add(track_id)
                                image_path = save_vehicle_image(frame, track_id, x1, y1, x2, y2, output_folder)
                                log_vehicle_data(track_id, "vehicle", image_path, collection,location)
                        else:
                            color = correct_way_color
                            label = "Right Way"
                    else:  # Down-Up direction
                        if y_diff <= -movement_threshold:
                            color = wrong_way_color
                            label = "Wrong Way"
                            if track_id not in counted_wrong_way_vehicles:
                                wrong_way_count += 1
                                counted_wrong_way_vehicles.add(track_id)
                                image_path = save_vehicle_image(frame, track_id, x1, y1, x2, y2, output_folder)
                                log_vehicle_data(track_id, "vehicle", image_path, collection,location)
                        elif y_diff > movement_threshold:
                            color = correct_way_color
                            label = "Right Way"

                    if track_id in counted_wrong_way_vehicles:
                        color = wrong_way_color
                        label = "Wrong Way"

            # Draw only the tracking box, colored by logic
            #cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f" ID {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Wrong-Way: {wrong_way_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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

    # After detection, re-encode the output video
    raw_output_path = output_path
    success = ensure_browser_compatible_mp4(raw_output_path, browser_output_path)
    if success:
        progress_dict[job_id] = {'progress': 100, 'output_video': 'output.mp4'}
        return browser_output_path
    else:
        progress_dict[job_id] = {'progress': 100, 'output_video': None}
        return None