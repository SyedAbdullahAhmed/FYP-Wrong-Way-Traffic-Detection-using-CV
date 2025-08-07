from flask import Flask, render_template, request, redirect, url_for,Response
import os
import cv2
import torch
from ultralytics import YOLO# Load YOLO model
from deep_sort_realtime.deepsort_tracker import DeepSort
from code_function.vehicle_utils import save_vehicle_image, log_vehicle_data 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("custom_model.pt").to(device)
app = Flask(__name__)
location="nipa"
# Initialize DeepSORT
tracker = DeepSort(max_age=25, n_init=3, nn_budget=30)
def generate_live_feed(collection=None):
    """Generator function to capture and process live video frames."""
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or replace with the camera index

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Video Properties
    direction_flag =True   # True for Up-Down, False for Down-Up
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ROI_Y1 = int(FRAME_HEIGHT * 0.4)  # Start 40% from top
    ROI_Y2 = int(FRAME_HEIGHT * 0.6)  # End at 60% from top
    ROI_X1 = 0 # Left boundary
    ROI_X2 = int(FRAME_WIDTH * 0.25)  # Right boundary
    # Colors
    default_color = (255, 0, 0)  # Blue for detected vehicles
    correct_way_color = (0, 255, 0)  # Green for correct movement
    wrong_way_color = (0, 0, 255)  # Red for wrong-way vehicles
    # Tracking Variables
    vehicle_info = {}
    wrong_way_count = 0
    counted_wrong_way_vehicles = set()
    output_folder = "wrong_way_vehicles"
    os.makedirs(output_folder, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # Resize for consistency

        # Draw the detection box (ROI)
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
        cv2.putText(frame, "Detection Zone", (10, ROI_Y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # YOLO Detection
        results = model(frame, device=device, verbose=False)
        detections = []
        for result in results:
            for det in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                if conf >= 0.5:  # Use confidence threshold only
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Light blue
                    detections.append(([x1, y1, x2, y2], conf, cls))

       
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            label = "Not Checked"
            # Default color (blue for detected vehicles)
            color = default_color

            # Track vehicles in ROI
            if (ROI_X1 <= centroid[0] <= ROI_X2) and (ROI_Y1 <= centroid[1] <= ROI_Y2):
                vehicle_info.setdefault(track_id, []).append(centroid)
                if len(vehicle_info[track_id]) > 10:
                    vehicle_info[track_id].pop(0)

                # Wrong-Way Detection
                if len(vehicle_info[track_id]) > 3:
                    y_positions = [pos[1] for pos in vehicle_info[track_id]]
                    y_diff = y_positions[-1] - y_positions[0]
                    movement_threshold = 20  # Adjust this value

                    if direction_flag:  # Up-Down direction
                        if y_diff >= movement_threshold:
                            color = wrong_way_color  # Red for wrong-way vehicles
                            label = "Wrong Way"
                            if track_id not in counted_wrong_way_vehicles:
                                wrong_way_count += 1
                                counted_wrong_way_vehicles.add(track_id)
                                image_path =save_vehicle_image(frame, track_id, x1, y1, x2, y2, output_folder)
                                log_vehicle_data(track_id, "vehicle", image_path,collection,location)
                        elif y_diff < -movement_threshold:
                            color = correct_way_color  # Green for correct movement
                            label = "Right Way"
                    else:  # Down-Up direction
                        if y_diff <= -movement_threshold:
                            color = wrong_way_color  # Red for wrong-way vehicles
                            label = "Wrong Way"
                            if track_id not in counted_wrong_way_vehicles:
                                wrong_way_count += 1
                                counted_wrong_way_vehicles.add(track_id)
                                image_path =save_vehicle_image(frame, track_id, x1, y1, x2, y2, output_folder)
                                log_vehicle_data(track_id, "vehicle", image_path,collection,location)
                        elif y_diff > movement_threshold:
                            color = correct_way_color  # Green for correct movement
                            label = "Right Way"

                # Draw Bounding Box
                #cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # If vehicle leaves the ROI, remove it from tracking
                if track_id in vehicle_info:
                    del vehicle_info[track_id]

        # Display Wrong-Way Count
        cv2.putText(frame, f"Wrong-Way: {wrong_way_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

