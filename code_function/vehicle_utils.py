import os
import cv2
from datetime import datetime
import ffmpeg
import csv

def save_vehicle_image(frame, track_id, x1, y1, x2, y2, output_folder):
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    # if x2 > x1 and y2 > y1:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    #     image_name = f"vehicle_{track_id}_{timestamp}.jpg"
    #     image_path = os.path.join(output_folder, image_name)
    #     cropped_img = frame[y1:y2, x1:x2]
    #     cv2.imwrite(image_path, cropped_img)
    #     return image_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_name = f"vehicle_{track_id}_{timestamp}.jpg"
    image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(image_path, frame)
    return image_path
    return None

def log_vehicle_data(track_id, vehicle_type, image_path, collection,location="Nipa"):
    new_data = {
        "track_id": track_id,
        "vehicle_type": vehicle_type,
        "image_path": image_path,
        "is_challan_generated": False,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,  # Replace with actual location if available
    }
    collection.insert_one(new_data)
    
def ensure_browser_compatible_mp4(input_path, output_path):
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec='libx264', acodec='aac', strict='experimental', movflags='faststart')
            .overwrite_output()
            .run(quiet=False)  # Set quiet=False to see error output
        )
        return True
    except ffmpeg.Error as e:
        print('FFmpeg error:', e)
        print('stdout:', e.stdout.decode() if e.stdout else '')
        print('stderr:', e.stderr.decode() if e.stderr else '')
        return False
    

def export_wrong_way_vehicles_to_csv(collection, csv_path):
    cursor = collection.find()
    data = list(cursor)
    #print("Fetched data:", data)  # Debug: See what is returned

    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    all_keys.discard('_id')

    if not all_keys:
        all_keys = {'vehicle_id', 'timestamp', 'wrong_way'}

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(all_keys))
        writer.writeheader()
        for item in data:
            row = {k: item.get(k, "") for k in all_keys}  # Fill missing keys with empty string
            writer.writerow(row)
    print(f"Exported {len(data)} wrong-way vehicles to {csv_path}")