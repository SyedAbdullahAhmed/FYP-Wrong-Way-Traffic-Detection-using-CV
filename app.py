from flask import Flask, render_template, request, redirect, url_for, Response, send_file
import os
from dotenv import load_dotenv
from deep_sort_realtime.deepsort_tracker import DeepSort
from pymongo import MongoClient
import threading, uuid
from code_function.pdf_challan_generator import generate_pdf_challan_from_mongo
from code_function.upload_images_to_roboflow import upload_images_to_roboflow
from code_function.detect_wrongway import detect_wrong_way
from code_function.generate_live_feed import generate_live_feed
from code_function.twolaneDetection import detect_wrong_way_twolane
from code_function.vehicle_utils import export_wrong_way_vehicles_to_csv
from flask import jsonify,render_template, request,send_from_directory
import math
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from collections import defaultdict
progress_dict = {}
jobs = {} 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
# MongoDB connection setup
mouriUrl=os.getenv("MONGODDB_URL")
client = MongoClient(mouriUrl)
db = client["car_logs"]
collection = db["logs"]
try:
    client.admin.command('ping')
    print("Connected to MongoDB!")
except Exception as e:
    print(f"Connection failed: {e}")

# Initialize DeepSORT
tracker = DeepSort(max_age=25, n_init=3, nn_budget=30)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pass
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file uploaded", 400
    video = request.files['video']
    if video.filename == '':
        return "No selected file", 400
    area = request.form.get('area')
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)
    #testcodeing
    job_id = str(uuid.uuid4())
    stop_event = threading.Event()
    def run_detection():
        detect_wrong_way(video_path, app.config['STATIC_FOLDER'], collection, job_id, progress_dict, stop_event,area)
    thread = threading.Thread(target=run_detection)
    jobs[job_id] = {'thread': thread, 'stop': stop_event}
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/uploadtwolane', methods=['POST'])
def upload_video_twolane():
    if 'video' not in request.files:
        return "No video file uploaded", 400
    video = request.files['video']
    if video.filename == '':
        return "No selected file", 400
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)
    area = request.form.get('area')
    # Example: Get box coordinates and directions from form data (as integers and booleans)
    box1 = (0, 270, 445, 377)     # (x1, y1, x2, y2) for lane 1
    box2=(500, 270, 840, 377)
   # box1=None
   #case 1 box1 is right box 2 wrong
    direction_flag1 = True           # True = Up-Down, False = Down-Up
    direction_flag2 = True          # True = Up-Down, False = Down-Up
    #case 2 box1 is wrong box 2 is right
    #direction_flag1 = False          # True = Up-Down, False = Down-Up
    #direction_flag2 = False          # True = Up-Down, False = Down-Up
    #case 3 both boxes are right
    # direction_flag1 = True           # True = Up-Down, False = Down-Up
    # direction_flag2 = False           # True = Up-Down, False = Down-Up
   # box2=None
    

    #testcodeing
    job_id = str(uuid.uuid4())
    stop_event = threading.Event()
    def run_detection():
        detect_wrong_way_twolane(
            video_path, app.config['STATIC_FOLDER'], collection, job_id, progress_dict, stop_event,
            box1, direction_flag1, box2, direction_flag2,area
        )
        #detect_wrong_way(video_path, app.config['STATIC_FOLDER'], collection, job_id, progress_dict, stop_event)
    thread = threading.Thread(target=run_detection)
    jobs[job_id] = {'thread': thread, 'stop': stop_event}
    thread.start()
    return jsonify({'job_id': job_id})
    
@app.route('/progress/<job_id>')
def progress(job_id):
    return jsonify(progress_dict.get(job_id, {'progress': 0}))

@app.route('/cancel_detection/<job_id>', methods=['POST'])
def cancel_detection(job_id):
    job = jobs.get(job_id)
    if job:
        job['stop'].set()
        progress_dict[job_id] = {'progress': 0, 'output_video': None, 'cancelled': True}
        return jsonify({'status': 'cancelled'})
    return jsonify({'status': 'not found'}), 404
# Global variable for live video feed
live_video_feed = None
lock = threading.Lock()
@app.route('/live')
def live():
    """Route to render the live detection page."""
    return render_template('live.html')
@app.route('/twolane')
def twolane():
    """Route to render the live detection page."""
    return render_template('twolane.html')

@app.route('/roboflow')
def roboflow():
    """Route to render the live detection page."""
    return render_template('roboflow.html')


@app.route('/live_feed')
def live_feed():
    """Route to provide the live video feed."""
    return Response(generate_live_feed(collection), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/challen')
def challen():   
  
    filter_date = request.args.get('filter_date')
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Base query
    query = {"is_challan_generated": True}
    
    if filter_date:
        print(f"Filter Date: {filter_date}")
        # Create regex pattern to match any time on the selected date
        date_pattern = f"^{filter_date}"
        query["date"] = {"$regex": date_pattern}
        today_str = filter_date
    else:
        # For today's date, use the same regex approach
        date_pattern = f"^{today_str}"
        query["date"] = {"$regex": date_pattern}
    sample = collection.find_one({"is_challan_generated": True, "date": {"$exists": True}})
    if sample:
        challan_data = list(
            collection.find(query)
            .sort("_id", -1).limit(150)
        )
    else:
        challan_data = list(
            collection.find({"is_challan_generated": True})
            .sort("_id", -1)
            .limit(150)
        )
    # Remove '_id' field for cleaner display
    #print("Challan Data:", challan_data)  # Debug: See what is returned
    for item in challan_data:
        item.pop('_id', None)
    current_date = datetime.now().strftime("%Y-%m-%d")
    return render_template('challens.html', data=challan_data, current_date=current_date,filter_date=today_str)
   # return render_template('challens.html')

@app.route('/diagram')
def diagram():   
    return render_template('diagram.html')
@app.route('/upload_to_roboflow', methods=['POST'])
def upload_to_roboflow():
    """Route to upload images to Roboflow."""
    image_folder = "wrong_way_vehicles"  # Folder containing wrong-way vehicle images
    api_key = os.getenv('api_key')  # Replace with your Roboflow API key
    workspace_name = os.getenv('workspace_name')  # Replace with your workspace name
    project_name = os.getenv('project_name')  # Replace with your project name    
    status = upload_images_to_roboflow(image_folder, api_key, workspace_name, project_name)
    return status

@app.route('/generate_challans', methods=['POST'])
def generate_challans():
    """Route to generate PDF challans."""
    db_name = "car_logs"
    collection_name = "logs"
    output_folder = "wrong_way_vehicles"
    pdf_output_folder = "pdf_challans"

    # Call the function to generate PDF challans
    status = generate_pdf_challan_from_mongo(mouriUrl, db_name, collection_name, output_folder, pdf_output_folder)
    return redirect(url_for('challen'))
   # return status
@app.route('/createchallenpdfcsv', methods=['POST'])
def create_challen_pdf_csv():   
    csv_output_folder = os.path.join('static', 'csv_exports')
    os.makedirs(csv_output_folder, exist_ok=True)
    csv_file_path = os.path.join(csv_output_folder, 'wrong_way_vehicles.csv')

    export_wrong_way_vehicles_to_csv(collection, csv_file_path)

    if not os.path.exists(csv_file_path):
        return "CSV file was not created. No wrong-way vehicle data found.", 404

    # Send the CSV file for download
    return send_file(csv_file_path, as_attachment=True)
    #return ""   

@app.route('/challan_table')
def challan_table():
    page = int(request.args.get('page', 1))
    per_page = 10  # Number of rows per page

    total_count = collection.count_documents({})
    total_pages = math.ceil(total_count / per_page)

    skip = (page - 1) * per_page
    cursor = collection.find({'is_challan_generated':True}).skip(skip).limit(per_page)
    data = list(cursor)

    # Remove '_id' and convert ObjectId to string if needed
    for item in data:
        item.pop('_id', None)

    return render_template('challan_table.html', data=data, page=page, total_pages=total_pages)

@app.route('/download_challan_pdf/<trackid>')
def download_challan_pdf(trackid):
    pdf_output_folder = "pdf_challans"
    filename = f"challan_{trackid}.pdf"
    return send_from_directory(pdf_output_folder, filename, as_attachment=True)

@app.route('/report')
def report():  
   
    current_date = datetime.now().strftime("%Y-%m-%d")
    # Get all logs for the current date
    #print(collection)
    # Fetch all logs for the current date (date only, ignoring time)
    data = list(collection.find({
        'date': {'$regex': f'^{current_date}'}
    }))
    #print("Fetched data:", data)  # Debug: See what is returned
    # Count by area and date
    area_counts = defaultdict(int)
    for row in data:
        area = row.get('location', 'Unknown')
        area_counts[area] += 1
    # Prepare summary table
    # Generate dummy data for demonstration (up to 50 rows)
    
    summary = [{'location': area, 'date': current_date, 'count': count} for area, count in area_counts.items()]
    # summary = [
    #     {'location': f'Area {i%5+1}', 'date': current_date, 'count': (i+1)*2}
    #     for i in range(50)
    # ]
    #print(summary)
   
    return render_template('report.html', data=summary, table_data=summary, current_date=current_date)

@app.route('/filter_report', methods=['POST'])
def filter_report():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    area = request.form.get('area')
    #print(f"Start Date: {start_date}, End Date: {end_date}, Area: {area}")
    # Build MongoDB query
    query = {}

    # Date filtering
    if start_date and end_date:
        # To include the entire end_date, add one day to end_date and use $lt
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        query['date'] = {'$gte': start_date, '$lt': end_date_obj.strftime("%Y-%m-%d")}
    elif start_date:
        query['date'] = {'$gte': start_date}
    elif end_date:
        query['date'] = {'$lte': end_date}
    else:
        # Use today's date with regex if no date provided
        current_date = datetime.now().strftime("%Y-%m-%d")
        query['date'] = {'$regex': f'^{current_date}'}

    # Area filtering
    if area:
        query['location'] = area

    # Fetch data from MongoDB
    data = list(collection.find(query))

    # Count by area and date
    area_date_counts = defaultdict(int)
    for row in data:
        loc = row.get('location', 'Unknown')
        date = row.get('date', '')[:10]
        area_date_counts[(loc, date)] += 1

    # Prepare summary table
    table_data = [
        {'location': loc, 'date': date, 'count': count}
        for (loc, date), count in area_date_counts.items()
    ]

    return render_template('report.html', data=table_data, table_data=table_data, current_date=datetime.now().strftime("%Y-%m-%d"),start_date= request.form.get('start_date'),
    end_date=request.form.get('end_date'),
    area= request.form.get('area'))

    

if __name__ == '__main__':
    app.run(debug=True)
