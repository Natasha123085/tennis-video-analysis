from flask import Blueprint, render_template, request, url_for
from analysis.analyze import analyze_video
from werkzeug.utils import secure_filename
import os
import uuid

views = Blueprint('views', __name__, static_folder='../static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@views.route('/')
def home():
    return render_template('home.html')


@views.route('/analyze', methods=['POST'])
def analyze():

    if 'video' not in request.files:
        return render_template('home.html', error="No video uploaded.")

    file = request.files['video']

    if file.filename == "":
        return render_template('home.html', error="No selected file.")

    if not allowed_file(file.filename):
        return render_template(
            'home.html',
            error="Invalid file type. Please upload MP4, AVI, MOV, or MKV video."
        )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return render_template(
            'home.html',
            error="File is too large. Please upload a video smaller than 200 MB."
        )

    original_filename = secure_filename(file.filename)
    file_ext = original_filename.rsplit('.', 1)[1].lower()

    unique_name = str(uuid.uuid4()) + "." + file_ext
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    output_filename = "processed_" + str(uuid.uuid4()) + ".mp4"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    result = analyze_video(filepath, output_path)

    video_url = url_for('static', filename='uploads/' + output_filename)

    return render_template(
        'result.html',
        video_url=video_url,
        result=result
    )