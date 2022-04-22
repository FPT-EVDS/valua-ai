from flask import Flask, request, jsonify, send_file, abort, send_from_directory, current_app

from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
import os
from pathlib import Path

from utils import verify_face_file, get_embbed_floder
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


# app.config['UPLOAD_PATH'] = Path('data') / "spring"


@app.route('/')
def hello_world():
    return 'Hello World'


# return file contain embbed vector 512D
@app.route('/embedding', methods=['GET', 'POST'])
def embedd_folder_image():
    # load
    global conf
    global mtcnn
    global model
    # get folder image and studentID
    if request.method == 'POST' and 'image' in request.files:
        embedding, file_path, filename = get_embbed_floder(request.files.getlist('image'))
        try:
            return send_from_directory(path=filename, directory=file_path, as_attachment=True)
        except FileNotFoundError:
            abort(404)
    else:
        return abort(502, description="Method request must be POST.")
    return abort(502, "Embedding folder fail")


# verify face
@app.route('/verify', methods=['GET', 'POST'])
def verify_face():
    # load
    global conf
    global mtcnn
    global model
    global learner

    # get folder image and studentID
    if request.method == 'POST':
        if 'image' in request.files and 'threshold' in request.form and "file" in request.files:
            file = request.files.get("file")
            image = request.files.get("image")
            threshold = float(request.form.get("threshold"))
            if None not in (file, image, threshold):
                result = verify_face_file(file, image, mtcnn, model, conf, "cosine")
                return "True" if result <= threshold else "False"

        return abort(502, description="Required parameter image,file or threshold is missing")
    return abort(502, description="Method request must be POST.")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_PATH'], filename))


def load_model():
    mtcnn = MTCNN()
    print('mtcnn loaded')

    # load config
    conf = get_config(False)

    # load model arcface
    learner = face_learner(conf, True)
    learner.threshold = 1

    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    model = learner.model
    model.eval()

    return mtcnn, conf, model, learner


if __name__ == '__main__':
    mtcnn, conf, model, learner = load_model()
    app.run(host='0.0.0.0')
