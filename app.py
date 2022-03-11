import torch
from flask import Flask, request, jsonify, send_file, abort, send_from_directory, current_app
from PIL import Image
import cv2
import valua
import numpy as np
from werkzeug.utils import secure_filename
import os
import time

from utils import draw_box_name

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_PATH'] = '/media/slyb/SlyB/1. CAPSTONE_UBUNTU/test_ai/ai/valua_ai/data/spring'


@app.route('/')
def hello_world():
    return 'Hello World'


# return file contain embbed vector 512D
@app.route('/embedding', methods=['GET', 'POST'])
def embedd_folder_image():
    # get folder image and studentID
    print(request.files)
    if request.method == 'POST' and 'image' in request.files:
        # check file dung k
        for f in request.files.getlist('image'):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_PATH'], filename))

        embedding, file_path, filename = get_embbed(request.files.getlist('image'))
        print(embedding)
        try:
            # return jsonify(vector=embedding.tolist())
            return send_from_directory(path=filename, directory=file_path, as_attachment=True)
        except FileNotFoundError:
            abort(404)
    else:
        abort(502)
    return "Connect"


# # verify face
@app.route('/verify', methods=['GET', 'POST'])
def verify_face():
    # get folder image and studentID
    start = time.time()
    if request.method == 'POST' and 'image' in request.files:
        file = request.files.get("file")
        image = request.files.get("image")
        save_file(file)
        save_file(image)
        end = time.time()
        print("The time of execution of save program is :", end - start)
        result = verify_face(request.files.get("file"), request.files.get("image"))
        end = time.time()
        print("The time of execution of verify program is :", end - start)
        return result and "True" or "False"

    # embbed folder image
    # return embbed vector and studentID
    return "Connect"


def get_embbed(images):
    # load
    global conf
    global mtcnn
    global model
    embs = []
    file_name = "file" + '.pth'
    file_path = conf.facebank_path
    # embedding
    for file in images:
        try:
            img = Image.open(file)
        except:
            continue
        if img.size != (112, 112):
            img = mtcnn.align(img)
        with torch.no_grad():
            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
    if len(embs) != 0:
        embedding = torch.cat(embs).mean(0, keepdim=True)
        torch.save(embedding, conf.facebank_path / file_name)
        return embedding.cpu().detach().numpy(), file_path, file_name
    else:
        return False


def verify_face(file, check_image):
    """
    :param file: file .pth face embedding vector in db
    :param image: image taken by app camera
    :return: true / false
    """
    global conf
    global mtcnn
    global model
    global learner

    maxsize = (2000, 2000)

    img = Image.open(check_image)
    img.thumbnail(maxsize, Image.ANTIALIAS)

    if img is not None:
        try:
            bboxes, faces = mtcnn.align_multi(img, conf.face_limit, 50)
        except:
            bboxes = []
            faces = []
        if len(bboxes) == 0:
            print('no face')
            return False
        else:
            image = np.array(img)
            bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
            target = torch.load(conf.data_path / "spring" / file.filename)
            results, score = learner.infer(conf, faces, target, True)
            for idx, bbox in enumerate(bboxes):
                if score[idx] > learner.threshold:
                    return False
                image = draw_box_name(bbox, file.filename + '_{:.2f}'.format(score[idx]), image)
            cv2.imwrite(os.path.join(app.config['UPLOAD_PATH'], check_image.filename), image)
    return True


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_PATH'], filename))


if __name__ == '__main__':
    mtcnn, conf, model, learner = valua.load_model()
    # app.run(debug=True)
    app.run(host='0.0.0.0')
