from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')
import io
from torchvision import transforms as trans
from data.data_pipe import de_preprocess
import torch
from model import l2_norm
import pdb
import cv2


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def prepare_facebank(conf, model, mtcnn, tta=True):
    parrent_dir = "mtcnn"
    model.eval()
    embeddings = []
    names = ['Unknown']
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    if img.size != (112, 112):
                        img = mtcnn.align(img)
                        # if not os.path.exists(os.path.join(parrent_dir, os.path.split(file)[0])):
                        #     os.makedirs(os.path.join(parrent_dir, os.path.split(file)[0]))
                        # cv2.imwrite(os.path.join(parrent_dir, os.path.split(file)[0], os.path.split(file)[1]),
                        #             np.array(img))
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path / 'facebank.pth')
    np.save(conf.facebank_path / 'names', names)
    return embeddings, names


def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path / 'facebank.pth')
    names = np.load(conf.facebank_path / 'names.npy')
    return embeddings, names


def face_reader(conf, conn, flag, boxes_arr, result_arr, learner, mtcnn, targets, tta):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:
            bboxes, faces = mtcnn.align_multi(image, limit=conf.face_limit)
        except:
            bboxes = []

        results = learner.infer(conf, faces, targets, tta)

        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
            assert bboxes.shape[0] == results.shape[0], 'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0  # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1  # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0


hflip = trans.Compose([
    de_preprocess,
    trans.ToPILImage(),
    trans.functional.hflip,
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def draw_box_name(bbox, name, frame):
    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
    frame = cv2.putText(frame,
                        name,
                        (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)
    return frame


def cosine_distance(vector_x, vector_y):
    """
    Caculate distance vector x and vector y by cosine distance
    Formular:
    Args:
        vector_x:
        vector_y:

    Returns:
        distance from 2 vector
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    vector_x = vector_x.to(device)
    vector_y = vector_y.to(device)
    cos = torch.nn.CosineSimilarity(dim=1)
    cos_similarity = cos(vector_x, vector_y)
    cos_distance = 1.0 - cos_similarity.item()
    return cos_distance


def euclidean_distance(vector_x, vector_y):
    """
     Caculate distance vector x and vector y by euclidean distance
    Formular:
    Args:
        vector_x:
        vector_y:

    Returns:
        distance from 2 vector
    """

    return torch.sum(torch.pow(vector_x - vector_y, 2), dim=1)


def verify_face_image(path_image1, path_image2, mtcnn, model, conf, metric="cosine"):
    vector_x = get_embedding(path_image1, mtcnn, model, conf)
    vector_y = get_embedding(path_image2, mtcnn, model, conf)
    if metric == "cosine":
        return cosine_distance(vector_x, vector_y)
    else:
        return euclidean_distance(vector_x, vector_y)


def verify_face_file(file, image_verify, mtcnn, model, conf, metric="cosine"):
    maxsize = (2000, 2000)

    img = Image.open(image_verify)
    img.thumbnail(maxsize, Image.ANTIALIAS)

    try:
        bboxes, faces = mtcnn.align_multi(img, conf.face_limit, 50)
    except:
        bboxes = []
        faces = []
    if len(bboxes) == 0:
        return False
    else:
        # get face's bounding box largest
        img = np.array(img)
        bboxes = bboxes[:, :-1]
        bboxes = bboxes.astype(int)
        area_boxxes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        max_index = np.where(area_boxxes == np.amax(area_boxxes))

        # bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
        target = torch.load(file)

        vector_x = get_embedding(faces[int(max_index[0])], mtcnn, model, conf)

        # TODO: removed
        img = draw_box_name(bboxes[int(max_index[0])], str(cosine_distance(vector_x, target)), img)
        cv2.imwrite("data/test/test.jpg", img)

        if vector_x is not None:
            if metric == "cosine":
                return cosine_distance(vector_x, target)
            else:
                return euclidean_distance(vector_x, target)
    return False


def get_embedding(image, mtcnn, model, conf):
    embs = []
    # embedding
    size = 112, 112
    if image:
        if isinstance(image, str):
            image = Image.open(image)
        if image.size != (112, 112):
            try:
                image = mtcnn.align(image)
            except Exception:
                image.thumbnail(size, Image.ANTIALIAS)
                pass
        with torch.no_grad():
            # img.show()
            mirror = trans.functional.hflip(image)
            emb = model(conf.test_transform(image).to(conf.device).unsqueeze(0))
            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
            embs.append(l2_norm(emb + emb_mirror))
    if len(embs) != 0:
        embedding = torch.cat(embs).mean(0, keepdim=True)
        return embedding


def get_embbed_floder(images, conf, mtcnn, model):
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
