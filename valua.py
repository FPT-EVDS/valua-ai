import cv2
from PIL import Image
import argparse
import numpy as np
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import os
import time


# def recognize_image(img, conf, targets, name, output, args):
#     count_correct_name = 0
#     try:
#         bboxes, faces = mtcnn.align_multi(img, conf.face_limit, 50)
#     except:
#         bboxes = []
#         faces = []
#     if len(bboxes) == 0:
#         print('no face')
#     else:
#         image = np.array(img)
#         bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
#         bboxes = bboxes.astype(int)
#         bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
#         results, score = learner.infer(conf, faces, targets, True)
#         for idx, bbox in enumerate(bboxes):
#             # print(names[results[idx] + 1] + '____{:.2f}'.format(score[idx]))
#             if names[results[idx] + 1] == name.split("_")[0]:
#                 count_correct_name += 1
#                 # if args.score:
#             image = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), image)
#         # else:
#         #     image = draw_box_name(bbox, names[results[idx] + 1], image)
#         cv2.imwrite(output, image)
#
#     """
#     check accuracy
#     - detect 1 faces correct name
#     """
#     if count_correct_name == 1:
#         return True
#     else:
#         print("Incorrect", output)
#         return False

def load_model():
    mtcnn = MTCNN()
    print('mtcnn loaded')

    # load config
    conf = get_config(False)

    # load model arcface
    learner = face_learner(conf, True)
    learner.threshold = 1.54

    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    model = learner.model
    model.eval()

    return mtcnn, conf, model, learner

# if __name__ == '__main__':
#     start = time.time()
#     parser = argparse.ArgumentParser(description='for face verification')
#     parser.add_argument("-f", "--file_name", help="image file name", type=str)
#     parser.add_argument("-fo", "--folder", help="image file dictionary to recognize", type=str)
#     parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
#     parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
#     parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
#     parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
#     parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
#     parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
#     parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
#
#     args = parser.parse_args()
#
#     maxsize = (2000, 2000)
#
#     conf = get_config(False)
#
#     mtcnn = MTCNN()
#     print('mtcnn loaded')
#
#     learner = face_learner(conf, True)
#     learner.threshold = args.threshold
#
#     if conf.device.type == 'cpu':
#         learner.load_state(conf, 'cpu_final.pth', True, True)
#     else:
#         learner.load_state(conf, 'final.pth', True, True)
#     learner.model.eval()
#     print('learner loaded')
#
#     if args.update:
#         targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
#         print('facebank updated')
#     else:
#         targets, names = load_facebank(conf)
#         print('facebank loaded')
#
#     end = time.time()
#
#     if args.folder:
#         num_image = 0
#         num_correct = 0
#         for filename in os.listdir(args.folder):
#             num_image = num_image + 1
#             img = Image.open(os.path.join(args.folder, filename))
#             img.thumbnail(maxsize, Image.ANTIALIAS)
#             if img is not None:
#                 name = os.path.splitext(filename)[0]
#                 file_extension = os.path.splitext(filename)[1]
#                 output = 'data/result/' + name + '_result' + file_extension
#                 if recognize_image(img, conf, targets, name, output, args):
#                     num_correct = num_correct + 1
#         # statistic
#         print('Total image: ', num_image)
#         print('Total correct: ', num_correct)
#         print('Accuracy: ', num_correct / num_image)
#         end = time.time()
#
#     if args.file_name:
#         filename = args.file_name
#         img = Image.open(args.file_name)
#
#         img.thumbnail(maxsize, Image.ANTIALIAS)
#         name = os.path.splitext(filename)[0]
#         file_extension = os.path.splitext(filename)[1]
#         output = name + '_result' + file_extension
#         if recognize_image(img, conf, targets, name, output, args):
#             print("------FINISH-------")
#         end = time.time()
#
#     print("The time of execution of above program is :", end - start)
