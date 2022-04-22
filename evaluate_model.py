import os

import numpy as np
import pandas as pd
import itertools
from utils import get_embedding, verify_face_image
from PIL import Image
import torch

from sklearn.metrics import confusion_matrix
from config import get_config
import argparse
import matplotlib.pyplot as plt
from Learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
from os import listdir
from model import l2_norm

from mtcnn import MTCNN


def prepare_data(data_path):
    name = []
    image_path = []
    for dirpath, dirnames, files in os.walk(data_path):
        if files:
            path = dirpath
            if os.path.isdir(path):
                dirname = os.path.basename(path)
                name.append(dirname)
                file_img_path = []
                for file_image in files:
                    file_img_path.append(os.path.join(dirpath, file_image))
                image_path.append(file_img_path)
    identities = dict(zip(name, image_path))

    # generate pair same identities
    positives = []
    for key, values in identities.items():
        for i in range(0, len(values) - 1):
            for j in range(i + 1, len(values)):
                positive = []
                positive.append(values[i])
                positive.append(values[j])
                # print(positive)
                positives.append(positive)

    positives = pd.DataFrame(positives, columns=["file_x", "file_y"])
    positives["decision"] = "Yes"
    # generate pair different identities
    samples_list = list(identities.values())

    negatives = []
    for i in range(0, len(identities) - 1):
        for j in range(i + 1, len(identities)):
            cross_product = itertools.product(samples_list[i], samples_list[j])
            cross_product = list(cross_product)

            for cross_sample in cross_product:
                negative = []
                negative.append(cross_sample[0])
                negative.append(cross_sample[1])
                # print(negative)
                negatives.append(negative)

    negatives = pd.DataFrame(negatives, columns=["file_x", "file_y"])
    negatives["decision"] = "No"

    df = pd.concat([positives, negatives]).reset_index(drop=True)
    print(df.decision.value_counts())
    return df


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='for face verification')
    # parser.add_argument("-f", "--file_name", help="image file name", type=str)
    # parser.add_argument("-fo", "--folder", help="image file dictionary to recognize", type=str)
    # parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
    # parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    # parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    # parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    # parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    # parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    # parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    #
    # args = parser.parse_args()

    # agedb_30, agedb_30_issame = get_val_pair(conf.emore_folder, 'agedb_30')
    # accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, agedb_30, agedb_30_issame, nrof_folds=10,
    #                                                               tta=True)
    # print('agedb_30 - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    # img = trans.ToPILImage()(roc_curve_tensor)
    # img.show()
    # img.save("data/faces_emore/agedb_30/evaluate.jpg")

    # evaluate by data celeb VN
    """
    Data: celeb-VN ( 24,127 image) 
    folder : /media/slyb/SlyB/1. CAPSTONE_UBUNTU/data
    """

    # read image folder
    folder_path = "/media/slyb/SlyB/1. CAPSTONE_UBUNTU/data/VN-celeb/VN-celeb/"
    test_path = "/home/slyb/Desktop/withface/"
    test_VN_path = "/media/slyb/SlyB/1. CAPSTONE_UBUNTU/data/VN-celeb/test/"
    deepface_test_path = "/media/slyb/SlyB/1. CAPSTONE_UBUNTU/data/deepface_data_Test/"
    df = prepare_data(test_VN_path)
    instances = df[["file_x", "file_y"]].values.tolist()

    maxsize = (2000, 2000)

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    # learner.threshold = args.threshold

    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    cosine = []
    eucliden = []
    for pair in instances:
        result_cosine = verify_face_image(pair[0], pair[1], mtcnn, learner.model, conf, "cosine")
        result_euclidean = verify_face_image(pair[0], pair[1], mtcnn, learner.model, conf, "euclidean")
        print(f"Result cosine: {pair} : {result_cosine}")
        print(f"Result_euclidean: {pair} : {result_euclidean}")
        cosine.append(round(result_cosine, 4))
        eucliden.append(np.round(result_euclidean.cpu().numpy(), 4))

    # distance_cos = []
    # distance_eucli = []

    df["distance_cosine"] = cosine
    df["distance_euclidean"] = eucliden

    df = pd.read_csv('data/arcface/threshold_final.csv')

    tp_mean = round(df[df.decision == "Yes"].mean().values[0], 4)
    tp_std = round(df[df.decision == "Yes"].std().values[0], 4)
    fp_mean = round(df[df.decision == "No"].mean().values[0], 4)
    fp_std = round(df[df.decision == "No"].std().values[0], 4)

    print("Mean of true positives: ", tp_mean)
    print("Std of true positives: ", tp_std)
    print("Mean of false positives: ", fp_mean)
    print("Std of false positives: ", fp_std)

    df[df.decision == "Yes"].distance_cosine.plot.kde()
    df[df.decision == "No"].distance_cosine.plot.kde()

    plt.savefig("data/test/mygraph_1.png")

    # df.to_csv("data/arcface/threshold_pivot_1.csv", index=False)

    "Evaluate model"
    threshold = 0.84
    df["prediction"] = "No"
    idx = df[df.distance_cosine <= threshold].index
    df.loc[idx, 'prediction'] = 'Yes'

    cm = confusion_matrix(df.decision.values, df.prediction.values)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")

    df.to_csv("data/arcface/threshold_final.csv", index=False)
