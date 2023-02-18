import csv
import os
import glob
import dlib
import numpy as np
import pandas as pd
import cv2

def data_writer_txt(data):
    with open("file_not_found2_testing.txt", 'a') as file:
        file.write(str(data))
        file.write("\n")
    file.close()

def data_writer_csv(data):
    with open("data_Testing_celeba.csv", 'a',newline='') as file:
        order = ['name_image', 'box_image', 'coord_point_image']
        writer = csv.DictWriter(file, fieldnames=order)
        writer.writerow(data)

k = 0
face_detector_images = dlib.get_frontal_face_detector()
winndows_ = dlib.image_window()
foud_dir ="C:/Users/jacques kafack/PycharmProjects/FaceDetectorTesiFinale//64_celebaMask-HQ-img/Testing/"
short_pts = ".jpg_imglab"

dictionary_list  = {}
for file1 in os.listdir(foud_dir):
    try:
        #if file1 == "Train":
        #print(file1)
        file_img = glob.glob(foud_dir + "/*.jpg")
        # file_pts = glob.glob(foud_dir + file1 + "/*.pts")
        for Path_images in file_img:
            count = 0
            count1 = 0
            image_detect = dlib.load_rgb_image(Path_images)
            detects = face_detector_images(image_detect, 1)
            image = cv2.imread(Path_images, cv2.IMREAD_UNCHANGED)
            k = k +1
            print("Number of faces detected: {}".format(len(detects)))
            if os.path.exists(Path_images):
                base1 = os.path.basename(str(Path_images))
                os.path.splitext(base1)
                var1 = os.path.splitext(base1)[0]
                pts = os.path.join(foud_dir + var1 +".pts")
                file_pts_table = pd.read_table(pts)
                #print(Path_images, file_pts_table)
                txt_np = np.array(file_pts_table[2:70])
            else:
                print("not corrisponde")
            matrix = []
            for line in txt_np:
                for val in line:
                    vals = tuple(int(i) for i in val.split(' '))
                    x, y = vals
                    matrix.append(vals)
            for j, face in enumerate(detects):
                left = face.left() - int(face.left() * 22 / 100)
                top = face.top() + int(face.top() * 2 / 100)
                right = int(face.right() * 6 / 100) + face.right()
                bottom = int(face.bottom() * 8 / 100) + face.bottom()
                boxes1 = (left, top)
                boxes2 = (right, bottom)
               #rect_detect = cv2.rectangle(image_detect, boxes1, boxes2, (0, 0, 255), 2)
                # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #     j, left, top, right, bottom))
                for i in matrix:
                    x = i[0]
                    y = i[1]
                    if (x, y) < boxes1:
                        count = count + 1
                    elif (x, y) > boxes2:
                        count = count + 1
                    elif y > boxes1[1] and y > boxes2[1]:
                        count = count + 1
                    elif x > boxes1[0] and x > boxes2[0]:
                        count = count + 1
                    if 0 < count < 10:
                        count = 0
                        left = face.left() - int(face.left() * 55 / 100)
                        top = face.top() - int(face.top() * 15 / 100)
                        right = face.right() + int(face.right() * 20 / 100)
                        bottom = face.bottom() + int(face.bottom() * 15 / 100)
                        boxes1 = (left, top)
                        if right > image_detect.shape[0]:
                            right = image_detect.shape[0]
                        if bottom > image_detect.shape[1]:
                            bottom = image_detect.shape[1]
                        boxes2 = (right, bottom)
                        if (x, y) < boxes1:
                            count1 = count1 + 1
                        if (x, y) > boxes2:
                            count1 = count1 + 1
                        if y > boxes1[1] and y > boxes2[1]:
                            count1 = count1 + 1
                        if x > boxes1[0] and x > boxes2[0]:
                            count1 = count1 + 1
                        rect_img = cv2.rectangle(image_detect, boxes1, boxes2, (0, 0, 255), 2)
                    else:
                        rect_img = cv2.rectangle(image_detect, boxes1, boxes2, (0, 255, 255), 2)
                    landmak_face = cv2.circle(rect_img, (x, y), 4, (255, 0, 255), -1)
                    for number, value in enumerate(matrix):
                        landmak_face = cv2.putText(landmak_face, str(number), value, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                   (255, 0, 0), 1,
                                                   cv2.LINE_AA)
                        landmark_detect = cv2.putText(rect_img, str(left), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                      (36, 255, 12), 1, cv2.LINE_AA)
                        landmark_detect = cv2.putText(rect_img, str(right), (right, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                      (36, 255, 12), 1, cv2.LINE_AA)
                        landmark_detect = cv2.putText(rect_img, str(bottom), (right, bottom), cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.9,
                                                      (36, 255, 12), 1, cv2.LINE_AA)
                        landmark_detect = cv2.putText(rect_img, str(top), (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                      (36, 255, 12), 1, cv2.LINE_AA)
                if count == 0 and count1 == 0:
                    list_box = boxes1 + boxes2
                    #bb = {"name_image": Path_images[63:], "box_image": list_box, "coord_point_image": matrix}
                    dictionary_list = {"name_image": Path_images[64:].replace("\\","/"), "box_image": list_box, "coord_point_image": matrix}
                    #data_writer_csv(dictionary_list)
                    print("list zip",dictionary_list)
                else:
                    #data_writer_txt(Path_images[64:].replace("\\","/"))
                    print("name image save to file ;point out")
            landmark_face = cv2.resize(landmark_detect, (500, 500))
            winndows_.clear_overlay()
            winndows_.set_image(landmark_face)
            print("point out ", count)
            print("point out second box", count1)
            print("total number file : {} ".format(k))
            dlib.hit_enter_to_continue()
    except IOError:
        print("File not accessible")
    break