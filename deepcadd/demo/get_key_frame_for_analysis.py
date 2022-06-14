import cv2
import numpy as np

from skimage.filters import frangi
from matplotlib import pyplot as plt

def get_video_frames(video_path):
    cap= cv2.VideoCapture(video_path)
    i=0
    frame_array = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            cv2.destroyAllWindows()
            return frame_array
        frame_array.append(frame)


def get_key_frame(video_path):
    
    max_n_of_pixels = 0
    max_n_of_pixels_idx = -1
    # frangi_result = ""

    frame_array = get_video_frames(video_path)

    for x, frame in enumerate(frame_array):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frangi_img = frangi(gray_image)
        norm_img = cv2.normalize(src=frangi_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gray = cv2.bilateralFilter(norm_img, 30, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel, iterations = 2)
        blur = cv2.blur(closing,(15,15))

        gray = cv2.medianBlur(blur,5)
        gray = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,45,0)
        contours, histogram= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Found "+str(len(contours))+" contours")
        rgb_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        screenCnt = []

        for contour in contours:
            if len(contour) > 600 and len(contour) < 1500:
                screenCnt.append(contour)

        mask = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(mask, screenCnt, -1, 255, -1)
        current_num_of_pixels = np.sum(mask == 255)
        if  current_num_of_pixels > max_n_of_pixels:
            max_n_of_pixels = current_num_of_pixels
            max_n_of_pixels_idx = x
            #frangi_result = mask
    print(max_n_of_pixels_idx)
    return frame_array[max_n_of_pixels_idx], max_n_of_pixels_idx