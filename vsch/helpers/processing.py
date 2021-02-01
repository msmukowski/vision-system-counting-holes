import cv2 as cv
import numpy as np

class Picture:
    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.image = cv.imread(image_path)
        self.objects = None
        self.image_preprocessed = None
    
    def preprocessing(self):
        img = cv.resize(self.image, (0, 0), fx=0.2, fy=0.2, interpolation=cv.INTER_AREA) # 0.2
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        copy = img.copy()
        threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 101, 45)
        dilate = cv.dilate(threshold, np.ones((3, 3), np.uint8), iterations=8)
        erode = cv.erode(dilate, np.ones((3, 3), np.uint8), iterations=3)
        mask = cv.medianBlur(erode, 5)
        self.image_preprocessed = cv.bitwise_and(copy, copy, mask=mask)
        #cv.imshow('masked', self.image_preprocessed)
        #cv.waitKey(0)
        #cv.imwrite(self.image_path[:-4]+'_processed.jpg', \
        #        cv.resize(self.image_preprocessed, \
        #        (0, 0), fx=0.52, fy=0.52, interpolation=cv.INTER_CUBIC))

    
    def count_objects(self):
        _, th = cv.threshold(cv.cvtColor(self.image_preprocessed ,cv.COLOR_BGR2GRAY),0,255,cv.THRESH_BINARY)
        contours, _ = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        #x = 0
        #y = 0
        #z = 0
        #u = 0
        objects = []
        for i,cnt in enumerate(contours):
            height, width, _ = self.image_preprocessed.shape
            roi = np.zeros((height,width,3), np.uint8)
            x = int(min(cnt[:, :, 0]))
            y = int(min(cnt[:, :, 1]))
            z = int(max(cnt[:, :, 0]))
            u = int(max(cnt[:, :, 1]))
            cv.drawContours(roi, contours, i, (255, 255, 255), -1)
            roi = cv.bitwise_and(self.image_preprocessed, roi)
            roi = roi[y:u, x:z]

            counted = self.get_holes(roi)
            print(f'Quantity of counted for {self.image_path} and roi_0{i} = {counted}')
            if counted[0] != 0:
                objects.append(counted)
        #return objects

    def get_holes(self, frame):
        rd, frame_red = self.redMask(frame)
        bl, frame_blue = self.blueMask(frame)
        yl, frame_yellow = self.yellowMask(frame)
        gr, frame_gray = self.grayMask(frame)
        wh, frame_white = self.whiteMask(frame,frame_red,frame_blue,frame_yellow,frame_gray)
        blocksAndCircles = []
        blocksAndCircles.append(0)
        frames = [frame_red,frame_blue,frame_yellow,frame_gray,frame_white]
        frame = cv.resize(frame, (0, 0), fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        for scene in frames:
            scene = cv.resize(scene, (0, 0), fx=2, fy=2, interpolation=cv.INTER_CUBIC)
            bgr = frame.copy()
            empty = frame.copy() * 0
            gray = cv.cvtColor(cv.medianBlur(frame, 3), cv.COLOR_BGR2GRAY)
            try:
                circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=14, param2=12, minRadius=9, maxRadius=13)
                circles = np.uint16(np.around(circles))
                blocksAndCircles[0] = len(circles[0, :])
                for it, i in enumerate(circles[0, :]):
                    # draw the center of the circle
                    cv.circle(empty, (i[0], i[1]), 20, (255, 255, 255), -1)
                    for point in circles[0, :]:
                        width = self.line_width(i[0], i[1], point[0], point[1])
                        if width > 10 and width < 37:
                            cv.line(empty, (i[0], i[1]), (point[0], point[1]), (255, 255, 255), 20)
            except:
                try:
                    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=20, param2=12, minRadius=9,
                                            maxRadius=50)
                    circles = np.uint16(np.around(circles))
                    numCircles = len(circles[0, :])
                    for it, i in enumerate(circles[0, :]):
                        # draw the center of the circle
                        cv.circle(empty, (i[0], i[1]), 20, (255, 255, 255), -1)
                        for point in circles[0, :]:
                            width = self.line_width(i[0], i[1], point[0], point[1])
                            if width > 10 and width < 32:
                                cv.line(empty, (i[0], i[1]), (point[0], point[1]), (255, 255, 255), 20)
                except:
                    pass

            count = self.check_block_gray(empty,scene)
            blocksAndCircles.append(count)
        return blocksAndCircles
    
    def redMask(self, frame):
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        hsv_blurred = hsv#cv.GaussianBlur(hsv,(3,3),7)
        low_red = np.array([0, 35, 0])
        hig_red = np.array([12, 255, 255])
        red_msk_low = cv.inRange(hsv_blurred, low_red, hig_red)
        low_red = np.array([142, 35, 0])
        hig_red = np.array([180, 255, 255])
        red_msk_hig = cv.inRange(hsv_blurred, low_red, hig_red)
        red_msk = cv.bitwise_or(red_msk_low, red_msk_hig)
        red_msk = cv.morphologyEx(red_msk, cv.MORPH_DILATE, np.ones((3, 3), dtype='uint8'), iterations=2)
        red_msk = cv.morphologyEx(red_msk, cv.MORPH_CLOSE, np.ones((3, 3), dtype='uint8'), iterations=3)
        red_msk = cv.morphologyEx(red_msk, cv.MORPH_ERODE, np.ones((3, 3), dtype='uint8'), iterations=3)
        red = cv.bitwise_and(frame, frame, mask=red_msk)
        return red, red_msk


    def blueMask(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_blurred = hsv  # cv.GaussianBlur(hsv,(3,3),7)
        low_blue = np.array([108, 29, 108])
        hig_blue = np.array([118, 255, 255])
        blue_msk = cv.inRange(hsv_blurred, low_blue, hig_blue)
        # blue
        blue_msk = cv.morphologyEx(blue_msk, cv.MORPH_DILATE, np.ones((3, 3), dtype='uint8'), iterations=2)
        blue_msk = cv.morphologyEx(blue_msk, cv.MORPH_CLOSE, np.ones((3, 3), dtype='uint8'), iterations=2)
        blue_msk = cv.morphologyEx(blue_msk, cv.MORPH_ERODE, np.ones((3, 3), dtype='uint8'), iterations=1)
        blue = cv.bitwise_and(frame, frame, mask=blue_msk)
        return blue,blue_msk


    def yellowMask(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_blurred = hsv  # cv.GaussianBlur(hsv,(3,3),7)
        low_yellow = np.array([24, 75, 150])
        hig_yellow = np.array([27, 255, 255])
        yellow_msk = cv.inRange(hsv_blurred, low_yellow, hig_yellow)
        # yellow
        yellow_msk = cv.morphologyEx(yellow_msk, cv.MORPH_DILATE, np.ones((3, 3), dtype='uint8'), iterations=3)
        yellow_msk = cv.morphologyEx(yellow_msk, cv.MORPH_CLOSE, np.ones((3, 3), dtype='uint8'), iterations=4)
        yellow_msk = cv.morphologyEx(yellow_msk, cv.MORPH_ERODE, np.ones((3, 3), dtype='uint8'), iterations=1)
        yellow = cv.bitwise_and(frame, frame, mask=yellow_msk)
        return yellow,yellow_msk


    def grayMask(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_blurred = hsv  # cv.GaussianBlur(hsv,(3,3),7)
        low_gray = np.array([157, 150, 90])
        hig_gray = np.array([175, 235, 255])
        gray_msk = cv.inRange(cv.cvtColor(cv.GaussianBlur(hsv, (7, 7), 5), cv.COLOR_BGR2HSV), low_gray, hig_gray)
        # gray
        gray_msk = cv.morphologyEx(gray_msk, cv.MORPH_DILATE, np.ones((3, 3), dtype='uint8'), iterations=2)
        gray_msk = cv.morphologyEx(gray_msk, cv.MORPH_CLOSE, np.ones((3, 3), dtype='uint8'), iterations=2)
        gray_msk = cv.morphologyEx(gray_msk, cv.MORPH_ERODE, np.ones((3, 3), dtype='uint8'), iterations=4)
        gray = cv.bitwise_and(frame, frame, mask=gray_msk)
        return gray,gray_msk


    def whiteMask(self, frame, red, blue, yellow, gray):
        #combo mask
        combo_mask = red + blue + yellow + gray
        combo_mask = cv.bitwise_and(frame,cv.bitwise_not(cv.cvtColor(combo_mask,cv.COLOR_GRAY2BGR)))
        #white mask
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        low_white = np.array([155, 150, 160])#([0,0,160])
        hig_white = np.array([180, 250, 255])#([255,37,255])
        white_msk = cv.inRange(cv.cvtColor(hsv, cv.COLOR_BGR2HSV), low_white, hig_white)
        complex_white_msk = white_msk+cv.cvtColor(combo_mask,cv.COLOR_BGR2GRAY)
        white = cv.bitwise_and(frame, frame, mask=(complex_white_msk))
        white = cv.bitwise_and(white,combo_mask)
        return white,complex_white_msk


    def check_block_gray(self, map,mask):
        map = cv.cvtColor(map,cv.COLOR_BGR2GRAY)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        check = cv.bitwise_and(map,mask)
        _, check = cv.threshold(check, 0, 255, cv.THRESH_BINARY)
        contour, _ = cv.findContours(check, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        check = cv.cvtColor(check, cv.COLOR_GRAY2BGR)
        empty = check.copy()*0
        numberOfColorBlocks = 0
        temp_list_of_box_edges = []
        for it, cnt in enumerate(contour):
            if cv.contourArea(contour[it]) > 250:
                numberOfColorBlocks += 1
                cv.drawContours(empty, [contour[it]], -1, (255, 255, it*50), -1)
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                temp_list_of_box_edges.append(self.line_width(box[0,0],box[0,1],box[1,0],box[1,1]))
                temp_list_of_box_edges.append(self.line_width(box[1, 0], box[1, 1], box[2, 0], box[2, 1]))
                temp_list_of_box_edges.append(self.line_width(box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                temp_list_of_box_edges.append(self.line_width(box[3, 0], box[3, 1], box[0, 0], box[0, 1]))
                edgeMin = min(temp_list_of_box_edges)
                if edgeMin > 50:
                    numberOfColorBlocks += 1
                box = np.int0(box)
                cv.drawContours(empty, [box], 0, (255, 255, 255), -1)
        return numberOfColorBlocks


    def line_width(self, x1, y1, x2, y2):
        if (x1 >= x2):
            x2 = np.power(x1 - x2, 2)
        else:
            x2 = np.power(x2 - x1, 2)
        if (y1 >= y2):
            y2 = np.power(y1 - y2, 2)
        else:
            y2 = np.power(y2 - y1, 2)
        return np.sqrt(x2 + y2)