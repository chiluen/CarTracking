import cv2
import numpy as np
from numpy.core.records import record

from yolov5.utils.plots import Annotator, colors

class Annotator_car(Annotator):
    def __init__(self, im, line,line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        super().__init__(im, line_width=line_width, font_size=font_size, font=font, pil=pil, example=example)
        self.record = {} #之前的所有center point的位置
        self.line = [int(i) for i in line.split(",")]


        self.number_car = {} #紀錄 Up, Down, Left, Right
        self.previous_record = {} #上一次的record
        self.now_record = {} #這一次的record
        self.finish = [] #已被紀錄通過線的車子

    def fill_missed(self):
        for k in self.previous_record.keys():
            if k not in self.now_record:
                self.now_record[k] = self.previous_record[k]

    def check_enter(self, center_now, center_previous, label): #確認有沒有通過那一條線

        flag = False
        #確認up line
        if (center_now[1] - center_previous[1]) < 0 and (center_now[1] <= self.line[0] <= center_previous[1]):
            #import ipdb; ipdb.set_trace()
            self.number_car["Up"] +=1
            flag = True
        #確認down line
        if (center_now[1] - center_previous[1]) > 0 and (center_now[1] >= self.im.shape[0] - self.line[1] >= center_previous[1]):
            self.number_car["Down"] +=1
            flag = True
        #確認left line
        if (center_now[0] - center_previous[0]) < 0 and (center_now[0] <= self.line[2] <= center_previous[0]):
            self.number_car["Left"] +=1
            flag = True
        #確認right line
        if (center_now[0] - center_previous[0]) > 0 and (center_now[0] >= self.im.shape[1] - self.line[3] >= center_previous[0]):
            self.number_car["Right"] +=1
            flag = True
        if flag:
            self.finish.append(label)
        
    def plot_line(self):
        #self.im.shape[0] height, self.im.shape[1] width :(720,972)
        #cv2 的座標是(width, height), 也就是x,y , 但是im.shape[0]是y, im.shape[1]是x
        
        if self.line[0] != -1: #上面的線
            cv2.line(self.im, (0,0+self.line[0]), (self.im.shape[1],0+self.line[0]), (0,0,255), 2)
        if self.line[1] != -1: #下面的線
            cv2.line(self.im, (0, self.im.shape[0] - self.line[1]), (self.im.shape[1], self.im.shape[0] - self.line[1]), (0, 0, 255), 2)
        if self.line[2] != -1: #左邊的線
            cv2.line(self.im, (0 + self.line[2], 0 ), (0 + self.line[2], self.im.shape[0]), (0, 0, 255), 2)
        if self.line[3] != -1: #右邊的線
            cv2.line(self.im, (self.im.shape[1] - self.line[3], 0 ), (self.im.shape[1] - self.line[3], self.im.shape[0]), (0, 0, 255), 2)
    def number_statistic(self):

        # plot for text information
        put_place = [(self.im.shape[1]-120, 30),(self.im.shape[1]-120, 50),(self.im.shape[1]-120, 70),(self.im.shape[1]-120, 90)]
        index = 0
        for direction, number in self.number_car.items():
            if number != 0:
                cv2.putText(self.im, f"{direction}: {number}", put_place[index],0,0.75,(0,0,255),2)
                index += 1
        #把目前的數值繼承給下一張image
        return self.number_car, self.now_record, self.finish

    def load_record(self, previous_data):
        self.record = previous_data
        for color_index in self.record.keys():
            centers = self.record[color_index]
            for c in centers:
                cv2.circle(self.im, c, 3, colors(color_index, True), -1)
    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        
        #直接指定cv2
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        center = ( int((box[0] + box[2])/2), int((box[1] + box[3])/2) )
        color_index = int(label.split(" ")[0])
        label = label.split(" ")[0]

        #存過去的circle
        try:
            self.record[color_index].append(center)
        except:
            self.record[color_index] = [center]

        self.now_record[label] = center #存這個iteration的centers
        
        if label in self.previous_record and label not in self.finish:
            self.check_enter(center, self.previous_record[label], label)

        cv2.circle(self.im, center, 3, colors(color_index, True), -1)
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)

        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

    def output_record(self):
        return self.record    