#- * - coding: UTF - 8 -*-
import cv2 as cv
import numpy as np


class YOLO:
    def __init__(self, cfg, weights, image_resize): 
        self.image_resize=image_resize
        self.net = cv.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)    #cv.dnn.DNN_BACKEND_INFERENCE_ENGINE
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)         #切换到GPU：cv.dnn.DNN_TARGET_OPENCL (only Intel GPU)

        #self.classes = ['dlb12s', 'dqzsd_1', 'fhzdq_1', 'jddq_0', 'xnkg_0', 'scdq_0', 'scdq_1', 'jddq_1', 'xnkg_1', 'sxwd', 'dqzsd_0', 
        #                'yb_0', 'dyb250', 'dlb12', 'cndq_0', 'hz_1', 'fz_1', 'cndq_1', 'fhzdq_0', 'xnkg_2', 'sf6_0', 'kk_1', 'yx_1', 
        #                'ddxsq_0', 'hz_0', 'cn_0', 'kk_0', 'scjx_0', 'scjx_1', 'yx_0', 'fz_0', 'cn_1', 'dyb450', 'yb_1']


    def detect(self, frame, conf_thresh=0.25, nmsThreshold=0.45):
        if frame is None:
            return None
        blob = cv.dnn.blobFromImage(frame, 1/255, self.image_resize, [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames())    #网络前向计算
        result = self.postprocess(frame, outs, conf_thresh, nmsThreshold)
        return result


    def postprocess(self, frame, outs, conf_thresh, nmsThreshold):
        # Remove the bounding boxes with low confidence using non-maxima suppression
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                
                confidence = scores[classId]
                if confidence > conf_thresh:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        
        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_thresh, nmsThreshold)

        result = []
        for i in indices:
            i = i[0]
            box = np.array(boxes[i])
            box[box < 0] = 0
            xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
            #label = self.classes[classIds[i]]
            obj = [xmin, ymin, xmax, ymax, confidences[i]]
            result.append(obj)
            
     
        return np.array(result)    


    def getOutputsNames(self):
        # Get the names of the net output layers
        layersNames = self.net.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
