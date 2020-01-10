import os
import time
import argparse
import cv2
import numpy as np
from yolo import YOLO
from sort import Sort


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        default='model/F8.cfg')
    parser.add_argument('--weights',
                        default='model/F8.weights')
    parser.add_argument('--image_resize', default=(320, 320), type=int)
    parser.add_argument('--det_conf_thresh', default=0.25, type=float)
    parser.add_argument('--video',default="video/test15.mp4")
    parser.add_argument('--sort_max_age',default=5,type=int)
    parser.add_argument('--sort_min_hit',default=3,type=int)
    return parser.parse_args()


def detrk(frame, det_conf_thresh, colours):
    s = time.time()
    result = Detector.detect(frame, args.det_conf_thresh)
    im=frame.copy()
    if len(result) > 0:
        det=result[:,0:5]
        trackers = mot_tracker.update(det)
        keep_line_idx = []
        for d in trackers:
            xmin=int(d[0])
            ymin=int(d[1])
            xmax=int(d[2])
            ymax=int(d[3])
            label=int(d[4])
            keep_line_idx.append(label)
            if label in all_pts:
                all_pts[label].append(((xmin+xmax)//2, (ymin+ymax)//2))
            else:
                all_pts[label] = [((xmin+xmax)//2, (ymin+ymax)//2)]
            cv2.putText(im, 'Obj %d'%d[4], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(int(colours[label%32,0]),int(colours[label%32,1]),int(colours[label%32,2])),3)

    '''  Draw history lines '''
    for l,pts in all_pts.items():
            if l in keep_line_idx:
                for i in range(1,len(pts)):
                    if pts[i-1]is None or pts[i]is None:
                        continue
                    #thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                    cv2.line(im, pts[i - 1], pts[i], (int(colours[l%32,0]),int(colours[l%32,1]),int(colours[l%32,2])), 2)

    fps = 1./float(time.time() - s)
    cv2.putText(im, 'FPS: {:.1f} Objs: {}'.format(fps, len(result)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    return im


if __name__=="__main__":
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')  #'I','4','2','0') #cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter('./track.avi', fourcc, 30.0, (1920,1080))

    args=parse_args()
    Detector = YOLO(args.cfg, args.weights,args.image_resize)
    mot_tracker = Sort(args.sort_max_age, args.sort_min_hit) 
    colours = np.random.rand(32,3)*255
    cap = cv2.VideoCapture(args.video)

    all_pts = {}
    while True:
        ret, frame = cap.read()
        if ret == True:
            im = detrk(frame, args.det_conf_thresh, colours)
            cv2.imshow("Tracking", im)
            if cv2.waitKey(1) & 0xff == 27:
                break
            out.write(im)
        else:
            break
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    
        
        
      
    
    
