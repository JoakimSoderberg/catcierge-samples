#!/usr/bin/env python

import time
import os
import numpy as np
import cv2
import cv2.cv as cv
from common import clock, draw_str

help_message = '''
USAGE: testcascade.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<image directory>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print help_message

    args, img_dir = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])

    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    img_count = 0
    match_count = 0

    for img_path in os.listdir(img_dir[0]):
        print("%s" % img_path)
        img = cv2.imread(os.path.join(img_dir[0], img_path))
        if img == None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        print("Found %d matches" % len(rects))

        for x1, y1, x2, y2 in rects:
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)

        img_count += 1

        #time.sleep(0.1)

        if (len(rects) > 0):
            match_count += 1
            #cv2.waitKey(0)
        else:
            cv2.waitKey(0)

        if 0xFF & cv2.waitKey(5) == 27:
            break

    print("%d of %d matches ok (%d)" % (match_count, img_count, float(match_count) / img_count))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
