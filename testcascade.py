#!/usr/bin/env python

import time
import os
import numpy as np
#import skimage
#from skimage import morphology
import cv2
import cv2.cv as cv
from common import clock, draw_str
import argparse
import glob
from subprocess import call

save_all = False

def perform_match(img, snout):
    matchres = cv2.matchTemplate(img, snout, cv2.TM_CCOEFF_NORMED)
    (min_x, max_x, minloc, maxloc) = cv2.minMaxLoc(matchres)
    (x, y) = maxloc

    snout_w = snout.shape[1]
    snout_h = snout.shape[0]

    return (max_x, (x, y), (x + snout_w, y + snout_h))

def template_match(img, vis, snout, snout_contours, flipped_snout, flipped_snout_contours):
    if (snout.shape > img.shape):
        print("Snout %r > image %r" % (snout.shape, img.shape))
        return False

    ret, threshimg = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Use template matching. Normal snout match.
    res, p1, p2 = perform_match(threshimg, snout)
    match_contour = snout_contours

    # See if flipped snout is a better match.
    if res < 0.8:
        print "Flipping"
        res2, p1b, p2b = perform_match(threshimg, flipped_snout)
        if res2 > res:
            print("  Flipped is better %f > %f" % (res2, res))
            res, p1, p2, match_contour = res2, p1b, p2b, flipped_snout_contours

    # Highlight the template match.
    cv2.rectangle(vis, p1, p2, (255, 255, 0), 1)

    # Draw the snout contours.
    vis_roi = vis[p1[1]:p2[1], p1[0]:p2[0]]
    cv2.drawContours(vis_roi, match_contour, -1, (0, 255, 255))

    print(" Template match: %s" % (res,))

    return (res >= 0.8)

def _get_contour_count(img_contours):
    countour_count = 0
    for contour in img_contours:
        area = cv2.contourArea(contour)
        big_enough = (area > 10.0)
        print("   Area %f %s" % (area, "" if big_enough else "(too small)"))
        countour_count += (area > 10.0)

    return countour_count

def adaptive_thresh(img):
    cv2.namedWindow("adpth1")
    cv2.namedWindow("adpth2")
    cv2.namedWindow("adpth3")
    cv2.moveWindow("adpth1", 350, 150)
    cv2.moveWindow("adpth2", 550, 150)
    cv2.moveWindow("adpth3", 750, 150)

    # Adaptive threshold
    adapt_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    cv2.imshow("adpth1", adapt_thresh)

    kernel2 = np.ones((2,2), np.uint8)
    adapt_thresh = cv2.morphologyEx(adapt_thresh, cv2.MORPH_CLOSE, kernel2, iterations = 1)
    """
    adapt_thresh = cv2.dilate(adapt_thresh, kernel2, iterations = 2)
    cv2.imshow("adpth2", adapt_thresh)

    #kernel3 = np.ones((2,2), np.uint8)
    #cv2.erode(adapt_thresh, kernel3, iterations = 3)
    #adapt_thresh = cv2.morphologyEx(adapt_thresh, cv2.MORPH_OPEN, kernel3, iterations = 3)
    #kernel3 = np.ones((12,12), np.uint8)
    adapt_thresh = cv2.erode(adapt_thresh, kernel2, iterations = 4)
    cv2.imshow("adpth3", adapt_thresh)
"""
    return adapt_thresh

def get_direction(img):

    ret, threshimg = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Calculate the direction by comparing the left most
    # and right most columns in the thresholded image.
    # We're most likely going in the direction with more white.
    h, w = img.shape
    print("w: %d, h: %d" % (w, h))
    left_side = sum(threshimg[0:h,0]) / 255
    right_side = sum(threshimg[0:h, (w - 1)]) / 255

    print("White pixel counts => Left side %d, Right side %d" % (left_side, right_side))

    # If the differeence is too close we set it to unknown.
    if abs(left_side - right_side) > 25:
        if left_side > right_side:
            direction = "Left"
        else:
            direction = "Right"
    else:
        direction = "Unknown"

    return direction

def get_prey_contours_alt(img_name, img, vis):
    kernel = np.ones((12,12), np.uint8)
    kernel_tall = np.ones((5, 1), np.uint8)
    cv2.namedWindow("org")
    cv2.moveWindow("org", 450, 0)
    cv2.namedWindow("thresh")
    cv2.moveWindow("thresh", 300, 100)
    #cv2.namedWindow("not_thresh")
    #cv2.moveWindow("not_thresh", 300, 200)
    cv2.namedWindow("adpthresh")
    cv2.moveWindow("adpthresh", 600, 100)
    #cv2.namedWindow("not_adpthresh")
    #cv2.moveWindow("not_adpthresh", 600, 200)
    cv2.namedWindow("combined")
    cv2.moveWindow("combined", 450, 300)
    cv2.namedWindow("opened_combined")
    cv2.moveWindow("opened_combined", 450, 400)
    cv2.namedWindow("eroded_combined")
    cv2.moveWindow("eroded_combined", 450, 500)
    cv2.namedWindow("inv_combined")
    cv2.moveWindow("inv_combined", 450, 600)

    cv2.imshow("org", img)
    if save_all:
        cv2.imwrite("%s_02_original.png" % img_name, img)

    # Normal threshold inverted.
    ret, threshimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow("thresh", threshimg)
    if save_all:
        cv2.imwrite("%s_03_thresh.png" % img_name, threshimg)

    # cv2.THRESH_BINARY_INV above does this..
    #not_thresh = threshimg #cv2.bitwise_not(threshimg)
    #cv2.imshow("not_thresh", not_thresh)

    # Adaptive threshold inverted.
    adpthresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    cv2.imshow("adpthresh", adpthresh)
    if save_all:
        cv2.imwrite("%s_04_adpthresh.png" % img_name, adpthresh)

    #not_adpthresh = adpthresh # cv2.bitwise_not(adpthresh)
    #cv2.imshow("not_adpthresh", not_adpthresh)

    # We invert the thresholding above so we can add them (white is 1, black 0).
    combined = cv2.add(threshimg, adpthresh)
    cv2.imshow("combined", combined)
    if save_all:
        cv2.imwrite("%s_05_combined.png" % img_name, combined)

    kernel = np.ones((2, 2), np.uint8)
    opened_combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations = 2)
    cv2.imshow("opened_combined", opened_combined)
    if save_all:
        cv2.imwrite("%s_06_opened_combined.png" % img_name, opened_combined)

    kernel = np.ones((3, 3), np.uint8)
    dilated_combined = cv2.dilate(opened_combined, kernel, iterations = 3)
    cv2.imshow("dilated_combined", dilated_combined)
    if save_all:
        cv2.imwrite("%s_07_dilated_combined.png" % img_name, dilated_combined)

    # Go back to black on white.
    inv_combined = cv2.bitwise_not(dilated_combined)
    cv2.imshow("inv_combined", inv_combined)
    if save_all:
        cv2.imwrite("%s_08_inv_combined.png" % img_name, inv_combined)

    # Get the image contours.
    threshimg_tmp = inv_combined.copy()
    img_contours, img_hierarchy = cv2.findContours(threshimg_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contour_count = _get_contour_count(img_contours)

    color = (0, 255, 0)
    if contour_count >= 2:
        color = (255, 255, 255)

    # Draw the image contours.
    cv2.drawContours(vis, img_contours, -1, color)

    draw_str(vis, (10, 20), "%d" % contour_count)

    print("Countour count %d" % contour_count)

    if save_all:
        cv2.imwrite("%s_09_contours.png" % img_name, vis)

    return (contour_count == 1)

def get_prey_contours(img_name, img, vis):
    kernel = np.ones((12,12), np.uint8)
    kernel_tall = np.ones((5, 1), np.uint8)

    ret, threshimg = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("thresh", threshimg)

    # Get the skeleton of the image
    # (just used to see if it would be useful. Might be faster than findContours)
    #bla = threshimg
    #bla2 = cv2.bitwise_not(threshimg)
    #skeleton = morphology.skeletonize(bla > 0)
    #skeleton2 = morphology.skeletonize(bla2 > 0)
    #skeleton = skimage.img_as_ubyte(skeleton)
    #skeleton2 = skimage.img_as_ubyte(skeleton2)
    #cv2.imshow("skeleton", skeleton)
    #cv2.imshow("skeleton2", skeleton2)

    # Get the image contours.
    threshimg_tmp = threshimg.copy()
    img_contours, img_hierarchy = cv2.findContours(threshimg_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contour_count = _get_contour_count(img_contours)

    # If we only get one contour, double check if there are more
    # if we morph the image some.
    # (If the prey doesn't hang enough from the mouth to touch the bottom
    # edge of the sub image, this might extend it enough so it does...)
    if contour_count == 1:
        cv2.namedWindow("before")
        cv2.namedWindow("after")
        cv2.moveWindow("before", 350, 50)
        cv2.moveWindow("after", 550, 50)
        print("Got only one contour, morphing")

        threshimg_tmp = threshimg.copy()
        cv2.imshow('before', threshimg_tmp)
        threshimg_tmp = cv2.erode(threshimg_tmp, kernel, iterations = 1)
        threshimg_tmp = cv2.morphologyEx(threshimg_tmp, cv2.MORPH_OPEN, kernel_tall, iterations = 1)
        cv2.imshow('after', threshimg_tmp)

        # Make a copy before it's destroyed for next try below...
        bla = threshimg_tmp.copy()

        img_contours, img_hierarchy = cv2.findContours(threshimg_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Count the contours with a sizeable enough area.
        contour_count = _get_contour_count(img_contours)

    # If we still haven't found anything try adaptive threshold
    # and add that to our current image. This will find thin
    # forms such as a mouse tail that might no be catched by the normal threshold.
    if (contour_count == 1):
        awesome = adaptive_thresh(img)

        # Invert the images so that we can add them
        # (When adding, the areas need to be white)
        awesome = cv2.bitwise_not(awesome)
        thrnot = cv2.bitwise_not(bla)

        # Add the old image and new to get both
        # the profile and thin features.
        awesome = cv2.add(awesome, thrnot)

        # Finally invert the result.
        awesome = cv2.bitwise_not(awesome)

        # Get rid of any extra noise.
        kernel2 = np.ones((3,3), np.uint8)
        awesome = cv2.morphologyEx(awesome, cv2.MORPH_OPEN, kernel2, iterations = 2)
        cv2.imshow("awesome", awesome)

        img_contours, img_hierarchy = cv2.findContours(awesome, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_count = _get_contour_count(img_contours)

    color = (0, 255, 0)
    if contour_count >= 2:
        color = (255, 255, 255)

    # Draw the image contours.
    cv2.drawContours(vis, img_contours, -1, color)

    draw_str(vis, (10, 20), "%d" % contour_count)

    print("Countour count %d" % contour_count)

    return (contour_count == 1)

def detect(img, cascade, minsize):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=minsize, flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade", metavar = "CASCADE", 
                    help = "The path to the cascade xml file.")

    parser.add_argument("--show", action = "store_true",
                    help = "Show the images while iterating over them.")

    parser.add_argument("--pause_fail", action = "store_true",
                    help = "Pause on failed match.")

    parser.add_argument("--pause_ok", action = "store_true",
                    help = "Pause on ok match")

    parser.add_argument("--min_height", metavar = "HEIGHT", type = int,
                    help = "The minimum height of a match.", default = 24)

    parser.add_argument("--min_width", metavar = "WIDTH", type = int,
                    help = "The minimum widht of a match.", default = 24)

    parser.add_argument("--frame_delay", metavar = "SECONDS", type = float,
                    help = "Delay this many seconds between images.", default = 0.0)

    parser.add_argument("--snout", metavar = "SNOUTIMAGE",
                    help = "The snout image.")

    parser.add_argument("--output", metavar = "OUTPUTDIR",
                    default = "output/",
                    help = "Save images in this output dir when pressing 's'.")

    parser.add_argument("--save_all", action = "store_true",
                    help = "Save all the steps for a match (a lot of images)")

    parser.add_argument("--montage", action = "store_true",
                    help = "If --save_all is used, this creates a montage from its output")

    parser.add_argument("--old_prey_method", action = "store_true",
                    help = "Use the old prey matching method.")

    parser.add_argument("images", metavar = "IMAGE", nargs = "+",
                    help = "The Catcierge match images to test. If a directory is specied, all .png files in that directory are used.")

    args = parser.parse_args()

    save_all = args.save_all

    if args.snout:
        snout_img_tmp = cv2.imread(args.snout)
        snout_img_gray = cv2.cvtColor(snout_img_tmp, cv2.COLOR_BGR2GRAY)
        ret, snout_img = cv2.threshold(snout_img_gray, 90, 255, 0)
        flipped_snout_img = cv2.flip(snout_img, 1)

        snout_img_tmp = snout_img.copy()
        flipped_snout_img_tmp = flipped_snout_img.copy()
        snout_contours, _ = cv2.findContours(snout_img_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        flipped_snout_contours, _ = cv2.findContours(flipped_snout_img_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cascade = cv2.CascadeClassifier(args.cascade)
    #nested = cv2.CascadeClassifier(nested_fn)

    image_paths = []

    for img_path in args.images:
        if os.path.isdir(img_path):
            image_paths += glob.glob(img_path + "/*.png")
        else:
            image_paths.append(img_path)

    img_count = 0
    match_count = 0
    w_sum = 0
    h_sum = 0

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        save_name = os.path.join(args.output, img_name)
        print("%s" % img_path)
        img = cv2.imread(img_path)
        if img == None:
            continue

        if save_all:
            cv2.imwrite("%s_00_image.png" % save_name, img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eqhist = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray_eqhist, cascade, (args.min_width, args.min_height))
        match_ok = (len(rects) > 0)
        vis = img.copy()
        #draw_rects(vis, rects, (0, 255, 0))
        print("Found %d matches" % len(rects))
        
        template_match_ok = False
        prey_match_ok = False
        direction = "Unknown"

        for x1, y1, x2, y2 in rects:
            w = (x2 - x1)
            h = (y2 - y1)
            w_sum += w
            h_sum += h

            org_roi = gray[y1:y2, x1:x2]

            if save_all:
                cv2.imwrite("%s_01_uncropped.png" % save_name, org_roi)

            direction = get_direction(org_roi)

            # Extend the rect a bit to the left.
            # This way for big mice and such we still get some white on each side of it.
            x1 = max(x1 - 30, 0)

            # Only use the lower part of the image.
            # (This will remove some false positives when the snout is too near the edge)
            y1 = (y1 + w / 2)
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]

            if args.snout:
                template_match_ok = template_match(roi, vis_roi, snout_img, snout_contours, flipped_snout_img, flipped_snout_contours)
            else:
                if args.old_prey_method:
                    prey_match_ok = get_prey_contours(save_name, roi, vis_roi)
                else:
                    prey_match_ok = get_prey_contours_alt(save_name, roi, vis_roi)

            draw_str(vis, (20, 40), "Direction %s" % direction)

        if prey_match_ok or template_match_ok:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            match_ok = False

        draw_rects(vis, rects, color)

        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))

        if save_all:
            cv2.imwrite("%s_10_final.png" % save_name, vis)

        show_on_ok = (match_ok and args.pause_ok)
        show_on_fail = (not match_ok and args.pause_fail)

        if args.show or show_on_ok or show_on_fail:
            cv2.namedWindow("catcierge")
            cv2.moveWindow("catcierge", 0, 200)
            cv2.imshow("catcierge", vis)

        if args.save_all and args.montage:
            #montage -background black fail01.png_{01,02,03,04,05,06,07,08,09}*.png fail01_montage_mid.png
            #montage -background black fail01.png_00*.png fail01_montage_mid.png fail01.png_10_final.png fail01_montage.png
            
            call(["montage", "-background", "black"] 
                + glob.glob("%s_0[1-9]*.png" % save_name) 
                + ["%s_montage_mid.png" % save_name])

            call(["montage", "-background", "black", "-tile", "1x3", "-geometry", "+2+2"] 
                + glob.glob("%s_00*.png" % save_name) 
                + ["%s_montage_mid.png" % save_name]
                + ["%s_10_final.png" % save_name]
                + ["%s_montage.png" % save_name])

        img_count += 1

        time.sleep(args.frame_delay)

        key_delay = 5

        if match_ok:
            match_count += 1

            if show_on_ok:
                key_delay = 0
        elif show_on_fail:
            key_delay = 0

        key = 0xFF & cv2.waitKey(key_delay)
        print("Key %d" % key)

        if key == 27: # Esc
            break
        elif key == 115: # s (save)
            if (not os.path.exists(args.output)):
                os.makedirs(args.output)

            save_path = os.path.join(args.output, os.path.basename(img_path)) + "_screenshot.png"
            print("Saving image %s" % save_path)
            cv2.imwrite(save_path, vis)

    if img_count > 0:
        print("%d of %d matches ok (%d)" % (match_count, img_count, float(match_count) / img_count))

        w_avg = float(w_sum) / img_count
        h_avg = float(h_sum) / img_count
        print("(%f, %f) average size of match" % (w_avg, h_avg))

    else:
        print "No images specified ..."

    cv2.destroyAllWindows()
