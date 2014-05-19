catcierge-samples
=================

Catcierge project samples for Haar cascade classifier

Description
-----------

The positive images marked in positives.txt were marked using: [imageclipper](https://github.com/JoakimSoderberg/imageclipper)

This list of positives where then fed to the Opencv createsamples program:
```bash
$ /path/to/opencv_createsamples -vec catcierge.vec -info positives.txt -bg negatives.txt -num 2588
```

It turns out the [manual](http://docs.opencv.org/doc/user_guide/ug_traincascade.html) for the [OpenCV](http://www.opencv.org) traincascade program isn't very verbose on what parameters are exactly.

This [question on stackoverflow](http://stackoverflow.com/questions/10863560/haar-training-opencv-assertion-failed) gives the answer on what the `-numPos` setting actually can be set to.

Just because your `.vec` file contains 2588 positive samples, doesn't mean you set this to 2588. Instead you need to use this formula:

```
vec-file has to contain >= (numPos + (numStages-1) * (1 - minHitRate) * numPos) + numNeg
```
Which in my case is:

```
2588 >= (x + (20-1)*(1-0.999)*x) + 2279
x <= ~302
```

So to train the haar cascade:

```bash
$ /path/to/opencv_traincascade -vec catcierge.vec -bg negatives.txt -numPos 302 -numNeg 2279 -minHitRate 0.999 -numStages 20 -precalcIdxBufSize 6144 -precalcValBufSize 3072 -w 24 -h 24 -bg negatives.txt -data cascade
```
Note! Make sure you don't specify more memory than you have available in the above commandline. My computer has 16GB.




