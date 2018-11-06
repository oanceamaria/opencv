// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace cv;
using namespace cv::dnn;
using namespace std;

string CLASSES[] = { "background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor" };


int detectareSticlaMobileNetSSD()
{
	String modelTxt = "MobileNetSSD_deploy.prototxt";
	String modelBinary = "MobileNetSSD_deploy.caffemodel";

	Net net = dnn::readNetFromCaffe(modelTxt, modelBinary);
	if (net.empty())
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxt << std::endl;
		std::cerr << "caffemodel: " << modelBinary << std::endl;
		exit(-1);
	}

	VideoCapture cap;

	cap.open(0);

	while(1)
	{
		Mat frame;
		cap >> frame;

		if (frame.empty()) break;

		if (waitKey(10) == 27) break; 

		Mat imageOfFrame = frame;

		Mat imageForProcessing;
		resize(imageOfFrame, imageForProcessing, Size(300, 300));
		Mat inputBlob = blobFromImage(imageForProcessing, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);

		net.setInput(inputBlob, "data");
		Mat detection = net.forward("detection_out");
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		float confidenceThreshold = 0.8;

		for (int inexOfDetection = 0; inexOfDetection < detectionMat.rows; inexOfDetection++)
		{
			float confidence = detectionMat.at<float>(inexOfDetection, 2);

			if (confidence > confidenceThreshold)
			{
				int index = static_cast<int>(detectionMat.at<float>(inexOfDetection, 1));
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(inexOfDetection, 3) * imageOfFrame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(inexOfDetection, 4) * imageOfFrame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(inexOfDetection, 5) * imageOfFrame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(inexOfDetection, 6) * imageOfFrame.rows);

				if (CLASSES[index] == "bottle" && confidence > 0.5) {
					Rect object((int)xLeftBottom, (int)yLeftBottom,
						(int)(xRightTop - xLeftBottom),
						(int)(yRightTop - yLeftBottom));

					rectangle(imageOfFrame, object, Scalar(0, 255, 0), 2);

					cout << CLASSES[index] << ": " << confidence << endl;

					String label = CLASSES[index] + ": " + to_string(confidence);
					int baseLine = 0;
					Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
					putText(imageOfFrame, label, Point(xLeftBottom, yLeftBottom),
						FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
				}
			}
		}
		imshow("Bottle", frame);
	}
	return 0;
}

