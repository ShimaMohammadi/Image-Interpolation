#include "opencv2\opencv.hpp";
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <stdio.h>


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <conio.h>
#include <iostream>

using namespace std;
using namespace cv;


Mat img;

void nearest_neighbour_interpolation(const Mat& image, Mat& result, double scaleFactor)
{
	if (scaleFactor < 1.0000)
		return;

	int width = image.cols*scaleFactor;
	int height = image.rows*scaleFactor;
	result = Mat(height, width, image.type());

	int x_ratio = (int)((image.cols << 16) / width) + 1;
	int y_ratio = (int)((image.rows << 16) / height) + 1;

	for (int c = 0; c < width; c++) {
		for (int r = 0; r < height; r++) {

			int imageC = ((c*x_ratio) >> 16);
			int imageR = ((r*y_ratio) >> 16);

			result.at<uchar>(Point(c, r)) = image.at<uchar>(Point(imageC, imageR));
		}
	}
}

void mouse_click(int event, int x, int y, int flags, void *param)
{
	Mat img2; img.copyTo(img2);
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
		Point center(x, y);
		circle(img2, center, 50, Scalar(0, 255, 0), 1, 8, 0);
		imshow("Image", img2);

		Rect r(center.x - 50, center.y - 50, 50 * 2, 50 * 2);
		Rect imgBounds(0, 0, img2.cols, img2.rows);
		r = r & imgBounds;
		Mat roi(img, r);
		Mat mask(roi.size(), roi.type(), Scalar::all(0));
		circle(mask, Point(50, 50), 50, Scalar::all(255), -1);
		Mat cropped = roi & mask;
		Mat cropped2;
		nearest_neighbour_interpolation(cropped, cropped2, 2.5);
		imshow("temp", cropped2);
	}
	break;
	case CV_EVENT_RBUTTONDOWN:
	{
		Point center(x, y);
		circle(img2, center, 50, Scalar(0, 255, 0), 1, 8, 0);
		imshow("Image", img2);

		Rect r(center.x - 50, center.y - 50, 50 * 2, 50 * 2);
		Rect imgBounds(0, 0, img2.cols, img2.rows);
		r = r & imgBounds;
		Mat roi(img, r);
		Mat mask(roi.size(), roi.type(), Scalar::all(0));
		circle(mask, Point(50, 50), 50, Scalar::all(255), -1);
		Mat cropped = roi & mask;
		Mat cropped2;


		double h_rate = 2.5;
		double w_rate = 2.5;


		int h = cropped.rows*h_rate;
		int w = cropped.cols*w_rate;

		Mat result_img(h, w, CV_8UC1, Scalar(0));

		for (int y = 0; y < result_img.rows - 1; y++) {
			for (int x = 0; x < result_img.cols - 1; x++) {
				int px = (int)(x / w_rate);
				int py = (int)(y / h_rate);

				if (px >= cropped.cols - 1 || py >= cropped.rows - 1) break;

				double fx1 = (double)x / (double)w_rate - (double)px;
				double fx2 = 1 - fx1;
				double fy1 = (double)y / (double)h_rate - (double)py;
				double fy2 = 1 - fy1;

				double w1 = fx2*fy2;
				double w2 = fx1*fy2;
				double w3 = fx2*fy1;
				double w4 = fx1*fy1;

				uchar p1 = cropped.at<uchar>(py, px);
				uchar p2 = cropped.at<uchar>(py, px + 1);
				uchar p3 = cropped.at<uchar>(py + 1, px);
				uchar p4 = cropped.at<uchar>(py + 1, px + 1);
				result_img.at<uchar>(y, x) = w1*p1 + w2*p2 + w3*p3 + w4*p4;
			}
		}
		imshow("temp", result_img);
	}

	break;


	}

}


int main(int argc, char** argv)
{
	img = imread("lenna.png", CV_LOAD_IMAGE_GRAYSCALE);

	
	imshow("Image", img);


	cvSetMouseCallback("Image", mouse_click, 0);

	while (cv::waitKey(1) != 27);

	return 0;
}




