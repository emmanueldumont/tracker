#include <iostream>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <stdio.h>

#include <signal.h>         // Handling ctrl+C signal to close everything properly

// Manages message with oroclient
#include "std_msgs/String.h"
#include <sstream>

using namespace std;
using namespace cv;

static const uint32_t MY_ROS_QUEUE_SIZE = 1000;

int iLowH = 0;
int iHighH = 255;

int iLowS = 0; 
int iHighS = 255;

int iLowV = 0;
int iHighV = 255;

void imgcb(const sensor_msgs::Image::ConstPtr& msg)
{
  try {
  
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(msg);

    cv::imshow("Control", cv_ptr->image);

		// Read the frame of the video
		cv::Mat* imgOriginal = new cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, cv_ptr->image.type());
		
    cv_ptr->image.copyTo(*imgOriginal);
		//imshow("Original", *imgOriginal); //show the original image

		cv::Mat* imgHSV = new cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, cv_ptr->image.type());
		
		cvtColor(*imgOriginal, *imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		Mat imgThresholded;

		inRange(*imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
		//inRange(*imgHSV, Scalar(0, 128, 0), Scalar(19, 199, 255), imgThresholded); //Threshold the image ORANGE
		//inRange(*imgHSV, Scalar(60, 64, 0), Scalar(90, 139, 255), imgThresholded); //Threshold the image VERT
		
		//morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

		//morphological closing (fill small holes in the foreground)
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		
		imshow("Thresholded Image", imgThresholded); //show the thresholded image
		//imshow("Original", *imgOriginal); //show the original image
		
		cv::waitKey(1);
		
		imgHSV->refcount = 0;
    imgHSV->deallocate();
    
		imgOriginal->refcount = 0;
    imgOriginal->deallocate();
	
	} catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
  }

}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "foo");

    std::cout << "Oh hai there!" << std::endl;
    
    

    ros::NodeHandle nh;
    // ros::Subscriber sub = nh.subscribe("camera/rgb/image_raw", MY_ROS_QUEUE_SIZE, imgcb);
    ros::Subscriber sub = nh.subscribe("camera/rgb/image_color", MY_ROS_QUEUE_SIZE, imgcb);
    
    
    namedWindow("Control"); //create a window called "Control"
    //Create trackbars in "Control" window
		cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
		cvCreateTrackbar("HighH", "Control", &iHighH, 179);

		cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
		cvCreateTrackbar("HighS", "Control", &iHighS, 255);

		cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
		cvCreateTrackbar("HighV", "Control", &iHighV, 255);

    //cv::namedWindow("foo");
    ros::spin();
    cv::destroyWindow("foo");

    std::cout << "byebye my friend" << std::endl;

    return 0;
}
