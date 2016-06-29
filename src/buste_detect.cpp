#include <iostream>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <stdio.h>

#include <signal.h>         // Handling ctrl+C signal to close everything properly

// Manages message with oroclient
#include "std_msgs/String.h"
#include <sstream>


enum enCommand
{
    CMD_ADD_INST =1,
    CMD_ADD_PROP,
    CMD_FIND,
    CMD_REMOVE,
    LAST_CMD
};

static const uint32_t MY_ROS_QUEUE_SIZE = 1000;


/*------ Declaration des variables globales ------*/

static CvHaarClassifierCascade *cascade_face;
static CvHaarClassifierCascade *cascade_upBody;
static CvMemStorage *storage_face;
static CvMemStorage *storage_upBody;
ros::Publisher gOroChatter_pub;


/*---------- Declaration des fonctions -----------*/

void detectFaces(IplImage *img);
void detect_upperbody(IplImage *img);

/*------------------------------------------------*/


// Create a function which manages a clean "CTRL+C" command -> sigint command
void sigint_handler(int dummy)
{
    ROS_INFO("- detect-human-face is shutting down...\n");
    
    // Liberation de l'espace memoire
    cvDestroyWindow("Window-Face");
    cvDestroyWindow("Window-UpBody");
    cv::destroyWindow("Initial");
    
    cvReleaseHaarClassifierCascade(&cascade_face);
    cvReleaseMemStorage(&storage_face);
    
    cvReleaseHaarClassifierCascade(&cascade_upBody);
    cvReleaseMemStorage(&storage_upBody);
    
    ROS_INFO("\n\n... Bye bye !\n   -Manu\n");
    exit(EXIT_SUCCESS); // Shut down the program
}


// Callback 
void imgcb(const sensor_msgs::Image::ConstPtr& msg)
{
    try
    {
        // Convert ROS images to cv::mat
        cv_bridge::CvImageConstPtr cv_ptr;
        cv_ptr = cv_bridge::toCvShare(msg);
        IplImage *img = new IplImage(cv_ptr->image); 

        cv::imshow("Initial", cv_ptr->image);
        cv::waitKey(1);  // Update screen
        
        /*********************************/
                
        // Creation de 2 fenetres
        cvNamedWindow("Window-Face", 1);
        cvNamedWindow("Window-UpBody");
        
        // Boucle de traitement 
        detectFaces(img);
        detect_upperbody(img);
        
        /*********************************/
        
        
    } catch (const cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char* argv[])
{
    ROS_INFO("\n\n\t-- Launching Face detector --\n\n");
        
    // Chargement du classifieur
    const char *filename_face = "/home/dumont/catkin_ws/src/tracker/src/haarcascade_frontalface_alt.xml";
    const char *filename_upBody = "/home/dumont/catkin_ws/src/tracker/src/haarcascade_upperbody.xml";
    cascade_face = (CvHaarClassifierCascade*)cvLoadHaarClassifierCascade( filename_face, cvSize(24, 24) );
    cascade_upBody = (CvHaarClassifierCascade*)cvLoadHaarClassifierCascade( filename_upBody, cvSize(24, 24) );
    
    // Initialisation de l'espace memoire
    storage_face = cvCreateMemStorage(0);
		storage_upBody = cvCreateMemStorage(0);

    // ROS init
    ros::init(argc, argv, "detect_face_rgb");

    ros::NodeHandle nk; // Communicate with the kinect
    ros::NodeHandle nOroCl;  // Communicate wit oroclient
    
    // Override the signal interrupt from ROS
    signal(SIGINT, sigint_handler);
    
    ros::Subscriber sub = nk.subscribe("camera/rgb/image_color", MY_ROS_QUEUE_SIZE, imgcb);
    gOroChatter_pub = nOroCl.advertise<std_msgs::String>("oroChatter", 10);
   
	  usleep(500000); // necessary to be able to send data

    cv::namedWindow("Initial");
    
    ros::spin();
    
    return EXIT_SUCCESS;
}

/*------------------------------------------------*/

void detect_upperbody(IplImage *img)
{  
    int i, nbUpBody = 0;
    std_msgs::String msg;
    std::stringstream ss;
    
    // Face detection
    CvSeq *upBody = cvHaarDetectObjects(img, cascade_upBody, storage_upBody, 1.2, 5, 0, cvSize(80,80));
    
    nbUpBody = (upBody?upBody->total:0);
    
    
    // Draw rectangles over faces
    for(i=0; i<nbUpBody; i++)  
    {  
        CvRect *r = (CvRect*)cvGetSeqElem(upBody, i);  
        cvRectangle(img, cvPoint(r->x, r->y), cvPoint(r->x + r->width, r->y + r->height), CV_RGB(255, 0, 0), 1, 8, 0);
    }

    
    cvShowImage("Window-UpBody", img);
} 



void detectFaces(IplImage *img)  
{  
    int i, nbFaces = 0;
    std_msgs::String msg;
    std::stringstream ss;
    
    // Face detection
    CvSeq *faces = cvHaarDetectObjects(img, cascade_face, storage_face, 1.2, 3, 0, cvSize(80,80));
    
    nbFaces = (faces?faces->total:0);
    
    
    // Draw rectangles over faces
    for(i=0; i<nbFaces; i++)  
    {  
        CvRect *r = (CvRect*)cvGetSeqElem(faces, i);  
        cvRectangle(img, cvPoint(r->x, r->y), cvPoint(r->x + r->width, r->y + r->height), CV_RGB(255, 0, 0), 1, 8, 0);
    }
    
    cvShowImage("Window-Face", img);
} 

