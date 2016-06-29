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

using namespace cv;


enum enCommand
{
    CMD_ADD_INST =1,
    CMD_ADD_PROP,
    CMD_FIND,
    CMD_REMOVE,
    LAST_CMD
};

static const uint32_t MY_ROS_QUEUE_SIZE = 1000;

#define MIN_AREA_RECTANGLE 10000
#define MIN_PIXEL_NUMBER 2000
#define FRAME_PER_SEC 12
#define FRAME_PER_SEC_DIV_2 6


/*------ Declaration des variables globales ------*/

static int gPrevNb;  
static CvHaarClassifierCascade *cascade;
static CvMemStorage *storage;
ros::Publisher gOroChatter_pub;

static cv::Mat prevImg;
static cv::Mat currImg;

static bool initial = FALSE;

int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

bool oneTwo = FALSE;

int nbGreen;
int nbOrange;
int nbFrames;

/*---------- Declaration des fonctions -----------*/

void mvtTracking();
void detectFaces(IplImage *img);

/*------------------------------------------------*/


// Create a function which manages a clean "CTRL+C" command -> sigint command
void sigint_handler(int dummy)
{
    ROS_INFO("- detect-human-face is shutting down...\n");
    
    // Liberation de l'espace memoire
    cv::destroyWindow("Initial");
    cv::destroyWindow("final");
    cv::destroyWindow("diffImg");
    
    prevImg.release();
    currImg.release();
    
    cvReleaseHaarClassifierCascade(&cascade);
    cvReleaseMemStorage(&storage);
    
    ROS_INFO("\n\n... Bye bye !\n   -Manu\n");
    exit(EXIT_SUCCESS); // Shut down the program
}

// Explain who am i
void sayMyName()
{
    ros::Rate loop_rate(10); // Communicate slow rate

    std_msgs::String msg;
    std::stringstream ss;
    char enumCmd = 0;
    
    //ss << "add\n[kinect1 rdf:type VideoSensor, kinect1 rdfs:label \"Big brother\", kinect1 isIn Bedroom]\n#end#\n";
    enumCmd = (char)CMD_ADD_INST;
    ss << "BigBrother#"<<enumCmd<<"#kinect#VideoSensor";
    msg.data = ss.str();

    ROS_INFO("%s", msg.data.c_str());

    gOroChatter_pub.publish(msg);

    ros::spinOnce();
	  loop_rate.sleep();
	
	  ss.str("");
	  enumCmd = (char)CMD_ADD_PROP;
    ss << "BigBrother#"<< enumCmd <<"#kinect#isIn#Bedroom";
    msg.data = ss.str();

    ROS_INFO("%s", msg.data.c_str());

    gOroChatter_pub.publish(msg);

    ros::spinOnce();
	  loop_rate.sleep();
}

bool detect_color(Scalar lColor2Detect, Scalar hColor2Detect, Mat* origin)
{
	bool retBool = FALSE;
	// Image will stores HSV data
	cv::Mat* imgHSV = new cv::Mat(origin->size(), origin->type());
	
	cvtColor(*origin, *imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

	Mat imgThresholded;

	inRange(*imgHSV, lColor2Detect, hColor2Detect, imgThresholded); //Threshold the image
	//inRange(*imgHSV, Scalar(0, 128, 0), Scalar(20, 200, 255), imgThresholded); //Threshold the image ORANGE
	//inRange(*imgHSV, Scalar(70, 64, 0), Scalar(100, 140, 255), imgThresholded); //Threshold the image VERT
	
	//morphological opening (remove small objects from the foreground)
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
	dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

	//morphological closing (fill small holes in the foreground)
	dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
	
	// Count number of white pixel
	int nb = countNonZero(imgThresholded);
	
	
	imgHSV->refcount = 0;
  imgHSV->deallocate();
	
	if(nb > MIN_PIXEL_NUMBER)
	{
		retBool = TRUE;
	}
	else
	{
		retBool = FALSE;
	}
	
	return retBool;
}

// Callback 
void imgcb(const sensor_msgs::Image::ConstPtr& msg)
{
	oneTwo = ~oneTwo;
	if(oneTwo)
	{
    try
    {
      // Convert ROS images to cv::mat
      cv_bridge::CvImageConstPtr cv_ptr;
      cv_ptr = cv_bridge::toCvShare(msg);
      IplImage *img = new IplImage(cv_ptr->image);
      
      // Erosion dilation factor
      int erosion_size = 0;
      int dilation_size = 0;
      
      Mat elementE;
      Mat elementD;
      
      if(initial == FALSE)
      {
        prevImg =  cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, cv_ptr->image.type());
        currImg =  cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, cv_ptr->image.type());
        
	      nbGreen  = 0;
				nbOrange = 0;
				nbFrames = 1;
        
        initial = TRUE;
      }
      
      // Matrice used
      cv::Mat *prevImgBlur = new cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, cv_ptr->image.type());
      cv::Mat *currImgBlur = new cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, cv_ptr->image.type());
      cv::Mat *diffImg     = new cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, CV_BGR2GRAY);
      cv::Mat *prevImgGrey     = new cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, CV_BGR2GRAY);
      cv::Mat *currImgGrey     = new cv::Mat(cv_ptr->image.rows, cv_ptr->image.cols, CV_BGR2GRAY);
      
      // Blur component
      cv::Point myAnchor =cv::Point(-1,-1);
      cv::Size sizeBlur = Size(10,10);

      cv::imshow("Initial", cv_ptr->image);
      cv::waitKey(1);  // Update screen
      
      /*********************************/
      
      ///////////////////
      //               //
      //    Tracker    //
      //               //
      ///////////////////
      
      // Get current image and update previous one
      currImg.copyTo(prevImg);
      cv_ptr->image.copyTo(currImg);
      
      cvtColor(prevImg, *prevImgGrey, CV_BGR2GRAY );
      cvtColor(currImg, *currImgGrey, CV_BGR2GRAY );
      
      // Filtre "blur" sur les deux images
      blur(*prevImgGrey, *prevImgBlur, sizeBlur, myAnchor, BORDER_DEFAULT );
      blur(*currImgGrey, *currImgBlur, sizeBlur, myAnchor, BORDER_DEFAULT );
      
      // Diff between both images
      absdiff(*prevImgBlur, *currImgBlur, *diffImg);
              
      // Threshold on the diff
      threshold( *diffImg, *diffImg, 3, 255, 0 );
      
      //morphological opening (remove small objects from the foreground)
			erode(*diffImg, *diffImg, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
			dilate( *diffImg, *diffImg, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

			//morphological closing (fill small holes in the foreground)
			dilate( *diffImg, *diffImg, getStructuringElement(MORPH_ELLIPSE, Size(20, 20)) ); 
			erode(*diffImg, *diffImg, getStructuringElement(MORPH_ELLIPSE, Size(20, 20)) );
			
      cv::imshow("diffImg", *diffImg);
      
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;
      
      // Find contours
      findContours( *diffImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);//, Point(0, 0) );

      // Approximate contours to polygons + get bounding rects and circles
      vector<vector<Point> > contours_poly( contours.size() );
      vector<Rect> boundRect( contours.size() );
			
      for( int i = 0; i < contours.size(); i++ )
      {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
      }
      
      int j = 0;
      vector<Rect> lBoundRect( contours.size() );
      for( int i = 0; i < contours.size(); i++ )
      {
        if( (boundRect[i].area()) > MIN_AREA_RECTANGLE)
        {
          lBoundRect[j] = boundRect[i];
          j++;
        }
      }
			
			// If there is some movement
			if(j != 0)
			{
		    // Draw polygonal contour + bonding rects + circles
		    Mat drawing = Mat::zeros( diffImg->size(), CV_8UC3 );
		    Mat andFilter = Mat::zeros( diffImg->size(), CV_8UC3 );
		    
		    for( int i = 0; i< j; i++ )
		    {
		      //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		      rectangle( drawing, lBoundRect[i].tl(), lBoundRect[i].br(), Scalar(255,255,255), CV_FILLED, 8, 0 );
		    }
		    
		    // AND filter on the initial image
		    bitwise_and(drawing, currImg, andFilter, noArray());
		    
		    // Find color
		    bool isGreen =  detect_color(Scalar(70,64,0), Scalar(100,139,255), &andFilter);
		    bool isOrange =  detect_color(Scalar(0,128,0), Scalar(20,200,255), &andFilter);
		    
		    if(isGreen)
		    {
		    	nbGreen += 1;
		    }
		    if(isOrange)
		    {
		    	nbOrange += 1;
		    }
		    
		    // Send data

		    /// Show in a window
		    cv::imshow("final", andFilter);
		    
      	drawing.refcount = 0;
      	drawing.deallocate();
	    }
	    else
	    {
	    	// Find color
		    bool isGreen =  detect_color(Scalar(70,64,0), Scalar(100,139,255), &currImg);
		    bool isOrange =  detect_color(Scalar(0,128,0), Scalar(20,200,255), &currImg);
		    
		    if(isGreen)
		    {
		    	nbGreen += 1;
		    }
		    if(isOrange)
		    {
		    	nbOrange += 1;
		    }
	    }
      
      // Destruct and free memory
      prevImgBlur->refcount = 0;
      prevImgBlur->deallocate();
      currImgBlur->refcount = 0;
      currImgBlur->deallocate();
      diffImg->refcount = 0;
      diffImg->deallocate();
      
      currImgGrey->refcount = 0;
      currImgGrey->deallocate();
      prevImgGrey->refcount = 0;
      prevImgGrey->deallocate();
      
      if(nbFrames > FRAME_PER_SEC)
      {
      	if(nbGreen > FRAME_PER_SEC_DIV_2)
		    {
		    	if(nbOrange > FRAME_PER_SEC_DIV_2)
		    	{
		    		ROS_INFO("Green & Orange");
		    	}
		  		else
		  		{
		  			ROS_INFO("Green");
	  			}
		    }
		    else
		    {
		    	if(nbOrange > FRAME_PER_SEC_DIV_2)
		    	{
		    		ROS_INFO("Orange");
	    		}
    		}
    		
    		nbGreen  = 0;
				nbOrange = 0;
				nbFrames = 0;
      }
      else
      {
      	nbFrames +=1;
      }
        
    } catch (const cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }
}

int main(int argc, char* argv[])
{
    ROS_INFO("\n\n\t-- Launching Face detector --\n\n");
    
    gPrevNb = 0;
    
    printf("%s %d\n", argv[argc-1], argc);
    
    // Chargement du classifieur
    const char *filename = "/home/dumont/catkin_ws/src/ros-openni-example-master/src/haarcascade_frontalface_alt.xml"; 
    cascade = (CvHaarClassifierCascade*)cvLoadHaarClassifierCascade( filename, cvSize(24, 24) );
    
    // Initialisation de l'espace memoire
    storage = cvCreateMemStorage(0);

    // ROS init
    ros::init(argc, argv, "tracker");

    ros::NodeHandle nk; // Communicate with the kinect
    ros::NodeHandle nOroCl;  // Communicate wit oroclient
    
    // Override the signal interrupt from ROS
    signal(SIGINT, sigint_handler);
    ros::Subscriber sub;
    
    if(argc > 1)
    {
    	 sub = nk.subscribe(argv[argc-1], MY_ROS_QUEUE_SIZE, imgcb);
    }
    else
    {
    	sub = nk.subscribe("camera/rgb/image_color", MY_ROS_QUEUE_SIZE, imgcb);
  	}
    gOroChatter_pub = nOroCl.advertise<std_msgs::String>("oroChatter", 10);
   
	  usleep(500000); // necessary to be able to send data

    // Identify myself
    //sayMyName();

    cv::namedWindow("Initial");
    
    ros::spin();
    
    
    cv::destroyWindow("Initial");

    ROS_INFO("\n\n... Bye bye !\n   -Manu\n");

    return EXIT_SUCCESS;
}

/*------------------------------------------------*/
void detectFaces(IplImage *img)  
{  
    int i, nbFaces = 0;
    std_msgs::String msg;
    std::stringstream ss;
    
    // Face detection
    CvSeq *faces = cvHaarDetectObjects(img, cascade, storage, 1.2, 3, 0, cvSize(80,80));
    
    nbFaces = (faces?faces->total:0);
    
    // If there is more faces than previously
    if(nbFaces > gPrevNb)
    {
        std_msgs::String msg;
        std::stringstream ss;
        
        // Add new human in the ontologie
        for(i=gPrevNb; i < nbFaces; i++)
        {
            ss << "add\n[human"<< i <<" rdf:type Human, human rdfs:label \"human"<< i <<"\", kinect1 canSee human"<< i <<"]\n#end#\n";
            msg.data = ss.str();

            ROS_INFO("%s", msg.data.c_str());

            gOroChatter_pub.publish(msg);

            ros::spinOnce();
        }
        
    }
    // If there is less faces than previously
    else if (nbFaces < gPrevNb)
    {
        std_msgs::String msg;
        std::stringstream ss;
        
        // Removes the last ones
        for(i=nbFaces; i < gPrevNb; i++)
        {
            ss << "remove\n[human"<< i <<" rdf:type Human, kinect1 canSee human"<< i <<"]\n#end#\n";
            msg.data = ss.str();

            ROS_INFO("%s", msg.data.c_str());

            gOroChatter_pub.publish(msg);

            ros::spinOnce();
        }
    }
        
    // Draw rectangles over faces
    for(i=0; i<(faces?faces->total:0); i++)  
    {  
        CvRect *r = (CvRect*)cvGetSeqElem(faces, i);  
        cvRectangle(img, cvPoint(r->x, r->y), cvPoint(r->x + r->width, r->y + r->height), CV_RGB(255, 0, 0), 1, 8, 0);
    }

    // Update number of faces
    gPrevNb = (faces?faces->total:0);
    
    cvShowImage("Window-FT", img);
} 

