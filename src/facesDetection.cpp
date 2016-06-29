#include <ios>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include < opencv2/video/background_segm.hpp>  
#include <opencv2/ml/ml.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <Windows.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace System::Runtime::InteropServices;
using namespace System;
using namespace System::IO;
using namespace std;

bool compareRect(cv::Rect r1, cv::Rect r2) { return r1.width > r2.width; }

bool compareVotes(std::pair<cv::Rect, int> p1, std::pair<cv::Rect, int> p2) { return p1.second > p2.second; }

double carree(int a) {
	return a*a;
}

double Distance(int x1, int y1, int x2, int y2) {
	return sqrt(carree(y2 - y1) + carree(x2 - x1));
}

vector<std::pair<cv::Rect, int>> compute_votes_and_ROIs(vector<cv::Rect> faces)
{
	vector<std::pair<cv::Rect, int>> selected_faces;
	int dist_thr = 40; 
	vector< vector<int> > distances_matrix;

	for(int i=0; i<faces.size(); i++)
	{
		vector<int> row;
		for(int j=0; j<faces.size(); j++)
			row.push_back( Distance(faces[i].x + (faces[i].width/2), faces[i].y + (faces[i].height/2), 
									faces[j].x + (faces[j].width/2), faces[j].y + (faces[j].height/2)) );
		distances_matrix.push_back(row);
	}


	// Look for similar faces and apply Non-Maximum Suppression algorithm
	for(int i=0; i<faces.size(); i++)
	{
		int votes = 0;
		cv::Rect faceROI(0,0,0,0);
		for(int j=0; j<faces.size(); j++)
		{
			if(distances_matrix[i][j] <= dist_thr)
			{
				votes++;
				faceROI.x = faceROI.x + faces[j].x;
				faceROI.y = faceROI.y + faces[j].y;
				faceROI.width = faceROI.width + faces[j].width;
				faceROI.height = faceROI.height + faces[j].height;
			}
		}
		faceROI.x = faceROI.x / votes;
		faceROI.y = faceROI.y / votes;
		faceROI.width = faceROI.width / votes;
		faceROI.height = faceROI.height / votes;
		selected_faces.push_back(std::make_pair(faceROI, votes));
	}
	
	// Remove repeated elements 
	for(int i=selected_faces.size()-1; i>0 && i<selected_faces.size(); --i)
	{
		bool found = false;
		for(int j=0; j<i && !found; j++)
		{
			if(selected_faces[j].first == selected_faces[i].first)
				found = true;
			else if(Distance(selected_faces[j].first.x + (selected_faces[j].first.width/2), 
				selected_faces[j].first.y + (selected_faces[j].first.height/2), 
				selected_faces[i].first.x + (selected_faces[i].first.width/2),
				selected_faces[i].first.y + (selected_faces[i].first.height/2)) <= dist_thr)
				found = true;
		}
		if (found)
			selected_faces.erase(selected_faces.begin() + i);
	}

	return selected_faces;
}

vector<std::pair<cv::Rect, int>> remove_masked_faces(vector<std::pair<cv::Rect, int>> detected_faces, cv::Mat mask)
{
	int masked_pixels_percent = 10;
	for(int i=detected_faces.size()-1; i>=0; --i)
	{
		int masked_pixels = 0;
		cv::Rect faceROI = detected_faces[i].first;
		int total_pixels = faceROI.width * faceROI.height;
		for(int j=faceROI.x; j<faceROI.x+faceROI.width; j++)
			for(int k=faceROI.y; k<faceROI.y+faceROI.height; k++)
			{
				if( mask.at<uchar>(k,j) == 0)
					masked_pixels++;
			}
		double percent_masked_pixels =  masked_pixels*100 / total_pixels;
		if(percent_masked_pixels > masked_pixels_percent)
			detected_faces.erase(detected_faces.begin() + i);
	}

	return detected_faces;
}

cv::Mat build_mask_from_string(string line, bool inverted)
{
	cv::Mat mask = cv::Mat::zeros(480, 640, CV_8U);
	int x, y, width, heigth;
	string aux_str, x_str, y_str, w_str, h_str;

	// Read ROI elements
	aux_str = line.substr(line.find_first_of(",")+1);
	x_str = aux_str.substr(0, aux_str.find_first_of(","));
	aux_str = aux_str.substr(aux_str.find_first_of(",")+1);
	y_str = aux_str.substr(0, aux_str.find_first_of(","));
	aux_str = aux_str.substr(aux_str.find_first_of(",")+1);
	w_str = aux_str.substr(0, aux_str.find_first_of(","));
	aux_str = aux_str.substr(aux_str.find_first_of(",")+1);
	h_str = aux_str.substr(0, aux_str.find_first_of(","));
	
	// build mask matrix
	cv::Rect ROI(atoi(x_str.c_str()), atoi(y_str.c_str()), atoi(w_str.c_str()), atoi(h_str.c_str()));
	for(int j=ROI.x; j<ROI.x+ROI.width; j++)
		for(int k=ROI.y; k<ROI.y+ROI.height; k++)
			mask.at<uchar>(k,j) = 255;

	if(inverted)
		mask = abs(mask - 255);

	return mask;
}


int main()
{
	// LOAD OpenCV FACE DETECTOR MODELS (frontal and profile)
	// Frontal face cascades
	string faceDetectionModel("../models/haarcascade_frontalface_alt2.xml");
	string faceDetectionModel2("../models/haarcascade_frontalface_alt_tree.xml");
	// Profile face cascade
	string profilefaceDetectionModel("../models/haarcascade_profileface.xml");
	cv::CascadeClassifier face_cascade;
	if( !face_cascade.load( faceDetectionModel ) )
	{ 
		cerr << "Error loading face detection model." << endl;
		return -1; 
	}
	cv::CascadeClassifier face_cascade2;
	if( !face_cascade2.load( faceDetectionModel2 ) )
	{ 
		cerr << "Error loading face detection model 2." << endl;
		return -1; 
	}
	cv::CascadeClassifier profile_face_cascade;
	if( !profile_face_cascade.load( profilefaceDetectionModel ) )
	{ 
		cerr << "Error loading profile face detection model." << endl;
		return -1; 
	}

	// Global variables
	//string winname("QMUL faces detector");
	//cv::namedWindow(winname);
	string mask_path = "../models/masks/background/mask.png"; // GENERIC MASK
	int n_sessions = 12;

	for (int n=10; n<=n_sessions; n++)
	{
		string session_folder_str = "C:/Users/Isabelle/Documents/QMUL database/S" + to_string(n) + "/";
		String^ session_folder = gcnew String(session_folder_str.c_str());
		array<String^>^ clips = Directory::GetDirectories(session_folder);

		for (int c=0; c<clips->Length; c++)
		{
		  // TODO Get image
			// Obtain clip name
			String^ currentclip = clips[c];
			IntPtr currentclip_ptrToNativeString = Marshal::StringToHGlobalAnsi(currentclip);
			char* currentclipStr = static_cast<char*>(currentclip_ptrToNativeString.ToPointer());
			std::string clipName(currentclipStr);
			int ini = clipName.find_last_of("/");
			clipName = clipName.substr(ini+1);

			cout << "Computing clip" + clipName << endl;

			// PATHS
			string videoPath = "../videos/S" + to_string(n) + "/Face_" + clipName + ".avi"; // OUTPUT VIDEO PATH
			string resultFolderName = "../results/FaceROIDetectionResults/S" + to_string(n) + "/Face_" + clipName; // RESULT FOLDER
			string computed_mask_path = "../models/masks/background/" + clipName + ".png"; // COMPUTED MASK
			string robot_mask_path = "C:/Users/Isabelle/Documents/FacesDetector_QMUL/results/RobotDetectionResults/CSV/S" + to_string(n) + "/RobotDetection_" + clipName + ".csv"; // ROBOT ROI
			string actors_mask_path = "C:/Users/Isabelle/Documents/FacesDetector_QMUL/results/ROIExtractionResults/CSV/S" + to_string(n) + "/ROIExtraction_" + clipName + ".csv"; // ACTORS ROI

			// Setup output video
			cv::VideoWriter output_cap(videoPath,
					   CV_FOURCC('W','M','V','1'),
					   10,
					   cv::Size (640,480),
					   true);

			// Get all images paths from the folder
			array<String^>^ frames = Directory::GetFiles(currentclip);

			// CREATE MASK FOR THE VIDEO
			cv::Mat frame;
			cv::Mat mask = cv::imread(mask_path,CV_LOAD_IMAGE_GRAYSCALE); // my background mask
			cv::Mat video_mask; //fg mask generated by MOG method (one per clip)
			cv::Mat robot_mask; // robot region mask (one per frame)
			cv::Mat actors_mask; // actors region mask (one per clip)

			// Obtain actors mask
			std::ifstream actors_mask_file(actors_mask_path);
			string line;
			if( actors_mask_file.good() ){
				getline(actors_mask_file, line); // ignore first line	
				getline(actors_mask_file, line); // read line with ROI data
				actors_mask = build_mask_from_string(line, false);
			}
			actors_mask_file.close();
			//imshow("actors mask", actors_mask);

			// Compute background mask (check if already exists)
			ifstream f(computed_mask_path.c_str());
			if (f.good()) 
			{
				cout << "Loading mask..." << endl;
				video_mask = cv::imread(computed_mask_path,CV_LOAD_IMAGE_GRAYSCALE);
				mask = mask.mul(video_mask);
				if(actors_mask.rows != 0 )
					mask = mask.mul(actors_mask);
				cout << "Mask loaded!" << endl;
			}
			else
			{
				cv::Ptr< cv::BackgroundSubtractor> pMOG; //MOG Background subtractor  
				pMOG = new cv::BackgroundSubtractorMOG(frames->Length, 5, 0.7, 0.1);  
				int erosion_size = 30;   
				cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
								  cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
								  cv::Point(erosion_size, erosion_size) );
				cout << "Building mask..." << endl;
				for (int i=0; i<frames->Length; i++)
				{	
					// Get current frame
					String^ currentfile = frames[i];
					IntPtr ptrToNativeString = Marshal::StringToHGlobalAnsi(currentfile);
					char* nativeString = static_cast<char*>(ptrToNativeString.ToPointer());
					frame = cv::imread(nativeString);
					if (frame.rows == 0 || frame.cols == 0)
						break;
					// Compute MOG background substractors
					pMOG->operator()(frame, video_mask); 
					dilate( video_mask, video_mask, element );
				}
				mask = mask.mul(video_mask);
				if(actors_mask.rows != 0 )
					mask = mask.mul(actors_mask);
				// save computed video mask
				vector<int> param; 
				param.push_back(CV_IMWRITE_PNG_COMPRESSION);
				param.push_back(0);
				cv::imwrite(computed_mask_path, video_mask, param); // we save the computed video_mask only
				cout << "Mask built!" << endl;
			}

			// Create folder
			if (! (CreateDirectoryA(resultFolderName.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError()))
			{
				cerr << "Error creating result folder for clip " + resultFolderName << endl;
				return -1; 
			}

			// Open the robot mask reader
			std::ifstream robotROI_file(robot_mask_path);
			string robotROI_line;
			if( robotROI_file.good() )
				getline(robotROI_file, robotROI_line); // ignore first line	

			// GO FOR DETECTION! ----------------------------------------------------------------
	
			for (int i=0; i<frames->Length; i++)
			{
				cv::Mat masked_frame;
				cv::Mat frame_mask;

				String^ currentfile = frames[i];
				IntPtr ptrToNativeString = Marshal::StringToHGlobalAnsi(currentfile);
				char* nativeString = static_cast<char*>(ptrToNativeString.ToPointer());
				frame = cv::imread(nativeString);
				if (frame.rows == 0 || frame.cols == 0)
					break;

				// Build robot mask 
				if( robotROI_file.good() ){
					getline(robotROI_file, robotROI_line);
					robot_mask = build_mask_from_string(robotROI_line, true);
					frame_mask = mask.mul(robot_mask);
					//imshow("only robot", robot_mask);
					//imshow("frame mask (background + robot + actors)", frame_mask);
				}else
					frame_mask = mask;

				// Create the csv file where faces ROIs and scores will be saved
				string str_nativeString(nativeString);
				string frame_name = str_nativeString.substr(str_nativeString.find("kinect2_color_"));
				frame_name.replace(frame_name.find(".png"), 4, ".csv");
				ofstream faces_frame_file;
				faces_frame_file.open(resultFolderName + "/" + frame_name);

				// Detect faces with all the haarcascades
				vector<cv::Rect> faces;
				vector<cv::Rect> faces2;
				vector<cv::Rect> profile_faces;
				face_cascade.detectMultiScale(frame, faces, 1.03, 0, 0, cv::Size(35, 35), cv::Size(55, 55)); // ad-hoc params for QMUL db where faces are 40x40 approx.
				face_cascade2.detectMultiScale(frame, faces2, 1.03, 0, 0, cv::Size(35, 35), cv::Size(55, 55)); 
				profile_face_cascade.detectMultiScale(frame, profile_faces, 1.03, 0, 0, cv::Size(35, 35), cv::Size(55, 55));
				faces.insert(faces.end(), faces2.begin(), faces2.end());
				faces.insert(faces.end(), profile_faces.begin(), profile_faces.end());

				// Keep only non-repeated faces 
				vector<std::pair<cv::Rect, int>> detected_faces;
		
				if(faces.size() > 1)
				{
					detected_faces = compute_votes_and_ROIs(faces);
					detected_faces = remove_masked_faces(detected_faces, frame_mask);
					sort(detected_faces.begin(), detected_faces.end(), compareVotes); // Order by number of votes
				}
				else if(faces.size() == 1)
				{
					detected_faces.push_back(std::make_pair(faces[0], 1));
					detected_faces = remove_masked_faces(detected_faces, frame_mask);
				}
		
				// Write the csv file and close it
				for(int i=0; i<detected_faces.size(); i++)
				{
					faces_frame_file << to_string(detected_faces[i].first.x) + " ";
					faces_frame_file << to_string(detected_faces[i].first.y) + " ";
					faces_frame_file << to_string(detected_faces[i].first.width) + " ";
					faces_frame_file << to_string(detected_faces[i].first.height) + " ";
					if(i==detected_faces.size()-1)
						faces_frame_file << to_string(detected_faces[i].second);
					else
						faces_frame_file << to_string(detected_faces[i].second) + "\n";
				}
				faces_frame_file.close();
		
				// Paint all detected faces (in white)
				frame.copyTo(masked_frame, frame_mask);
				for(int j=0; j<faces.size(); j++)
					cv::rectangle(masked_frame, faces[j], cv::Scalar(255,255,255), 1, 8, 0);

				// Paint final merged faces (in green)
				for(int j=2; j<detected_faces.size(); j++)
				{
					cv::rectangle(masked_frame, detected_faces[j].first, cv::Scalar(0,255,0), 2, 8, 0);
					cv::putText(masked_frame, to_string(detected_faces[j].second), cv::Point(detected_faces[j].first.x-5,detected_faces[j].first.y-5), 
						cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,255,0), 1, CV_AA);
				}
				// Paint two most voted faces (in blue)
				if(detected_faces.size() > 0)
				{
					cv::rectangle(masked_frame, detected_faces[0].first, cv::Scalar(255,0,0), 2, 8, 0);
					cv::putText(masked_frame, to_string(detected_faces[0].second), cv::Point(detected_faces[0].first.x-5,detected_faces[0].first.y-5), 
								cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255,0,0), 1, CV_AA);
				}
				if(detected_faces.size() > 1)
				{
					cv::rectangle(masked_frame, detected_faces[1].first, cv::Scalar(255,0,0), 2, 8, 0);
					cv::putText(masked_frame, to_string(detected_faces[1].second), cv::Point(detected_faces[1].first.x-5,detected_faces[1].first.y-5), 
								cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255,0,0), 1, CV_AA);
				}

				//cv::imshow(winname,masked_frame);
				output_cap.write(masked_frame);

				//cv::waitKey(5);
			}

			robotROI_file.close();
			output_cap.release();
		
		} // Clip
	} // Session

	return 0;

}
