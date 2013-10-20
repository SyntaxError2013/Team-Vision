#include <opencv2/core/core.hpp>
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <cmath>
#include <math.h>

using namespace cv;
using namespace std;

//Declaration of Global Variables
int equalThreshold=1e-7,size=16,r=40;

Mat src;
Mat src_gray, threshold_output,blue,blue_2,im,face_ROI,im2,im3;
Vec3b k;
Vec3b p;
BackgroundSubtractorMOG bgs;	
 
float maxAr=0.0;
int maxi=0;
int thresh = 100;
int max_thresh = 255;
double sigmaX=0;
double sigmaY=0;
int x1,x2,y11,y2;
RNG rng(12345);

 String face_cascade_name = "C:/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "C:/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier cascade;

void thresh_callback(int, void* );
  
//Used to compute angle 
double angle(vector<Point>& contour,int pt,int r)
{
	int size = contour.size();
    Point p0=(pt>0)?contour[pt%size]:contour[size-1+pt];
    Point p1=contour[(pt+r)%size];
    Point p2=(pt>r)?contour[pt-r]:contour[size-1-r];

        double ux=p0.x-p1.x;
        double uy=p0.y-p1.y;
        double vx=p0.x-p2.x;
        double vy=p0.y-p2.y;
        return (ux*vx + uy*vy)/sqrt((ux*ux + uy*uy)*(vx*vx + vy*vy));
}

void Mouse_Move(DWORD dx,DWORD dy)
{
	DWORD event=0;
	event = MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_MOVE;
	mouse_event(event,((dx*65535/1366)), ((dy*65535/768)), 0, 0);
} 

//Used to compute coordinates of finger tips
signed int rotation(std::vector<cv::Point>& contour, int pt, int r)
{
        int size = contour.size();
        Point p0=(pt>0)?contour[pt%size]:contour[size-1+pt];
        Point p1=contour[(pt+r)%size];
        Point p2=(pt>r)?contour[pt-r]:contour[size-1-r];

        double ux=p0.x-p1.x;
        double uy=p0.y-p1.y;
        double vx=p0.x-p2.x;
        double vy=p0.y-p2.y;
        
		return (ux*vy - vx*uy);
}

bool isEqual(double a, double b)
{
        return fabs(a - b) <= equalThreshold;
}


int main()
{
	
	if( !cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	CvCapture* capture;       

    capture = cvCaptureFromCAM(0);
    if(!capture)
	{
		printf("Capture failure\n");
        return -1;
    }

	for(int u=0;u<50;u++)
	im =cvQueryFrame(capture);
	
	im2=im.clone();
	im3=im.clone();
	
	// imshow("im2",im2);
	vector<Rect> faces;

	cascade.detectMultiScale( im, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	for( int i = 0; i < faces.size(); i++ )
	{
		Point tl( faces[i].x, faces[i].y );
		Point br( faces[i].x + faces[i].width, faces[i].y + faces[i].height );
    	rectangle(im, tl, br, Scalar(255,0,255),1,8,0);
		face_ROI = im( faces[i]);
		cvtColor(face_ROI, face_ROI,CV_BGR2HSV);
		k=face_ROI.at<Vec3b>(faces[i].width/4, faces[i].height/2);
	}
   
	//imwrite("D:/hand.jpg", face_ROI);
	  
	while(1)
	{
		src =cvQueryFrame(capture) ;
		cvtColor( src, src_gray, CV_BGR2HSV ) ;
		
		//imshow( "source", src );
		thresh_callback( 0, 0 );
		int c = cvWaitKey(10);
        
		if((char)c==27 ) break;  
	}
	cvDestroyAllWindows() ;
	cvReleaseCapture(&capture);
	return(0);
}

void thresh_callback(int, void* )
{
	vector<vector<Point> > contours;
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy;
	vector<Vec4i> hierarchy2;

	threshold_output=cvCreateImage(src_gray.size(),IPL_DEPTH_8U, 1);
	inRange(src_gray, Scalar(k[0]-20,k[1]-75,k[2]-80), Scalar(k[0]+20,k[1]+75,k[2]+80),threshold_output);

    Mat element;
	erode(threshold_output,threshold_output, element,Point(),1);
	dilate(threshold_output, threshold_output, element,Point(),3);
	Mat middle=threshold_output.clone();
   
	for ( int i = 1; i <9 ; i=i+2 )
	{
		GaussianBlur( threshold_output, threshold_output,Size(3,3),0,0,4 );
	}
  
	findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	p[0]=255;
	p[1]=0;
	p[2]=255;
    vector<vector<Point> > contours_poly2( contours2.size() );
	vector<Point2f>center2( contours2.size() );
	vector<float>radius2( contours2.size() );
	vector<vector<Point> >hull( contours.size() );
	vector<int> hullIndices;
	
	Point center;
	vector<vector<Vec4i> > defects(contours.size());
	vector<Point> fingers;
 
	int c=0;
	for( int i = 0; i < contours.size(); i++ )
	{  
		float area=contourArea(contours[i]);
		if(area>=11000 )
		{
		  cout<<area<<endl;
		  convexHull( Mat(contours[i]), hull[c], false ); 

		  Moments m=moments(contours[i]);
		  center.x=m.m10/m.m00;
		  center.y=m.m01/m.m00;

		  for(int j = 0; j < contours[i].size(); j+= step )
          {
             double cos0 = angle (contours[i], j,r);
 
			 if ((cos0 > 0.5)&&(j+step < contours[i].size()))
             {
                 double cos1 = angle (contours[i], j - step, r);
                 double cos2 = angle (contours[i], j + step, r);
                 double maxCos = max(max(cos0, cos1), cos2);
                 bool equal = isEqual(maxCos , cos0);
                 signed int z = rotation (contours[i], j, r);
                 if (equal == 1 && z<0)
                 {
					fingers.push_back(contours[i][j]);
                 }
			 }    
		    }
		}  
		 
	 for(int j=0;j<fingers.size();j++)
	 circle( src, fingers[j], 3, Scalar(255,0,0), 2, 8, 0 );
	 circle( src, center, 5, Scalar(255,0,255), 2, 8, 0 );
	 c++;
	}

  
   for( int i = 0; i< c; i++ )
   {
	 drawContours( src, hull, i, Scalar(255,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
	}
   
	if(fingers.size()>3)
	Mouse_Move(((fingers[0].x)*1366)/640,((fingers[0].y)*768)/480);
	Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
	imshow( "Hull demo", drawing );
	imshow("middle", threshold_output);
  	imshow( "Tracking", src );

}