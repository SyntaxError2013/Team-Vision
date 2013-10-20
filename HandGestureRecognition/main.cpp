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
int equalThreshold=1e-7;
  
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
	cout<<"Team- Vision"<<endl;
	return(0);
}