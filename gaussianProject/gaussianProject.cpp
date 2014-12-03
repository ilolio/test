// gaussianProject.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>

#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

typedef struct _FFormation{
	cv::Point2d center;
	double radius;
} FFormation;

double gaussian(const double num, const double average, const double miu2){

	return (1/sqrt(2*M_PI*miu2)) * exp( (-pow(num-average,2)) / (2*miu2) );
}

double estimateWithinFformation(const FFormation& fformation,const Point2d,)

int _tmain(int argc, _TCHAR* argv[])
{


	printf("num0:%f,num1:%f,num2:%f",gaussian(0,0,1),gaussian(1,0,1),gaussian(2,0,1));


	return 0;
}

