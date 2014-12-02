// ConsoleApplication1.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>

int _tmain(int argc, _TCHAR* argv[]){

	const int NUM_POS=10;//ポジティブデータ数.　Pｐｓ
	const int NUM_NEG=10;//ネガティブ数　NEG
	const int NUM_ALL=NUM_POS+NUM_NEG;//全データ個数
	const int USE_FEATURE_COUNT = 2;//1つのデータの次元数

	const int CHECK_DATA_COUNT=5;

	cv::Mat trainingData;//ここにデータを入れておきます．
	cv::Mat trainingLabels;//+１か-1かラベルをいれます．

	trainingData.create(NUM_ALL,USE_FEATURE_COUNT,CV_32FC1);//全データ数，次元数のMat型を用意
	trainingLabels.create(NUM_ALL,1,CV_32FC1);//ラベルだけなので，次元数は1でいいです．

	std::vector<std::vector<const double>> data(NUM_ALL);
	for( int i=0; i<NUM_ALL; i++ ){
		data[i].resize(USE_FEATURE_COUNT);
	}

	const double _data[NUM_ALL][USE_FEATURE_COUNT]={
		{0.83,0.23},
		{1.01,0.67},
		{0.89,0.24},
		{0.96,0.52},
		{1.28,0.11},
		{1.01,0.3},
		{0.7,0.1},
		{0.88,0.24},
		{0.76,0.4},
		{0.6,0.3},
		////////////////////////////↑ポジティブデータ ↓NGデータとします.
		{0.21,0.96},
		{0.32,0.99},
		{0.14,1.21},
		{0.21,1.01},
		{0.26,0.74},
		{0.51,0.96},
		{0.02,0.69},
		{0.34,1.21},
		{0.11,1.1},
		{0.3,0.55}
	};

	for(int i=0;i<data.size();i++){
		for(int j=0;j<data[i].size();j++){
			data[i][j]=_data[i][j];
		}
	}

	const double check_data[CHECK_DATA_COUNT][USE_FEATURE_COUNT]={
		{0.1,0.8},
		{0.2,0.6},
		{0.4,0.4},
		{0.6,0.2},
		{1.0,0.15}
	};

	////POSデータの設定です．
	for(int i=0;i<NUM_POS;i++){
		for(int j=0;j<USE_FEATURE_COUNT;j++)
			trainingData.at<float>(i,j)=data[i][j];//(i番目,i番目のj要素)=代入
		trainingLabels.at<float>(i,0)=1;//(i番目,0（配列は0からはじまるため)）=代入するラベル
		std::cout << ".";
	}

	////NGの設定
	//注意！　trainingDataを使いまわすので，i+NUM_POSとしないとPOSを上書きしてしまいます．
	for(int i=0;i<NUM_NEG;i++){
		for(int j=0;j<USE_FEATURE_COUNT;j++)
			trainingData.at<float>(i+NUM_POS,j)=data[i+NUM_POS][j];
		trainingLabels.at<float>(i+NUM_POS,0)=-1;
		std::cout << ".";
	}



	CvSVM svm;// = CvSVM();
	CvSVMParams svm_param;
	CvTermCriteria criteria;

	criteria = cvTermCriteria (CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	svm_param = CvSVMParams (CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);
	svm.train(trainingData,trainingLabels,cv::Mat(),cv::Mat(),svm_param);
	svm.save("SVM.xml");//trainしたデータを書きだす．

	///////////////////↑学習終了//////////////////

	////////////////↓識別開始///////////////////

	cv::Mat result;
	result.create(1,USE_FEATURE_COUNT,CV_32FC1);//確認したいデータ数,次元数　

	for(int i=0;i<CHECK_DATA_COUNT;i++){
		for(int j=0;j<USE_FEATURE_COUNT;j++)
			result.at<float>(0,j)=check_data[i][j];// 0.8 0.2とかだと，1になる ここの値をいろいろ変えて，predictの様子を見てください
		cv::SVM svm_predict;
		svm_predict.load("SVM.xml");
		std::cout << std::endl;
		std::cout << "SVM診断結果:" << svm_predict.predict(result);//predictで診断します．　1or-1が戻り値
	}
	cv::Mat bufferMat( 320, 240, CV_8UC4 );
	cv::imshow("test",bufferMat);
	cv::waitKey();
}