// ConsoleApplication1.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>


// csv file読み込み
std::vector<std::vector<std::string>> my_csv_import(std::string filename, int& length){
	std::ifstream file(filename);
    std::vector<std::vector<std::string>> values;
	std::vector<double> nums;
    std::string str;
    int p;
 
//	getline(file,str);//1行読み捨て
	length=0;
    while(getline(file, str)){
        //コメント箇所は除く
        if( (p = str.find("//")) != str.npos ) continue;
        std::vector<std::string> inner;

        //コンマがあるかを探し、そこまでをvaluesに格納
        while( (p = str.find(",")) != str.npos ){
            inner.push_back(str.substr(0, p));

            //strの中身は","の1文字を飛ばす
            str = str.substr(p+1);
        }
        inner.push_back(str);
        values.push_back(inner);

		length++;
    }

	return values;
}

void setTrainingData(cv::Mat& trainingData, cv::Mat& trainingLabels){
	const int NUM_POS=10;//ポジティブデータ数.　POS
	const int NUM_NEG=10;//ネガティブ数　NEG
	const int NUM_ALL=NUM_POS+NUM_NEG;//全データ個数
	const int USE_FEATURE_COUNT = 2;//1つのデータの次元数

	//my_csv_import()
	//NUM_ALL=length
	//USE_FEATURE_COUNT
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


	std::vector<std::vector<const double>> data(NUM_ALL);
	for( int i=0; i<NUM_ALL; i++ ){
		data[i].resize(USE_FEATURE_COUNT);
	}
	for(int i=0;i<data.size();i++){
		for(int j=0;j<data[i].size();j++){
			data[i][j]=_data[i][j];
		}
	}

	std::vector<const double> label(NUM_ALL);
	for(int i=0;i<label.size();i++){
		if(i<NUM_POS){
			label.at(i) = 1;
		}else{
			label.at(i) = -1;
		}
	}


	trainingData.create(NUM_ALL,USE_FEATURE_COUNT,CV_32FC1);//全データ数，次元数のMat型を用意
	trainingLabels.create(NUM_ALL,1,CV_32FC1);//ラベルだけなので，次元数は1でいいです．

	for(int i=0;i<NUM_ALL;i++){
		for(int j=0;j<USE_FEATURE_COUNT;j++){
			trainingData.at<float>(i,j)=data[i][j];//(i番目,i番目のj要素)=代入
		}
		trainingLabels.at<float>(i,0) = label.at(i);
		std::cout << ".";
	}

}

int _tmain(int argc, _TCHAR* argv[]){


	const int USE_FEATURE_COUNT = 2;//1つのデータの次元数

	const int CHECK_DATA_COUNT=5;

	cv::Mat trainingData;//ここにデータを入れておきます．
	cv::Mat trainingLabels;//+１か-1かラベルをいれます．

	setTrainingData(trainingData,trainingLabels);

	std::cout << trainingData << trainingLabels;

	CvSVM svm;// = CvSVM();
	CvSVMParams svm_param;
	CvTermCriteria criteria;

	//criteria = cvTermCriteria (CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//svm_param = CvSVMParams (CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);
	//svm.train(trainingData,trainingLabels,cv::Mat(),cv::Mat(),svm_param);
	//svm.save("SVM.xml");//trainしたデータを書きだす．
	criteria = cvTermCriteria (CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	svm_param = CvSVMParams (CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);
	svm.train_auto(trainingData,trainingLabels,cv::Mat(),cv::Mat(),svm_param);
	CvSVMParams svm_param_trainauto = svm.get_params();
	std::cout << "defaultParam" << std::endl
		<< svm_param.C
		<< svm_param.class_weights
		<< svm_param.coef0
		<< svm_param.degree
		<< svm_param.gamma
		<< svm_param.kernel_type
		<< svm_param.nu
		<< svm_param.p
		<< svm_param.svm_type
		<< svm_param.term_crit.epsilon
		<< svm_param.term_crit.max_iter
		<< svm_param.term_crit.type
		<< std::endl;
	std::cout << "trainauto" << std::endl
		<< svm_param_trainauto.C
		<< svm_param_trainauto.class_weights
		<< svm_param_trainauto.coef0
		<< svm_param_trainauto.degree
		<< svm_param_trainauto.gamma
		<< svm_param_trainauto.kernel_type
		<< svm_param_trainauto.nu
		<< svm_param_trainauto.p
		<< svm_param_trainauto.svm_type
		<< svm_param_trainauto.term_crit.epsilon
		<< svm_param_trainauto.term_crit.max_iter
		<< svm_param_trainauto.term_crit.type
		<< std::endl;
	svm.save("SVM.xml");//trainしたデータを書きだす．
	///////////////////↑学習終了//////////////////

	////////////////↓識別開始///////////////////
	const double check_data[CHECK_DATA_COUNT][USE_FEATURE_COUNT]={
		{0.1,0.8},
		{0.2,0.6},
		{0.4,0.4},
		{0.6,0.2},
		{1.0,0.15}
	};
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