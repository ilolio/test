// ConsoleApplication1.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//
// 機械学習プログラム
// trainingData.csvを読み込んで学習
// SVM.xml, SVMTrainautoParam.xml を出力
// checkData.csv 

#include "stdafx.h"

#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>



// csv file読み込み
std::vector<std::vector<std::string>> my_csv_import(std::string filename){
	std::ifstream file(filename);
	std::vector<std::vector<std::string>> values;
	std::vector<double> nums;
	std::string str;
	int p;

	//	getline(file,str);//1行読み捨て
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

	}

	return values;
}

void showSvmParams(cv::SVMParams svm_param){
	std::cout << "Param" << std::endl
		<< "C:" << svm_param.C << std::endl
		<< "class_weights:" << svm_param.class_weights << std::endl
		<< "coef0:" << svm_param.coef0 << std::endl
		<< "degree:" << svm_param.degree << std::endl
		<< "gamma:" << svm_param.gamma << std::endl
		<< "kernel_type:" << svm_param.kernel_type << std::endl
		<< "nu:" << svm_param.nu << std::endl
		<< "p:" << svm_param.p << std::endl
		<< "svm_type:" << svm_param.svm_type << std::endl
		<< "term_crit.epsilon:" << svm_param.term_crit.epsilon << std::endl
		<< "term_crit.max_iter:" << svm_param.term_crit.max_iter << std::endl
		<< "term_crit.type:" << svm_param.term_crit.type << std::endl
		<< std::endl;
}

//csvからトレーニングデータとラベルを読み込み
void setTrainingData(const std::string filename, cv::Mat& trainingData, cv::Mat& trainingLabels){

	std::vector<std::vector<std::string>> csvFileData;
	csvFileData = my_csv_import(filename);
	const int NUM_ALL = csvFileData.size();//全データ個数
	const int USE_FEATURE_COUNT = csvFileData.at(1).size()-1;//1つのデータの次元数//2行目の特徴数から決定//1つめはlabelなので-1

	std::cout << "学習データ数" << NUM_ALL << std::endl;
	std::cout << "特徴数" << USE_FEATURE_COUNT << std::endl;

	trainingData.create(NUM_ALL,USE_FEATURE_COUNT,CV_32FC1);//全データ数，次元数のMat型を用意
	trainingLabels.create(NUM_ALL,1,CV_32FC1);//ラベルだけなので，次元数は1でいいです．

	for(int i=0;i<NUM_ALL;i++){
		for(int j=0;j<USE_FEATURE_COUNT;j++){
			trainingData.at<float>(i,j)=atof(csvFileData.at(i).at(j+1/*0はlabelのため+1*/).c_str());//data[i][j];//(i番目,i番目のj要素)=代入
		}
		trainingLabels.at<float>(i,0) = atof(csvFileData.at(i).at(0).c_str());
		std::cout << ".";
	}
	std::cout << std::endl;

}

void setTrainingData(std::vector<std::vector<std::string>> csvFileData, cv::Mat& trainingData, cv::Mat& trainingLabels, const int kFold, const int k){

//	std::vector<std::vector<std::string>> csvFileData;
//	csvFileData = my_csv_import(filename);
	const int NUM_ALL = csvFileData.size();//全データ個数
	const int KFOLD_NUM_ALL = (NUM_ALL / kFold) *(kFold-1);
	const int USE_FEATURE_COUNT = csvFileData.at(1).size()-1;//1つのデータの次元数//2行目の特徴数から決定//1つめはlabelなので-1

	std::cout << "学習データ数" << KFOLD_NUM_ALL << std::endl;
	std::cout << "特徴数" << USE_FEATURE_COUNT << std::endl;


	trainingData.create(KFOLD_NUM_ALL,USE_FEATURE_COUNT,CV_32FC1);//全データ数，次元数のMat型を用意
	trainingLabels.create(KFOLD_NUM_ALL,1,CV_32FC1);//ラベルだけなので，次元数は1

	int skip=0;
	for( int i=0; i<KFOLD_NUM_ALL; i++ ){

		if(i==(NUM_ALL / kFold)*k){//fold部分
			skip+=(NUM_ALL / kFold);
		}

		for(int j=0;j<USE_FEATURE_COUNT;j++){
			trainingData.at<float>(i,j)=atof(csvFileData.at(i+skip).at(j+1/*0はlabelのため+1*/).c_str());//data[i][j];//(i番目,i番目のj要素)=代入
		}
		trainingLabels.at<float>(i,0) = atof(csvFileData.at(i+skip).at(0).c_str());
		std::cout << ".";
	}
	std::cout << std::endl;

}

//引数のsvmパラメータでcsvファイル内の特徴から推定
void checkSVMResult(const cv::SVM& svm, const std::string& checkFilesCSV){
	std::vector<std::vector<std::string>> csvFileData;
	csvFileData = my_csv_import(checkFilesCSV);
	const int CHECK_DATA_COUNT = csvFileData.size();//全データ個数
	const int USE_FEATURE_COUNT = csvFileData.at(1).size()-1;//1つのデータの次元数//2行目の特徴数から決定//1つめはlabelなので-1

	std::cout << "確認データ数" << CHECK_DATA_COUNT << std::endl;
	std::cout << "特徴数" << USE_FEATURE_COUNT << std::endl;

	cv::Mat result;
	result.create(1,USE_FEATURE_COUNT,CV_32FC1);

	for(int i=0;i<CHECK_DATA_COUNT;i++){
		for(int j=0;j<USE_FEATURE_COUNT;j++){
			result.at<float>(0,j)=atof(csvFileData.at(i).at(j+1).c_str());
		}
		float truenum = atof(csvFileData.at(i).at(0).c_str());
		std::cout << std::endl
			<< "SVM診断結果:" << svm.predict(result)
			<< "真値:" << truenum
			<< "真値との比較:" << ((svm.predict(result)==truenum)?"true":"false");
	}
}

void writeSVMParam(const std::string filename, const CvSVMParams svm_param){
	cv::FileStorage   cvfs(filename,CV_STORAGE_WRITE);

	cv::write(cvfs,"C",svm_param.C);
	cv::write(cvfs,"class_weights",svm_param.class_weights);
	cv::write(cvfs,"coef0",svm_param.coef0);
	cv::write(cvfs,"degree",svm_param.degree);
	cv::write(cvfs,"gamma",svm_param.gamma);
	cv::write(cvfs,"kernel_type",svm_param.kernel_type);
	cv::write(cvfs,"nu",svm_param.nu);
	cv::write(cvfs,"p",svm_param.p);
	cv::write(cvfs,"svm_type",svm_param.svm_type);
	cv::write(cvfs,"term_crit_epsilon",svm_param.term_crit.epsilon);
	cv::write(cvfs,"term_crit_max_iter",svm_param.term_crit.max_iter);
	cv::write(cvfs,"term_crit_type",svm_param.term_crit.type);

}

CvSVMParams readSVMParam(const std::string filename){
	CvSVMParams svm_param;
	cv::FileStorage cvfs(filename,CV_STORAGE_READ);

	cv::FileNode node(cvfs.fs, NULL); // Get Top Node

	CvTermCriteria criteria;
	criteria = cvTermCriteria (node["term_crit_type"], node["term_crit_max_iter"], node["term_crit_epsilon"]);

	svm_param = CvSVMParams (
		node["svm_type"], node["kernel_type"],
		node["degree"],node["gamma"],node["coef0"],
		node["C"],node["nu"],node["p"],
		NULL, criteria);

	return svm_param;
}

//学習データから,trainautoした結果(svmの学習結果,パラメータ)を書き出す
void mySVMTrainautoLearn(const std::string kTrainautoLearnedSVMxmlFileName, const std::string kTrainautoParamxmlFileName,const std::string kTrainingDatacsvFileName){
	cv::Mat trainingData;//特徴点データ
	cv::Mat trainingLabels;//ラベル
	setTrainingData(kTrainingDatacsvFileName, trainingData, trainingLabels);

	//std::cout << trainingData << std::endl << trainingLabels << std::endl;

	CvSVM svm;// = CvSVM();
	CvSVMParams svm_default_param;

	std::cout << "svm.train_auto" << std::endl;
	svm.train_auto(trainingData,trainingLabels,cv::Mat(),cv::Mat(),svm_default_param);
	CvSVMParams svm_param_trainauto = svm.get_params();
	std::cout << "show trainauto Param" << std::endl;
	showSvmParams(svm_param_trainauto);
	svm.save(kTrainautoLearnedSVMxmlFileName.c_str());//trainautoしたデータを書きだす．
	writeSVMParam(kTrainautoParamxmlFileName,svm_param_trainauto);//trainautoした場合のSVM学習パラメータを書き出す．
}

//k-cross validationを行う
//順番通りにファイルを使うので，真値の順番が偏っている場合はランダムの順番に変更する必要アリ
void crossValidation(const int kFold, const std::string ParamxmlFileName,const std::string kTrainingDatacsvFileName,const bool doShuffle=false){

	std::vector<std::vector<std::string>> csvFileData;
	csvFileData = my_csv_import(kTrainingDatacsvFileName);
	const int NUM_ALL = csvFileData.size();//全データ個数
	const int USE_FEATURE_COUNT = csvFileData.at(1).size()-1;//1つのデータの次元数//2行目の特徴数から決定//1つめはlabelなので-1
	const int KFOLD_NUM_ALL = (NUM_ALL / kFold) *(kFold-1);

	if(doShuffle){
		std::random_shuffle(csvFileData.begin()+1, csvFileData.end());
	}

	std::cout << "学習データ数" << NUM_ALL << std::endl;
	std::cout << "特徴数" << USE_FEATURE_COUNT << std::endl;

	CvSVMParams svmparam = readSVMParam(ParamxmlFileName);
	showSvmParams(svmparam);

	int correctnum=0;
	int maxcorrectnum=0;

	for(int i=0;i<kFold;i++){
		CvSVM svm;
		cv::Mat trainingData;//特徴点データ
		cv::Mat trainingLabels;//ラベル

		setTrainingData(csvFileData, trainingData, trainingLabels,kFold,i);
		std::cout << std::endl;
		svm.train(trainingData,trainingLabels,cv::Mat(),cv::Mat(),svmparam);


		cv::Mat result;
		result.create(1,USE_FEATURE_COUNT,CV_32FC1);

		for(int j=0;j<(NUM_ALL / kFold);j++){
			for(int k=0;k<USE_FEATURE_COUNT;k++){
				result.at<float>(0,k)=atof(csvFileData.at(j+(NUM_ALL / kFold)*i).at(k+1).c_str());
			}
			float truenum = atof(csvFileData.at(j+(NUM_ALL / kFold)*i).at(0).c_str());
			std::cout << std::endl
				<< "SVM診断結果:" << svm.predict(result)
				<< "真値:" << truenum
				<< "真値との比較:" << ((svm.predict(result)==truenum)?"true":"false");

			maxcorrectnum++;
			if(svm.predict(result)==truenum)
				correctnum++;
		}
					std::cout << std::endl;

	}
	
	std::cout << std::endl
		<< "正答率:" << (double)correctnum/maxcorrectnum << "(" << correctnum << "/" << maxcorrectnum
		<< std::endl;

}



int _tmain(int argc, _TCHAR* argv[]){

	const std::string kTrainautoLearnedSVMxmlFileName = "SVM.xml";
	const std::string kTrainautoParamxmlFileName = "SVMTrainautoParam.xml";
	const std::string kTrainingDatacsvFileName = "trainingData.csv";


	mySVMTrainautoLearn(kTrainautoLearnedSVMxmlFileName, kTrainautoParamxmlFileName, kTrainingDatacsvFileName);

	//CvSVMParams trainautoParam = readSVMParam("SVMParam.xml");
	//std::cout << "read params:" << std::endl;
	//showSvmParams(trainautoParam);
	//svm.train(trainingData,trainingLabels,cv::Mat(),cv::Mat(),trainautoParam);

	////////////////↓識別開始///////////////////
	std::cout << "checkSVMResult" << std::endl;

	const std::string SVMFileName="SVM.xml";
	cv::SVM svm_predict;
	svm_predict.load(SVMFileName.c_str());
	checkSVMResult(svm_predict,"checkData.csv");

	crossValidation(/*kFold=*/3, kTrainautoParamxmlFileName, kTrainingDatacsvFileName,true);


	cv::Mat bufferMat( 320, 240, CV_8UC4 );
	cv::imshow("test",bufferMat);
	cv::waitKey();
}
