// ConsoleApplication1.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//
// �@�B�w�K�v���O����
// trainingData.csv��ǂݍ���Ŋw�K
// SVM.xml, SVMTrainautoParam.xml ���o��
// checkData.csv 

#include "stdafx.h"

#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>



// csv file�ǂݍ���
std::vector<std::vector<std::string>> my_csv_import(std::string filename){
	std::ifstream file(filename);
	std::vector<std::vector<std::string>> values;
	std::vector<double> nums;
	std::string str;
	int p;

	//	getline(file,str);//1�s�ǂݎ̂�
	while(getline(file, str)){
		//�R�����g�ӏ��͏���
		if( (p = str.find("//")) != str.npos ) continue;
		std::vector<std::string> inner;

		//�R���}�����邩��T���A�����܂ł�values�Ɋi�[
		while( (p = str.find(",")) != str.npos ){
			inner.push_back(str.substr(0, p));

			//str�̒��g��","��1�������΂�
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

//csv����g���[�j���O�f�[�^�ƃ��x����ǂݍ���
void setTrainingData(const std::string filename, cv::Mat& trainingData, cv::Mat& trainingLabels){

	std::vector<std::vector<std::string>> csvFileData;
	csvFileData = my_csv_import(filename);
	const int NUM_ALL = csvFileData.size();//�S�f�[�^��
	const int USE_FEATURE_COUNT = csvFileData.at(1).size()-1;//1�̃f�[�^�̎�����//2�s�ڂ̓��������猈��//1�߂�label�Ȃ̂�-1

	std::cout << "�w�K�f�[�^��" << NUM_ALL << std::endl;
	std::cout << "������" << USE_FEATURE_COUNT << std::endl;

	trainingData.create(NUM_ALL,USE_FEATURE_COUNT,CV_32FC1);//�S�f�[�^���C��������Mat�^��p��
	trainingLabels.create(NUM_ALL,1,CV_32FC1);//���x�������Ȃ̂ŁC��������1�ł����ł��D

	for(int i=0;i<NUM_ALL;i++){
		for(int j=0;j<USE_FEATURE_COUNT;j++){
			trainingData.at<float>(i,j)=atof(csvFileData.at(i).at(j+1/*0��label�̂���+1*/).c_str());//data[i][j];//(i�Ԗ�,i�Ԗڂ�j�v�f)=���
		}
		trainingLabels.at<float>(i,0) = atof(csvFileData.at(i).at(0).c_str());
		std::cout << ".";
	}
	std::cout << std::endl;

}

void setTrainingData(std::vector<std::vector<std::string>> csvFileData, cv::Mat& trainingData, cv::Mat& trainingLabels, const int kFold, const int k){

//	std::vector<std::vector<std::string>> csvFileData;
//	csvFileData = my_csv_import(filename);
	const int NUM_ALL = csvFileData.size();//�S�f�[�^��
	const int KFOLD_NUM_ALL = (NUM_ALL / kFold) *(kFold-1);
	const int USE_FEATURE_COUNT = csvFileData.at(1).size()-1;//1�̃f�[�^�̎�����//2�s�ڂ̓��������猈��//1�߂�label�Ȃ̂�-1

	std::cout << "�w�K�f�[�^��" << KFOLD_NUM_ALL << std::endl;
	std::cout << "������" << USE_FEATURE_COUNT << std::endl;


	trainingData.create(KFOLD_NUM_ALL,USE_FEATURE_COUNT,CV_32FC1);//�S�f�[�^���C��������Mat�^��p��
	trainingLabels.create(KFOLD_NUM_ALL,1,CV_32FC1);//���x�������Ȃ̂ŁC��������1

	int skip=0;
	for( int i=0; i<KFOLD_NUM_ALL; i++ ){

		if(i==(NUM_ALL / kFold)*k){//fold����
			skip+=(NUM_ALL / kFold);
		}

		for(int j=0;j<USE_FEATURE_COUNT;j++){
			trainingData.at<float>(i,j)=atof(csvFileData.at(i+skip).at(j+1/*0��label�̂���+1*/).c_str());//data[i][j];//(i�Ԗ�,i�Ԗڂ�j�v�f)=���
		}
		trainingLabels.at<float>(i,0) = atof(csvFileData.at(i+skip).at(0).c_str());
		std::cout << ".";
	}
	std::cout << std::endl;

}

//������svm�p�����[�^��csv�t�@�C�����̓������琄��
void checkSVMResult(const cv::SVM& svm, const std::string& checkFilesCSV){
	std::vector<std::vector<std::string>> csvFileData;
	csvFileData = my_csv_import(checkFilesCSV);
	const int CHECK_DATA_COUNT = csvFileData.size();//�S�f�[�^��
	const int USE_FEATURE_COUNT = csvFileData.at(1).size()-1;//1�̃f�[�^�̎�����//2�s�ڂ̓��������猈��//1�߂�label�Ȃ̂�-1

	std::cout << "�m�F�f�[�^��" << CHECK_DATA_COUNT << std::endl;
	std::cout << "������" << USE_FEATURE_COUNT << std::endl;

	cv::Mat result;
	result.create(1,USE_FEATURE_COUNT,CV_32FC1);

	for(int i=0;i<CHECK_DATA_COUNT;i++){
		for(int j=0;j<USE_FEATURE_COUNT;j++){
			result.at<float>(0,j)=atof(csvFileData.at(i).at(j+1).c_str());
		}
		float truenum = atof(csvFileData.at(i).at(0).c_str());
		std::cout << std::endl
			<< "SVM�f�f����:" << svm.predict(result)
			<< "�^�l:" << truenum
			<< "�^�l�Ƃ̔�r:" << ((svm.predict(result)==truenum)?"true":"false");
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

//�w�K�f�[�^����,trainauto��������(svm�̊w�K����,�p�����[�^)�������o��
void mySVMTrainautoLearn(const std::string kTrainautoLearnedSVMxmlFileName, const std::string kTrainautoParamxmlFileName,const std::string kTrainingDatacsvFileName){
	cv::Mat trainingData;//�����_�f�[�^
	cv::Mat trainingLabels;//���x��
	setTrainingData(kTrainingDatacsvFileName, trainingData, trainingLabels);

	//std::cout << trainingData << std::endl << trainingLabels << std::endl;

	CvSVM svm;// = CvSVM();
	CvSVMParams svm_default_param;

	std::cout << "svm.train_auto" << std::endl;
	svm.train_auto(trainingData,trainingLabels,cv::Mat(),cv::Mat(),svm_default_param);
	CvSVMParams svm_param_trainauto = svm.get_params();
	std::cout << "show trainauto Param" << std::endl;
	showSvmParams(svm_param_trainauto);
	svm.save(kTrainautoLearnedSVMxmlFileName.c_str());//trainauto�����f�[�^�����������D
	writeSVMParam(kTrainautoParamxmlFileName,svm_param_trainauto);//trainauto�����ꍇ��SVM�w�K�p�����[�^�������o���D
}

//k-cross validation���s��
//���Ԓʂ�Ƀt�@�C�����g���̂ŁC�^�l�̏��Ԃ��΂��Ă���ꍇ�̓����_���̏��ԂɕύX����K�v�A��
void crossValidation(const int kFold, const std::string ParamxmlFileName,const std::string kTrainingDatacsvFileName,const bool doShuffle=false){

	std::vector<std::vector<std::string>> csvFileData;
	csvFileData = my_csv_import(kTrainingDatacsvFileName);
	const int NUM_ALL = csvFileData.size();//�S�f�[�^��
	const int USE_FEATURE_COUNT = csvFileData.at(1).size()-1;//1�̃f�[�^�̎�����//2�s�ڂ̓��������猈��//1�߂�label�Ȃ̂�-1
	const int KFOLD_NUM_ALL = (NUM_ALL / kFold) *(kFold-1);

	if(doShuffle){
		std::random_shuffle(csvFileData.begin()+1, csvFileData.end());
	}

	std::cout << "�w�K�f�[�^��" << NUM_ALL << std::endl;
	std::cout << "������" << USE_FEATURE_COUNT << std::endl;

	CvSVMParams svmparam = readSVMParam(ParamxmlFileName);
	showSvmParams(svmparam);

	int correctnum=0;
	int maxcorrectnum=0;

	for(int i=0;i<kFold;i++){
		CvSVM svm;
		cv::Mat trainingData;//�����_�f�[�^
		cv::Mat trainingLabels;//���x��

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
				<< "SVM�f�f����:" << svm.predict(result)
				<< "�^�l:" << truenum
				<< "�^�l�Ƃ̔�r:" << ((svm.predict(result)==truenum)?"true":"false");

			maxcorrectnum++;
			if(svm.predict(result)==truenum)
				correctnum++;
		}
					std::cout << std::endl;

	}
	
	std::cout << std::endl
		<< "������:" << (double)correctnum/maxcorrectnum << "(" << correctnum << "/" << maxcorrectnum
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

	////////////////�����ʊJ�n///////////////////
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
