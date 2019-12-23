#include <cstdlib>
#include <iostream>
#include <vector>
#include <bitset>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>


#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "Thirdparty/DBoW2/DBoW2/FLFNET.h"
#include "Thirdparty/DBoW2/DBoW2/FDLF.h"

#include "DLFhandler.h"
#include "omp.h"

using namespace DBoW2;
using namespace cv;
using namespace std;


void writeMatToFile(cv::Mat& m, basic_string<char, char_traits<char>, allocator<char>> filename)
{
    std::ofstream fout(filename);

    if (!fout)
    {
        std::cout << "File Not Opened" << std::endl;
        return;
    }

    for (int i = 0; i<m.rows; i++)
    {
        for (int j = 0; j<m.cols; j++)
        {
            fout << m.at<float>(i, j) << "\t";
        }
        fout << std::endl;
    }

    fout.close();
}


int main() {
    //0. 参数
    string strAssociationFilename = "/home/wang/workspace/data/rgbd_dataset_freiburg1_room/1.txt";
    string datapath = "/home/wang/workspace/data/rgbd_dataset_freiburg1_room/rgb";
    vector<double> vTimestamps;

    const int kpts_num = 1000;//超参数,可以根据情况修改 每张照片提取特征点的个数 1000个,目前是跟ORB本身的个数是一样的.

    vector<vector<float>> dspts;
    typedef TemplatedVocabulary<FDLF::TDescriptor, FDLF> FDLFVocabulary;
    //!!Note: 9 and 3 is the parameters you may want to change
    FDLFVocabulary DLFvoc(10, 6, TF_IDF, L1_NORM);
    vector<vector<vector<float>>> features;

    //1. load features
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while (!fAssociation.eof()) {
        string s;
        getline(fAssociation, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    int nImages = vTimestamps.size();// 一共多少照片
    cout << "all images number = " << nImages << endl;

    ORB_SLAM2::DLF DLFextract("/home/wang/workspace/RFNet_JIT/NASNet_0.1/des.pt",
                              "/home/wang/workspace/RFNet_JIT/NASNet_0.1/kpt.pt", 32);

    for (int ni = 0; ni < nImages; ni = ni + 6) {
        double tframe = vTimestamps[ni];
        string strTimeStamp = to_string(tframe);
        cv::Mat imRGB = cv::imread(datapath + "/" + strTimeStamp + ".png", CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imGray;
        cvtColor(imRGB, imGray, CV_RGB2GRAY);
        cv::Mat img_float;
        imGray.convertTo(img_float, CV_32FC1, 1.f / 255.f, 0);
        torch::Tensor patches;
        auto kps = DLFextract.get_kps(img_float, patches);
        auto des = DLFextract.get_des(patches);
//        for (int i = 0; i < kpts_num; i++)//定义行循环
//        {
//            mvKpts[i].pt.x = kps[i][1].item().toFloat();
//            mvKpts[i].pt.y = kps[i][0].item().toFloat();
//        }
//        std::cout<< des<<endl;
        torch::Tensor des_tensor_cpu = des.cpu();
        cv::Mat descripter(des_tensor_cpu.size(0), des_tensor_cpu.size(1), CV_32F, des_tensor_cpu.data<float_t>());
//        writeMatToFile(descripter,"/home/wang/workspace/"+strTimeStamp+".txt");
//        std::cout<< descripter.cols<<endl;

        for (int i = 0; i < kpts_num; i++)//定义行循环
        {
            vector<float> dspt;
            dspt.resize(128);
            for (int j = 0; j < 128; j= j+1){
                dspt[j] = descripter.at<float>(i, j);
            }
            dspts.push_back(dspt);
//            cout << "dspts size now is: " << dspts.size() << endl;
        }
        features.push_back(dspts);
        cout << "features size now is: " << features.size() << endl;
    }

    cout << "creating the vocabulary..." << endl;
    DLFvoc.create(features);
    cout << "saving the vocabulary to text file..." << endl;
    DLFvoc.saveToTextFile("/home/wang/workspace/OURL1.txt");
}


#pragma clang diagnostic pop
