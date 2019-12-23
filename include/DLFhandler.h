//
// Created by wang on 2019/12/5.
//

#ifndef ORB_SLAM2_DLFHANDLER_H
#define ORB_SLAM2_DLFHANDLER_H

#include <iostream>
#include <vector>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <ATen/Tensor.h>

#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>


namespace ORB_SLAM2 {

    class DLF {
    public:
        DLF(std::string des_path, std::string det_path, int PSIZE);

        torch::Tensor
        clip_patch(torch::Tensor &kpts_byxc, torch::Tensor kpts_scale, torch::Tensor &kpts_ori, torch::Tensor &image);

        torch::Tensor get_des(torch::Tensor &patches);

        torch::Tensor get_kps(const cv::Mat &imgGray, torch::Tensor &patches);

    private:
        std::string des_module_path;
        std::string det_module_path;
        std::shared_ptr<torch::jit::script::Module> det_module;
        std::shared_ptr<torch::jit::script::Module> des_module;

        int PSIZE;

        torch::DeviceType device_type = torch::kCUDA;
        torch::Device device = torch::Device(device_type);
    };
}
#endif //ORB_SLAM2_DLFHANDLER_H
