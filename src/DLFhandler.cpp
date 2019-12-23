//
// Created by wang on 2019/12/5.
//

#include <utility>
#include "DLFhandler.h"

torch::Tensor
ORB_SLAM2::DLF::clip_patch(torch::Tensor &kpts_byxc, torch::Tensor kpts_scale, torch::Tensor &kpts_ori,
                  torch::Tensor &image) {
    assert(kpts_byxc.sizes()[0] == kpts_scale.sizes()[0]);
    int out_width = PSIZE;
    int out_height = PSIZE;

    int B = 1;
    int im_height = image.sizes()[2];
    int im_width = image.sizes()[3];
    int num_kp = kpts_byxc.sizes()[0];
    int max_y = im_height - 1;
    int max_x = im_width - 1;

    auto temp = torch::meshgrid({torch::linspace(-1, 1, out_height).to(torch::kFloat32).to(device),
                                 torch::linspace(-1, 1, out_width).to(torch::kFloat32).to(device)});
    torch::Tensor y_t = temp[0];
    torch::Tensor x_t = temp[1];
    torch::Tensor one_t = torch::ones_like(x_t);
    x_t = x_t.contiguous().view(-1);
    y_t = y_t.contiguous().view(-1);
    one_t = one_t.view(-1);
    torch::Tensor grid = torch::stack({x_t, y_t, one_t});
    grid = grid.view(-1).repeat(num_kp).view({num_kp, 3, -1});

    torch::Tensor thetas = torch::eye(2, 3).to(torch::kFloat).to(device);
    thetas = thetas.unsqueeze(0).repeat({num_kp, 1, 1});
    kpts_scale = kpts_scale.view({B, -1});
    kpts_scale = kpts_scale.view(-1) / 2.0;
    thetas = thetas * kpts_scale.unsqueeze(-1).unsqueeze(-1);
    torch::Tensor ones(torch::tensor({0, 0, 1}).unsqueeze(0).unsqueeze(0));
    ones = ones.to(torch::kFloat).to(device).repeat({num_kp, 1, 1});
    thetas = torch::cat({thetas, ones}, 1);

    torch::Tensor cos = kpts_ori.slice(1, 0, 1);
    torch::Tensor sin = kpts_ori.slice(1, 1, 2);
    torch::Tensor zeros = torch::zeros_like(cos);
    torch::Tensor _ones = torch::ones_like(cos);
    torch::Tensor R = torch::cat({cos, -sin, zeros, sin, cos, zeros, zeros, zeros, _ones}, -1);
    R = R.view({-1, 3, 3});
    thetas = torch::matmul(thetas, R);

    torch::Tensor T_g = torch::matmul(thetas, grid);
    torch::Tensor x = T_g.slice(1, 0, 1).squeeze(1);
    torch::Tensor y = T_g.slice(1, 1, 2).squeeze(1);
    torch::Tensor kp_x_ofst = kpts_byxc.slice(1, 2, 3).view({B, -1}).to(torch::kFloat32);
    kp_x_ofst = kp_x_ofst.view({-1, 1});

    torch::Tensor kp_y_ofst = kpts_byxc.slice(1, 1, 2).view({B, -1}).to(torch::kFloat32);
    kp_y_ofst = kp_y_ofst.view({-1, 1});
    x = x + kp_x_ofst;
    y = y + kp_y_ofst;
    x = x.view({-1});
    y = y.view({-1});
    torch::Tensor x0 = x.floor();
    torch::Tensor x1 = x0 + 1;
    torch::Tensor y0 = y.floor();
    torch::Tensor y1 = y0 + 1;

    x0 = x0.clamp(0, max_x);
    x1 = x1.clamp(0, max_x);
    y0 = y0.clamp(0, max_y);
    y1 = y1.clamp(0, max_y);
    int dim2 = im_width;
    int dim1 = im_width * im_height;
    torch::Tensor batch_inds = kpts_byxc.slice(1, 0, 1);
    torch::Tensor base = batch_inds.repeat({1, out_height * out_width}).to(torch::kFloat32);
    base = base.view({-1}) * dim1;
    torch::Tensor base_y0 = base + y0 * dim2;
    torch::Tensor base_y1 = base + y1 * dim2;
    torch::Tensor im_flat = image.view({-1});
    torch::Tensor idx_a = (base_y0 + x0).to(torch::kLong).to(device);
    torch::Tensor idx_b = (base_y1 + x0).to(torch::kLong).to(device);
    torch::Tensor idx_c = (base_y0 + x1).to(torch::kLong).to(device);
    torch::Tensor idx_d = (base_y1 + x1).to(torch::kLong).to(device);
    torch::Tensor Ia = im_flat.gather(0, idx_a);
    torch::Tensor Ib = im_flat.gather(0, idx_b);
    torch::Tensor Ic = im_flat.gather(0, idx_c);
    torch::Tensor Id = im_flat.gather(0, idx_d);
    torch::Tensor x0_f = x0.to(torch::kFloat);
    torch::Tensor x1_f = x1.to(torch::kFloat);
    torch::Tensor y0_f = y0.to(torch::kFloat);
    torch::Tensor y1_f = y1.to(torch::kFloat);
    torch::Tensor wa = (x1_f - x) * (y1_f - y);
    torch::Tensor wb = (x1_f - x) * (y - y0_f);
    torch::Tensor wc = (x - x0_f) * (y1_f - y);
    torch::Tensor wd = (x - x0_f) * (y - y0_f);
    torch::Tensor output = wa * Ia + wb * Ib + wc * Ic + wd * Id;

    output = output.view({num_kp, out_height, out_width});
    return output.unsqueeze(1);
}

torch::Tensor ORB_SLAM2::DLF::get_kps(const cv::Mat& imgGray, torch::Tensor &patches) {
    cv::Mat img_float;
    imgGray.convertTo(img_float, CV_32FC1, 1.f / 255.f, 0);
    torch::Tensor img_tensor = torch::from_blob(img_float.data, {1, 1, 480, 640}, at::kFloat).to(device);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor);
    auto output = det_module->forward(inputs).toTuple();
    torch::Tensor kpts = output->elements()[0].toTensor();
    torch::Tensor scales = output->elements()[1].toTensor();
    torch::Tensor orints = output->elements()[2].toTensor();
    patches = clip_patch(kpts,scales,orints,img_tensor);
    return kpts.slice(1,1,3);
}

torch::Tensor ORB_SLAM2::DLF::get_des(torch::Tensor &patches) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(patches);
    auto output = des_module->forward(inputs).toTensor();
    return output;
}

ORB_SLAM2::DLF::DLF(std::string des_path, std::string det_path, int PSIZE) {
    des_module_path = std::move(des_path);
    det_module_path = std::move(det_path);
    DLF::PSIZE = PSIZE;   //32
    det_module = std::make_shared<torch::jit::script::Module>(torch::jit::load(det_module_path));
    des_module = std::make_shared<torch::jit::script::Module>(torch::jit::load(des_module_path));
}
