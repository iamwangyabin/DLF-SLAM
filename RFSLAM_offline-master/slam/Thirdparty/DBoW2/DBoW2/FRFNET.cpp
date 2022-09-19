//
// Created by wang on 19-8-17.
//

#include <vector>
#include <string>
#include <sstream>

#include "FRFNET.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

    const int FRFNET::L=128;

    void FRFNET::meanValue(const std::vector<FRFNET::pDescriptor> &descriptors,
                           FRFNET::TDescriptor &mean)
    {
        mean.resize(0);
        mean.resize(FRFNET::L, 0);

        float s = descriptors.size();

        vector<FRFNET::pDescriptor>::const_iterator it;
        for(it = descriptors.begin(); it != descriptors.end(); ++it)
        {
            const FRFNET::TDescriptor &desc = **it;
            for(int i = 0; i < FRFNET::L; i += 4)
            {
                mean[i  ] += desc[i  ] / s;
                mean[i+1] += desc[i+1] / s;
                mean[i+2] += desc[i+2] / s;
                mean[i+3] += desc[i+3] / s;
            }
        }
    }

// --------------------------------------------------------------------------

    float FRFNET::distance(const FRFNET::TDescriptor &a,
                           const FRFNET::TDescriptor &b)
    {
        float sqd = 0.;
        for(int i = 0; i < FRFNET::L; i += 4)
        {
            sqd += (a[i  ] - b[i  ])*(a[i  ] - b[i  ]);
            sqd += (a[i+1] - b[i+1])*(a[i+1] - b[i+1]);
            sqd += (a[i+2] - b[i+2])*(a[i+2] - b[i+2]);
            sqd += (a[i+3] - b[i+3])*(a[i+3] - b[i+3]);
        }
        //sqd = sqrt(sqd);
        return sqd;
    }

// --------------------------------------------------------------------------

    std::string FRFNET::toString(const FRFNET::TDescriptor &a)
    {
        stringstream ss;
        for(int i = 0; i < FRFNET::L; ++i)
        {
            ss << a[i] << " ";
        }
        return ss.str();
    }

// --------------------------------------------------------------------------

    void FRFNET::fromString(FRFNET::TDescriptor &a, const std::string &s)
    {
        a.resize(FRFNET::L);

        stringstream ss(s);
        for(int i = 0; i < FRFNET::L; ++i)
        {
            ss >> a[i];
        }
    }

// --------------------------------------------------------------------------

    void FRFNET::toMat32F(const std::vector<TDescriptor> &descriptors,
                          cv::Mat &mat)
    {
        if(descriptors.empty())
        {
            mat.release();
            return;
        }

        const int N = descriptors.size();
        const int L = FRFNET::L;

        mat.create(N, L, CV_32F);

        for(int i = 0; i < N; ++i)
        {
            const TDescriptor& desc = descriptors[i];
            float *p = mat.ptr<float>(i);
            for(int j = 0; j < L; ++j, ++p)
            {
                *p = desc[j];
            }
        }
    }

} // namespace DBoW2


