//
// Created by wang on 19-8-17.
//

#ifndef ORB_SLAM2_FRFNET_H
#define ORB_SLAM2_FRFNET_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"

namespace DBoW2 {

/// Functions to manipulate ORB descriptors
    class FRFNET: protected FClass
    {
    public:

        /// Descriptor type
        typedef std::vector<float> TDescriptor; // CV_8U
        /// Pointer to a single descriptor
        typedef const TDescriptor *pDescriptor;
        /// Descriptor length (in bytes)
        static const int L;

        /**
         * Returns the number of dimensions of the descriptor space
         * @return dimensions
         */
        inline static int dimensions()
        {
            return L;
        }

        /**
         * Calculates the mean value of a set of descriptors
         * @param descriptors
         * @param mean mean descriptor
         */
        static void meanValue(const std::vector<pDescriptor> &descriptors,
                              TDescriptor &mean);

        /**
         * Calculates the distance between two descriptors
         * @param a
         * @param b
         * @return distance
         */
        static float distance(const TDescriptor &a, const TDescriptor &b);

        /**
         * Returns a string version of the descriptor
         * @param a descriptor
         * @return string version
         */
        static std::string toString(const TDescriptor &a);

        /**
         * Returns a descriptor from a string
         * @param a descriptor
         * @param s string version
         */
        static void fromString(TDescriptor &a, const std::string &s);

        /**
         * Returns a mat with the descriptors in float format
         * @param descriptors
         * @param mat (out) NxL 32F matrix
         */
        static void toMat32F(const std::vector<TDescriptor> &descriptors,
                             cv::Mat &mat);

        //static void toMat8U(const std::vector<TDescriptor> &descriptors,
        //cv::Mat &mat);

    };

} // namespace DBoW2

#endif //ORB_SLAM2_FRFNET_H
