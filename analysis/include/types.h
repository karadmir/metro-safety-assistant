#ifndef TYPES_H
#define TYPES_H

#include <opencv2/core/types.hpp>

enum class BodyPart {
    NOSE,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE
};

struct Person {
    std::map<BodyPart, cv::Point> pose;
    cv::Rect bounding_box;
    float confidence;
};

struct Train {
    bool is_present;
};

struct SafetyLine {
    std::vector<cv::Point> polygon;
};

#endif //TYPES_H
