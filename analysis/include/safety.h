#ifndef SAFETY_H
#define SAFETY_H

#include "types.h"

extern "C" cv::Mat process(cv::Mat input, const std::vector<Person>& people, const Train& train, const SafetyLine& safety_line);

#endif //SAFETY_H
