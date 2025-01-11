#include "safety.h"
#include <opencv2/opencv.hpp>

extern "C" cv::Mat process(cv::Mat input, const std::vector<Person>& people, const Train& train, const SafetyLine& safety_line) {
    return input;
}

int main() {
    cv::Mat input = cv::imread("test.png");

    std::vector<Person> people {
        {{}, {158, 760, 156, 321}, 0.83},
        {{}, {248, 642, 101, 253}, 0.82},
        {{}, {300, 534, 94, 194}, 0.83},
        {{}, {353, 483, 52, 155}, 0.39},
        {{}, {214, 448, 80, 170}, 0.80}
    };
    SafetyLine safety_line {
        {{354, 1275}, {390, 395}, {422, 395}, {516, 1275}}
    };
    Train train {true};

    cv::Mat result = process(input, people, train, safety_line);

    cv::imshow("Result", result);
    cv::waitKey(0);

    return 0;
}