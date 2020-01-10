#ifndef UTILS_H
#define UTILS_H

#undef slots
#include "torch/torch.h"
#include "torch/jit.h"
#include "torch/nn.h"
#include "torch/script.h"
#define slots Q_SLOTS

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <time.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <QObject>
#include <QString>
#include <QImage>
#include <QFile>

using namespace torch;
using namespace std;
using namespace cv;

bool LoadImage(std::string file_name, cv::Mat &image);

bool LoadImageNetLabel(std::string file_name,
                       std::vector<std::string> &labels);

Mat textDetect(QString modelPath, QString img_path);

Mat pse(Mat label_map, vector<Mat> kernals, int c = 6);

cv::Mat QImageToMat(QImage image);

QImage MatToQImage(cv::Mat mtx);

#endif // UTILS_H
