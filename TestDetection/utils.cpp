#include "utils.h"
#include <QDebug>

#define MIN_AREA 5
#define THRELD 186
#define SCORE_FILTER 237

bool LoadImage(std::string file_name, cv::Mat &image)
{
    image = cv::imread(file_name);  // CV_8UC3
    if (image.empty() || !image.data)
        return false;

    cv::cvtColor(image, image, CV_BGR2RGB);
    std::cout << "== image size: " << image.size() << " ==" << std::endl;

    // scale image to fit
    double scale_factor = 1;
    if(image.rows >= image.cols)
    {
        scale_factor = 2240.0 / image.rows;
        int shortEdge = scale_factor*image.cols;
        cv::resize(image, image, Size(shortEdge, 2240), 0, 0, INTER_NEAREST);
    }
    else
    {
        scale_factor = 2240.0 / image.cols;
        int shortEdge = scale_factor*image.rows;
        cv::resize(image, image, Size(2240, shortEdge), 0, 0, INTER_NEAREST);
    }

    std::cout << "== resize size: " << image.size() << " ==" << std::endl;

    // resize(image, image, Size(image.cols*2, image.rows*2), 0, 0, INTER_NEAREST);
    // convert [unsigned int] to [float]
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);  //

    return true;
}

Mat textDetect(QString modelPath, QString img_path)
{
    clock_t start,finish;
    double totaltime;

    cv::Mat image;
    Mat imageCopy;

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
        cout<<"device: "<<"cuda";
    } else {
        device_type = torch::kCPU;
        cout<<"device: "<<"cpu";
    }
    torch::Device device(device_type);

    start=clock();
    torch::jit::script::Module module = torch::jit::load(modelPath.toStdString(), device);
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"模型加载时间:"<<totaltime<<"秒"<<endl;
    module.eval();

    // assert(module != nullptr);
    std::cout << "== ResNet50 loaded!\n";

    if (LoadImage(img_path.toStdString(), image))
    {
        if(image.data == NULL)
            return image;
        image.copyTo(imageCopy);
        //resize(imageCopy, imageCopy, Size(image.cols/2, image.rows/2), 0, 0, INTER_NEAREST);

        int width = image.cols;
        int height = image.rows;
        auto input_tensor = torch::from_blob(
                    image.data, {1, height, width, 3}, torch::kFloat);
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        //input_tensor = input_tensor.toType(torch::kFloat);
        cout<<"input tensor size:"<<input_tensor.sizes()<<endl;
        //norm
        //        input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
        //        input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
        //        input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

        input_tensor = input_tensor.to(device);

        start=clock();
        torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();
        //out_tensor = out_tensor[0];
        cout<<"out tensor size:"<<out_tensor.sizes()<<endl;
        finish=clock();
        totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
        cout<<"运行时间:"<<totaltime<<"秒"<<endl;
        //tensor---->Mat
        out_tensor = torch::sigmoid(out_tensor);
        out_tensor = out_tensor.squeeze().detach().permute({1, 2, 0});

        //see tip3，tip4
        out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);  //
        out_tensor = out_tensor.to(torch::kCPU);

        QList<Mat> Kernals;
        for(int i=0; i<6; i++)
        {
            torch::Tensor tempTensor = out_tensor.select(2, i);
            cv::Mat tempImg = Mat::zeros(height/4, width/4, CV_8UC1);
            //copy the data from out_tensor to resultImg
            memcpy((void *) tempImg.data, tempTensor.data_ptr(), sizeof(torch::kU8) * tempTensor.numel());

            resize(tempImg, tempImg, Size(width, height), 0, 0, INTER_NEAREST);

            for(int i=0; i<tempImg.rows; i++)
            {
                for(int j=0; j<tempImg.cols; j++)
                {
                    int value = tempImg.at<uchar>(i, j);
                    // kernal.at<uchar>(i, j) = value;
                    if(value >= THRELD)
                        tempImg.at<uchar>(i, j) = 255;
                    else
                        tempImg.at<uchar>(i, j) = 0;
                }
            }
            Kernals.append(tempImg);

        }
        cout<<">threld"<<endl;

        Mat score = Mat::zeros(height/4, width/4, CV_8UC1);
        torch::Tensor scoreTensor = out_tensor.select(2, 5);
        //copy the data from out_tensor to resultImg
        memcpy((void *) score.data, scoreTensor.data_ptr(), sizeof(torch::kU8) * scoreTensor.numel());
        resize(score, score, Size(width, height), 0, 0, INTER_NEAREST);

        //从最小kernel开始扩张算法
        Mat minKernel = Kernals.at(0);

        Mat labels, stats, centroids;
        cout<<"connectedComponentsWithStats"<<endl;

        int num = connectedComponentsWithStats(minKernel, labels, stats, centroids, 4, CV_16U); //从最小的kernel开始
        centroids.release();
        minKernel.release();

        cout<<"连通域数:"<<num<<endl;

        QList<int> label_values;
        //去掉较小连通域
        for(int i=0; i<labels.rows; i++)
        {
            for(int j=0; j<labels.cols; j++)
            {
                int value = labels.at<ushort>(i, j);
                if(value == 0)
                    continue;
                if(stats.at<int>(value, CC_STAT_AREA) < MIN_AREA)
                {
                    labels.at<ushort>(i, j) = 0;
                }
            }
        }
        stats.release();

        start=clock();
        Mat res = pse(labels, Kernals, 6);

        labels.release();
        finish=clock();
        totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
        cout<<"PSE时间:"<<totaltime<<"秒"<<endl;

        Kernals.clear();

start=clock();
        QMap<int, vector<Point2i>> ptMap;
        QMap<int, int> scoreMap;
        for (int y=0; y < res.rows; ++y)
            for (int x=0; x < res.cols; ++x) {
                int idx = res.at<uchar>(y, x);
                if (idx == 0) continue;

                ptMap[idx].emplace_back(cv::Point(x, y));
                scoreMap[idx] += abs(score.at<uchar>(y, x));
            }
        finish=clock();
        totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
        cout<<"get pts时间:"<<totaltime<<"秒"<<endl;
        cout<<"draw"<<endl;

        QMap<int, vector<Point2i>>::const_iterator ptIter = ptMap.constBegin();
        QMap<int, int>::const_iterator scoreIter = scoreMap.constBegin();

        while (ptIter != ptMap.constEnd()) {
            vector<Point2i> pts = ptIter.value();
            int score = scoreIter.value();
            float ave_score = score*1.0/pts.size();
            cout<<"ave score: "<<ave_score<<endl;
            if(ave_score < SCORE_FILTER)  //< SCORE_FILTER
            {
                ++ptIter;
                ++scoreIter;
                continue;
            }

            RotatedRect rect = minAreaRect(pts);
            pts.clear();

            Point2f cornerPts[4];
            rect.points(cornerPts);//外接矩形的4个顶点
            for (int i = 0; i < 4; i++)//绘制外接矩形
            {
                line(imageCopy, cornerPts[i], cornerPts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }

            ++ptIter;
            ++scoreIter;
        }

        cout<<"all finished."<<endl;

        cv::cvtColor(imageCopy, imageCopy, CV_RGB2BGR);
        resize(imageCopy, imageCopy, Size(image.cols, image.rows), 0, 0, INTER_NEAREST);

        imshow("res", imageCopy);
        waitKey(0);
    }
    else
    {
        std::cout << "Can't load the image, please check your path." << std::endl;
    }

    return imageCopy;
}

//S5->S0, small->big
Mat pse(Mat label_map, QList<Mat> kernals, int c)
{
    if (label_map.rows==0 || label_map.cols==0)
        throw std::runtime_error("label map must have a shape of (h>0, w>0)");
    int h = label_map.rows;
    int w = label_map.cols;
    if (kernals.count() != c || kernals.at(0).rows != h || kernals.at(0).cols != w)
        throw std::runtime_error("Sn must have a shape of (c>0, h>0, w>0)");

    Mat res = Mat::zeros(h, w, CV_8UC1);

    std::queue<std::tuple<int, int, ushort>> q, next_q;

    for (int i = 0; i < h; i++)
    {
        ushort* p_label_map = label_map.ptr<ushort>(i);
        for (int j = 0; j < w; j++)
        {
            ushort label = p_label_map[j];
            if (label>0)
            {
                q.push(std::make_tuple(i, j, label));
                res.at<uchar>(i, j) = label;

            }
        }
    }

    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};

    for(int i = 1; i<c; i++)
    {
        Mat kernal = kernals.at(i);

        while(!q.empty()){
            //get each queue menber in q
            auto q_n = q.front();
            q.pop();
            int y = std::get<0>(q_n);
            int x = std::get<1>(q_n);
            ushort l = std::get<2>(q_n);
            //store the edge pixel after one expansion
            bool is_edge = true;
            for (int idx=0; idx<4; idx++)
            {
                int index_y = y + dy[idx];
                int index_x = x + dx[idx];
                if (index_y<0 || index_y>=h || index_x<0 || index_x>=w)
                    continue;
                if (kernal.at<uchar>(index_y, index_x) == 0 || res.at<uchar>(index_y, index_x)>0)
                    continue;
                q.push(std::make_tuple(index_y, index_x, l));
                res.at<uchar>(index_y, index_x) = l;
                is_edge = false;
            }
            if (is_edge){
                next_q.push(std::make_tuple(y, x, l));
            }
        }
        std::swap(q, next_q);
    }
    return res;
}



cv::Mat QImageToMat(QImage image)
{
    cv::Mat mat;
    switch (image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Grayscale8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}

QImage MatToQImage(cv::Mat InputMat)
{
    cv::Mat TmpMat;

    // convert the color space to RGB
    if (InputMat.channels() == 1)

    {
        cv::cvtColor(InputMat, TmpMat, CV_GRAY2RGB);
    }

    else

    {
        cv::cvtColor(InputMat, TmpMat, CV_BGR2RGB);
    }


    // construct the QImage using the data of the mat, while do not copy the data

    QImage Result = QImage((const uchar*)(TmpMat.data), TmpMat.cols, TmpMat.rows,

                           QImage::Format_RGB888);

    // deep copy the data from mat to QImage

    Result.bits();

    return Result;

}
