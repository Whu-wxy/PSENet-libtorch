#include "utils.h"

#define MIN_AREA 3
#define THRELD 30
#define SCORE_FILTER 30

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
        cv::resize(image, image, Size(shortEdge, 2240));
    }
    else
    {
        scale_factor = 2240.0 / image.cols;
        int shortEdge = scale_factor*image.rows;
        cv::resize(image, image, Size(2240, shortEdge));
    }

    std::cout << "== resize size: " << image.size() << " ==" << std::endl;

    // resize(image, image, Size(image.cols*2, image.rows*2));
    // convert [unsigned int] to [float]
   image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
    // image.convertTo(image, CV_32FC3);

    return true;
}

Mat textDetect(QString modelPath, QString img_path)
{
    clock_t start,finish;
    double totaltime;
    start=clock();

    cv::Mat image;
    Mat imageCopy;

    torch::jit::script::Module module = torch::jit::load(modelPath.toStdString());
    module.eval();

    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"模型加载时间:"<<totaltime<<"秒"<<endl;

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    module.to(device);

    // assert(module != nullptr);
    std::cout << "== ResNet50 loaded!\n";

    if (LoadImage(img_path.toStdString(), image))
    {
        if(image.data == NULL)
            return image;
        image.copyTo(imageCopy);
       // imageCopy.convertTo(imageCopy, CV_32FC3, 1.0f / 255.0f);


        int width = image.cols;
        int height = image.rows;
        auto input_tensor = torch::from_blob(
                    image.data, {1, height, width, 3}, torch::kFloat);
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        //input_tensor = input_tensor.toType(torch::kFloat);
        cout<<"input tensor size:"<<input_tensor.sizes()<<endl;
        //norm
        input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
        input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
        input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

        input_tensor = input_tensor.to(device);

        start=clock();
        torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();
        out_tensor = out_tensor[0];
        cout<<"out tensor size:"<<out_tensor.sizes()<<endl;
        finish=clock();
        totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
        cout<<"运行时间:"<<totaltime<<"秒"<<endl;
        //tensor---->Mat
        out_tensor = out_tensor.squeeze().detach().permute({1, 2, 0});
        out_tensor = torch::softmax(out_tensor, 2);

        //see tip3，tip4
        out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
        out_tensor = out_tensor.to(torch::kCPU);

        // cv::Mat resultImg(image.rows, image.cols, CV_8UC(6));  //CV_8UC1
        cv::Mat resultImg = Mat::zeros(height, width, CV_8UC(6));
        //copy the data from out_tensor to resultImg
        memcpy((void *) resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());
        cout<<"tensor to Mat."<<endl;

        std::vector<Mat> Kernals;
        split(resultImg, Kernals);
        Mat score = Kernals.at(0);

        for(vector<Mat>::iterator k=Kernals.begin(); k!=Kernals.end();k++)
        {
            Mat kernal = *k;
            for(int i=0; i<resultImg.rows; i++)
            {
                for(int j=0; j<resultImg.cols; j++)
                {
                    int value = kernal.at<uchar>(i, j);
                    if(value >= THRELD)
                        kernal.at<uchar>(i, j) = 1;
                    else
                        kernal.at<uchar>(i, j) = 0;
                }
            }
            *k = kernal;
        }
        cout<<">threld"<<endl;

        //从最小kernel开始扩张算法
        Mat minKernel = Kernals.at(5);

        for(int i=0; i<Kernals.size(); i++)
        {
            Mat minKernel2 = Kernals.at(i);
            minKernel2.convertTo(minKernel2, CV_8UC1, 255);
            imshow("kernal", minKernel2);
            waitKey(0);
        }

        Mat labels, stats, centroids;
        cout<<"connectedComponentsWithStats"<<endl;
        int num = connectedComponentsWithStats(minKernel, labels, stats, centroids, 4); //从最小的kernel开始
        centroids.release();
        minKernel.release();

        cout<<"轮廓数:"<<num<<endl;

        QList<int> label_values;
        //去掉较小连通域
        for(int i=0; i<labels.rows; i++)
        {
            for(int j=0; j<labels.cols; j++)
            {
                int value = labels.at<int>(i, j);
                if(value == 0)
                    continue;
                if(stats.at<int>(value, CC_STAT_AREA) < MIN_AREA)
                {
                    labels.at<int>(i, j) = 0;
                }
                else
                {
                    if(!label_values.contains(value))
                        label_values.append(value);
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
        cout<<"get rects."<<endl;
        for(int i=0; i<label_values.count(); i++)
        {
            int value = label_values.at(i);
            float score_i=0.0;
            int count = 0;
            vector<Point2i> lab_pts;
            for(int m=0; m<res.rows; m++)
            {
                for(int n=0; n<res.cols; n++)
                {
                    if(res.at<uchar>(m, n) == value)
                    {
                        score_i += abs(score.at<uchar>(m, n));
                        count++;
                        lab_pts.push_back(Point(m, n));
                    }
                }
            }
            if(count == 0)
                continue;
            score_i = score_i/count;
            cout<<"score_i:"<<score_i<<endl;

            //            if(score_i < SCORE_FILTER)
            //                continue;

            RotatedRect rect = minAreaRect(lab_pts);
            lab_pts.clear();

            Point2f cornerPts[4];
            rect.points(cornerPts);//外接矩形的4个顶点
            for (int i = 0; i < 4; i++)//绘制外接矩形
            {
                line(imageCopy, cornerPts[i], cornerPts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                cout<<"cornerPts:"<<cornerPts[i]<<endl;

            }
        }
        cout<<"all finished."<<endl;

        cv::cvtColor(imageCopy, imageCopy, CV_RGB2BGR);
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
Mat pse(Mat label_map, vector<Mat> kernals, int c)
{
    if (label_map.rows==0 || label_map.cols==0)
        throw std::runtime_error("label map must have a shape of (h>0, w>0)");
    int h = label_map.rows;
    int w = label_map.cols;
    if (kernals.size() != c || kernals.at(0).rows != h || kernals.at(0).cols != w)
        throw std::runtime_error("Sn must have a shape of (c>0, h>0, w>0)");

    Mat res = Mat::zeros(h, w, CV_8UC1);

    std::queue<std::tuple<int, int, int32_t>> q, next_q;

    int nRows = label_map.rows;
    int nCols = label_map.cols;
    if (label_map.isContinuous())
    {
        nCols *= nRows;
        nRows = 1; // 如果连续，外层循环执行只需一次，以此提高执行效率
    }
    for (int i = 0; i < nRows; ++i)
    {
        uchar* p_label_map = label_map.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j)
        {
            int32_t label = p_label_map[j];
            if (label>0)
            {
                q.push(std::make_tuple(i, j, label));
                res.at<uchar>(i, j) = label;
            }
        }
    }

    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};

    for (int i = c-2; i>=0; i--)
    {
        Mat kernal = kernals.at(i);
        uchar* p_kernal = label_map.ptr<uchar>(0);
        while(!q.empty()){
            //get each queue menber in q
            auto q_n = q.front();
            q.pop();
            int y = std::get<0>(q_n);
            int x = std::get<1>(q_n);
            int32_t l = std::get<2>(q_n);
            //store the edge pixel after one expansion
            bool is_edge = true;
            for (int idx=0; idx<4; idx++)
            {
                int index_y = y + dy[idx];
                int index_x = x + dx[idx];
                if (index_y<0 || index_y>=h || index_x<0 || index_x>=w)
                    continue;
                if (!p_kernal[index_y*w+index_x] || res.at<uchar>(index_x, index_y)>0)
                    continue;
                q.push(std::make_tuple(index_y, index_x, l));
                res.at<uchar>(index_x, index_y) = l;
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
