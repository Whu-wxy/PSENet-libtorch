#include "utils.h"


bool LoadImage(std::string file_name, cv::Mat &image)
{
    image = cv::imread(file_name);  // CV_8UC3
    if (image.empty() || !image.data)
        return false;

    cv::cvtColor(image, image, CV_BGR2RGB);
    std::cout << "== image size: " << image.size() << " ==" << std::endl;

    // scale image to fit
//    double scale_factor = 1;
//    if(image.rows >= image.cols)
//    {
//        scale_factor = 2240.0 / image.rows;
//        int shortEdge = scale_factor*image.cols;
//        cv::resize(image, image, Size(shortEdge, 224));
//    }
//    else
//    {
//        scale_factor = 2240.0 / image.cols;
//        int shortEdge = scale_factor*image.rows;
//        cv::resize(image, image, Size(224, shortEdge));
//    }

    std::cout << "== resize size: " << image.size() << " ==" << std::endl;

    resize(image, image, Size(image.cols*2, image.rows*2));
    // convert [unsigned int] to [float]
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
    //image.convertTo(image, CV_32FC3);

    return true;
}

Mat textDetect(QString modelPath, QString img_path)
{
    clock_t start,finish;
    double totaltime;
    start=clock();

    cv::Mat mat2Draw = cv::imread(img_path.toStdString());  // CV_8UC3
    if (mat2Draw.empty() || !mat2Draw.data)
        return mat2Draw;

    cv::Mat image;

    torch::jit::script::Module module = torch::jit::load(modelPath.toStdString());

    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"模型加载时间:"<<totaltime<<"秒"<<endl;

//    // to GPU
//    // module.to(at::kCUDA);

    // assert(module != nullptr);
    std::cout << "== ResNet50 loaded!\n";

    if (LoadImage(img_path.toStdString(), image))
    {
        if(image.data == NULL)
            return image;

        auto input_tensor = torch::from_blob(
                    image.data, {1, image.rows, image.cols, 3});
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        cout<<"input tensor size:"<<input_tensor.sizes()<<endl;
        //norm
        input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
        input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
        input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

        // to GPU
        //  input_tensor = input_tensor.to(at::kCUDA);

        start=clock();

        torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();
        out_tensor = out_tensor[0];
cout<<"out tensor size:"<<out_tensor.sizes()<<endl;

        finish=clock();
        totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
        cout<<"运行时间:"<<totaltime<<"秒"<<endl;
      //  cout<<"out tensor:"<<out_tensor.sizes()<<endl;


        //tensor---->Mat
        out_tensor = out_tensor.squeeze().detach().permute({1, 2, 0});
        out_tensor = out_tensor.sum(2);
        out_tensor = torch::softmax(out_tensor, 0);
//        out_tensor = out_tensor.squeeze().detach();
//        out_tensor = out_tensor[0];
//       // out_tensor = out_tensor.permute({1, 2, 0});
//        out_tensor = torch::softmax(out_tensor, 0);

        //see tip3，tip4
        out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
        out_tensor = out_tensor.to(torch::kCPU);
        cv::Mat resultImg(image.rows, image.cols, CV_8UC1);
        //copy the data from out_tensor to resultImg
        std::memcpy((void *) resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());
//        for(int i=0; i<resultImg.rows; i++)
//        {
//            for(int j=0; j<resultImg.cols; j++)
//            {
//                int value = resultImg.at<uchar>(i, j);
//                if(value >= 15)
//                    resultImg.at<uchar>(i, j) = 255;
//                else
//                    resultImg.at<uchar>(i, j) = 0;
//            }
//        }

        //nccomps = connectedComponent(img_gray, labels,stats,centroids);
        imshow("res", resultImg);

//        auto results = out_tensor.sort(-1, true);
//        auto softmaxs = std::get<0>(results)[0].softmax(0);
//        auto indexs = std::get<1>(results)[0];

//        std::string topPred = "";
//        for (int i = 0; i < kTOP_K; ++i)
//        {
//            auto idx = indexs[i].item<int>();
//            if(i==0)
//                topPred = labels[idx];
//            std::cout << "    ============= Top-" << i + 1
//                      << " =============" << std::endl;
//            std::cout << "    Label:  " << labels[idx] << std::endl;
//            std::cout << "    With Probability:  "
//                      << softmaxs[i].item<float>() * 100.0f << "%" << std::endl;
//        }

//        if(topPred != "")
//            cv::putText(mat2Draw, topPred, cv::Point(50, 50), 1, 2, cv::Scalar(0, 255, 0), 2);

    }
    else
    {
        std::cout << "Can't load the image, please check your path." << std::endl;
    }

    return mat2Draw;
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
