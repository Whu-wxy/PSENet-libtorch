#-------------------------------------------------
#
# Project created by QtCreator 2020-01-06T16:59:18
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestDetection
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++14

SOURCES += \
        main.cpp \
        textdectdlg.cpp \
        utils.cpp

HEADERS += \
        textdectdlg.h \
        utils.h

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


unix{
INCLUDEPATH += /home/beidou/libtorch/include/torch/csrc/api/include \
    /home/beidou/libtorch/include
DEPENDPATH += /home/beidou/libtorch/include/torch/csrc/api/include \
    /home/beidou/libtorch/include


LIBS += -L/home/beidou/libtorch/lib -lc10 -lc10_cuda \
-lcaffe2_detectron_ops_gpu \
-lcaffe2_module_test_dynamic \
-lcaffe2_nvrtc -lcaffe2_observers \
-lonnx -lonnx_proto \
-ltorch_cuda -ltorch_cpu -ltorch

# add /home/beidou/libtorch/lib in LD_LIBRARY_PATH

-INCLUDE:?warp_size@cuda@at@@YAHXZ

INCLUDEPATH += /home/beidou/opencv-3.4.6/build/include
DEPENDPATH += /home/beidou/opencv-3.4.6/build/

LIBS += -L/home/beidou/opencv-3.4.6/build/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
}

win32{
INCLUDEPATH += D:\OpenCVMinGW3.4.1\include
LIBS += D:\OpenCVMinGW3.4.1\bin\libopencv_*.dll
}

QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
