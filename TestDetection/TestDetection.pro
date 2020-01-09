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

CONFIG += c++11

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
INCLUDEPATH += /home/wxy/libtorch/include/torch/csrc/api/include \
    /home/wxy/libtorch/include
DEPENDPATH += /home/wxy/libtorch/include/torch/csrc/api/include \
    /home/wxy/libtorch/include

LIBS += -L/home/wxy/libtorch/lib -lc10 \
-lcaffe2_detectron_ops \
-lcaffe2_module_test_dynamic \
-lclog -lcpuinfo \
-lonnx -lonnx_proto \
-ltorch

INCLUDEPATH += /home/wxy/opencv-3.4.1/build/include
DEPENDPATH += /home/wxy/opencv-3.4.1/build/

LIBS += -L/home/wxy/opencv-3.4.1/build/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
}

win32{
INCLUDEPATH += D:\OpenCVMinGW3.4.1\include
LIBS += D:\OpenCVMinGW3.4.1\bin\libopencv_*.dll
}
