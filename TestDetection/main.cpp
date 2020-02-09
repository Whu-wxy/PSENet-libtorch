#include "textdectdlg.h"
#include <QApplication>
#include "utils.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
//    TextDectDlg w;
//    w.show();

    textDetect("/home/wxy/QtWork/TestDetection/psenet.pt", "/home/wxy/QtWork/TestDetection/img_8.jpg");

    return a.exec();
}
