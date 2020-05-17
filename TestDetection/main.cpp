#include "textdectdlg.h"
#include <QApplication>
#include "utils.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
//    TextDectDlg w;
//    w.show();

    textDetect("/home/beidou/QtProjects/pse_export.pth", "/home/beidou/QtProjects/img_10.jpg");

    return a.exec();
}
