#include <QCoreApplication>
#include "TensorFlow.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    TensorFlow t;
    int ret = t.createTensor();
    qDebug() << "The createTensor function returned" << ret;
    return ret;
//    return a.exec(); //use this if event loop is needed
}
