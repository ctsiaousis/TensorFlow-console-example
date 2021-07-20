#include "TensorFlow.h"

TensorFlow::TensorFlow()
{
    qDebug() << "TensorFlow::TensorFlow - Hello from TensorFlow C library version"  << TF_Version();
}


void TensorFlow::DeallocateTensor(void *data, std::size_t s, void * arg) {
    std::free(data);
    qDebug() << "TensorFlow::DeallocateTensor - size" << s ;
    qDebug() << "TensorFlow::DeallocateTensor - arg" << *static_cast<QString*>(arg);
}

int TensorFlow::createTensor()
{
    const std::vector<std::int64_t> dims = {1, 5, 12};
    const auto data_size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<std::int64_t>{});

    auto data = static_cast<float*>(std::malloc(data_size));

    std::copy(vals.begin(), vals.end(), data); // init input_vals.

    QString dealArg = "This is an argument for the deallocation function";
    auto tensor = TF_NewTensor(TF_FLOAT,
                               dims.data(), static_cast<int>(dims.size()),
                               data, data_size,
                               DeallocateTensor, &dealArg);

    // The lambda will be executed right before your function returns
    auto cleanup = qScopeGuard([=]{
        qDebug() << "TensorFlow::createTensor - Calling TF_DeleteTensor";
        TF_DeleteTensor(tensor);
    });

    if (tensor == nullptr) {
      qDebug() << "TensorFlow::createTensor - Wrong creat tensor";
      return 1;
    }

    if (TF_TensorType(tensor) != TF_FLOAT) {
      qDebug() << "TensorFlow::createTensor - Wrong tensor type";
      return 2;
    }

    if (TF_NumDims(tensor) != static_cast<int>(dims.size())) {
      qDebug() << "TensorFlow::createTensor - Wrong number of dimensions";
      return 3;
    }

    for (std::size_t i = 0; i < dims.size(); ++i) {
      if (TF_Dim(tensor, static_cast<int>(i)) != dims[i]) {
        qDebug() << "TensorFlow::createTensor - Wrong dimension size for dim: " << i;
        return 4;
      }
    }

    if (TF_TensorByteSize(tensor) != data_size) {
      qDebug() << "TensorFlow::createTensor - Wrong tensor byte size";
      return 5;
    }

    auto tensor_data = static_cast<float*>(TF_TensorData(tensor));

    if (tensor_data == nullptr) {
      qDebug() << "TensorFlow::createTensor - Wrong data tensor";
      return 6;
    }

    for (std::size_t i = 0; i < vals.size(); ++i) {
      if (tensor_data[i] != vals[i]) {
        qDebug() << "TensorFlow::createTensor - Element: " << i << " does not match";
        return 7;
      }
    }

    qDebug() << "TensorFlow::createTensor - Success creating tensor";
    return 0;
}
