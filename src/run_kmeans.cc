#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>

#include "data.hpp"
#include "data_handler.hpp"
#include "kmeans.hpp"

int main(int argc, char *argv[])
{
    int k;
    if (argc < 2)
    {
        k = 480;
    }else{
        k = atoi(argv[1]);
    }
    const std::string train_data = "mnist/train-images-idx3-ubyte";
    const std::string train_label = "mnist/train-labels-idx1-ubyte";
    const std::string test_data = "mnist/t10k-images-idx3-ubyte";
    const std::string test_label = "mnist/t10k-labels-idx1-ubyte";

    DataHandler dh;
    dh.read_feature_vector(train_data);
    dh.read_feature_labels(train_label);
    dh.count_classes();
    dh.split_data();

    kmeans trainer(k);
    printf("k = %d\n", k);
    trainer.set_test_data(dh.get_test_data());
    trainer.set_training_data(dh.get_training_data());
    trainer.set_validation_data(dh.get_validataion_data());
    trainer.init_clusters();
    trainer.train();
    printf("current performance: %.3lf%% k = %d\n", trainer.test(), k);
    return 0;
}