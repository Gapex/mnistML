#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>

#include "data.hpp"
#include "data_handler.hpp"
#include "knn.hpp"
#include "kmeans.hpp"


int main(int argc, char * argv[])
{
    const std::string train_data = "mnist/train-images-idx3-ubyte";
    const std::string train_label = "mnist/train-labels-idx1-ubyte";
    const std::string test_data = "mnist/t10k-images-idx3-ubyte";
    const std::string test_label = "mnist/t10k-labels-idx1-ubyte";
 
    DataHandler dh;
    dh.read_feature_vector(train_data);
    dh.read_feature_labels(train_label);
    dh.count_classes();
    dh.split_data();

    // knn trainer;
    // if(argc > 1){
    //     trainer.set_k(atoi(argv[1]));
    // }
    // printf("k = %d\n", trainer.get_k());
    // trainer.set_training_data(dh.get_training_data());
    // trainer.set_test_data(dh.get_test_data());
    // trainer.set_validation_data(dh.get_validataion_data());
    // trainer.test_performance();

    kmeans trainer;
    trainer.set_test_data(dh.get_test_data());
    trainer.set_training_data(dh.get_training_data());
    trainer.set_validation_data(dh.get_validataion_data());
    trainer.train();
    trainer.test_performance();
    trainer.validation_performance();

    return 0;
}