#pragma once

#include "data.hpp"
#include <vector>

class knn
{
public:
    knn();
    knn(int);
    ~knn();

    void set_k(int val);
    int get_k();

    void find_knearest(Data *query_point);

    void set_training_data(std::vector<Data*> *);
    void set_test_data(std::vector<Data*> *);
    void set_validation_data(std::vector<Data*> *);

    std::vector<Data*>* get_training_data();
    std::vector<Data*>* get_test_data();
    std::vector<Data*>* get_validation_data();

    int predict();
    double calculate_distance(Data *, Data *);
    double validate_performance();
    double test_performance();



private:
    int k;
    std::vector<Data*> *neighbors;
    std::vector<Data*> *training_data;
    std::vector<Data*> *test_data;
    std::vector<Data*> *validation_data;
};