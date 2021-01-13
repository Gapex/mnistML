#include "kmeans.hpp"

kmeans::kmeans() : test_data(nullptr), training_data(nullptr), validation_data(nullptr)
{
}

void free_feature_vector(std::vector<Data *> *ptr)
{
    if (ptr)
    {
        for (Data *data : *ptr)
        {
            delete data;
        }
    }
}

kmeans::~kmeans()
{
    free_feature_vector(training_data);
    free_feature_vector(test_data);
    free_feature_vector(validation_data);
}

void kmeans::set_training_data(std::vector<Data *> *vec)
{
    free_feature_vector(training_data);
    training_data = vec;
}
void kmeans::set_test_data(std::vector<Data *> *vec)
{
    free_feature_vector(test_data);
    test_data = vec;
}
void kmeans::set_validation_data(std::vector<Data *> *vec)
{
    free_feature_vector(validation_data);
    validation_data = vec;
}

std::vector<Data *> *kmeans::get_training_data()
{
    return training_data;
}

std::vector<Data *> *kmeans::get_test_data()
{
    return test_data;
}

std::vector<Data *> *kmeans::get_validation_data()
{
    return validation_data;
}

void kmeans::train()
{
    uint32_t ndimension = training_data->front()->get_feature_vector_size();
    for (Data *data : *training_data)
    {
        uint8_t label = data->get_label();
        if (label_to_focus[label] == nullptr)
        {
            label_to_focus[label] = data;
        }else{
            label_to_focus[label] = 
        }

    }
}

uint8_t kmeans::predict(Data *query)
{
}

double kmeans::test_performance()
{
}
double kmeans::validation_performance()
{
}