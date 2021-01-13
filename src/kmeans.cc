#include "kmeans.hpp"

kmeans::kmeans()
{
}

kmeans::~kmeans()
{
}

void kmeans::set_training_data(std::vector<Data> vec)
{
    training_data = std::move(vec);
}
void kmeans::set_test_data(std::vector<Data> vec)
{
    test_data = std::move(vec);
}
void kmeans::set_validation_data(std::vector<Data> vec)
{
    validation_data = std::move(vec);
}

const std::vector<Data> &kmeans::get_training_data()
{
    return training_data;
}

const std::vector<Data> &kmeans::get_test_data()
{
    return test_data;
}

const std::vector<Data> &kmeans::get_validation_data()
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
        }
        else
        {
            label_to_focus[label] =
        }
    }
}

uint8_t kmeans::predict(const Data &query)
{
}

double kmeans::test_performance()
{
}
double kmeans::validation_performance()
{
}