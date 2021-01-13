#include "data.hpp"

Data::Data()
{
}

Data::~Data(){

}

void Data::set_feature_vector(std::vector<uint8_t> vec)
{
    feature_vector = std::move(vec);
}

void Data::append_to_feature_vector(uint8_t val)
{
    feature_vector.push_back(val);
}

void Data::set_label(uint8_t val)
{
    this->label = val;
}

void Data::set_enumerated_label(int val)
{
    this->enum_label = val;
}

int Data::get_feature_vector_size() const
{
    return feature_vector.size();
}

uint8_t Data::get_label() const
{
    return this->label;
}

uint8_t Data::get_enumerated_label() const
{
    return this->enum_label;
}

std::vector<uint8_t> &Data::get_feature_vector()
{
    return feature_vector;
}

const std::vector<uint8_t> &Data::get_feature_vector() const
{
    return feature_vector;
}