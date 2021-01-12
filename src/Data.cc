#include "data.hpp"

Data::Data()
{
    feature_vector = new std::vector<uint8_t>();
}

Data::~Data(){

}

void Data::set_feature_vector(std::vector<uint8_t> *vec)
{
    this->feature_vector = vec;
}

void Data::append_to_feature_vector(uint8_t val)
{
    this->feature_vector->push_back(val);
}

void Data::set_label(uint8_t val)
{
    this->label = val;
}

void Data::set_enumerated_label(int val)
{
    this->enum_label = val;
}

int Data::get_feature_vector_size()
{
    return this->feature_vector->size();
}

uint8_t Data::get_label()
{
    return this->label;
}

uint8_t Data::get_enumerated_label()
{
    return this->enum_label;
}

std::vector<uint8_t> *Data::get_feature_vector()
{
    return this->feature_vector;
}