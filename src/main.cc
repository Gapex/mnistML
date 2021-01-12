#include <iostream>

#include <vector>
#include <memory>

#include <fstream>
#include <cstring>

using byte = uint8_t;

#include "data.hpp"
#include "data_handler.hpp"
#include "knn.hpp"

bool is_little_endian()
{
    static unsigned data = 0b01;
    static bool res = reinterpret_cast<unsigned char *>(&data)[0] == 0 ? false : true;
    return res;
}

uint32_t convert_little_endian(uint32_t val)
{
    __asm__ volatile(
        "bswap %%edi\n"
        "movl %%edi, %%eax\n"
        : "=r"(val));
    return val;
}

template <typename T>
class guard
{
private:
    T &t;

public:
    guard(T &ref) : t(ref) {}
    ~guard()
    {
        t.close();
    }
};

void read_nbytes(std::ifstream &is, void *buf, size_t n)
{
    is.read(reinterpret_cast<std::basic_istream<char>::char_type *>(buf), n);
}

template <typename DType, typename LType>
class DataSet
{
public:
    using Sample = std::vector<DType>;
    using TaggedSample = std::tuple<Sample, LType>;

private:
    std::vector<TaggedSample> tagged_samples;
    uint32_t nrows, ncols;
    const std::string data_file;
    const std::string label_file;

public:
    DataSet(std::string data, std::string label) : data_file(std::move(data)), label_file(std::move(label))
    {
        load(data_file, label_file);
    }

    void load(const std::string &train, const std::string &label)
    {
        load_data(train);
        load_label(label);
    }

private:
    TaggedSample make_tagged_sample(LType label, Sample sample)
    {
        return std::make_tuple<Sample, LType>(
            std::move(sample),
            std::move(label));
    }

    void load_data(const std::string &filename)
    {
        std::ifstream is;
        guard<std::ifstream> g(is);
        is.open(filename, std::ifstream::binary); //读二进制文件
        uint32_t magic;
        uint32_t nimages;
        byte pixel;
        read_nbytes(is, &magic, sizeof(magic));
        read_nbytes(is, &nimages, sizeof(nimages));
        read_nbytes(is, &nrows, sizeof(nrows));
        read_nbytes(is, &ncols, sizeof(ncols));
        nimages = convert_little_endian(nimages);
        nrows = convert_little_endian(nrows);
        ncols = convert_little_endian(ncols);
        magic = convert_little_endian(magic);
        if (magic != 2051U)
        {
            throw std::logic_error("invalid magic");
        }
        uint32_t image_size = nrows * ncols;
        byte image_buffer[image_size];
        tagged_samples.reserve(nimages);
        for (uint32_t j = 0; j < nimages; ++j)
        {
            read_nbytes(is, image_buffer, image_size);
            tagged_samples.emplace_back(
                make_tagged_sample(LType(),
                                   std::vector<uint8_t>(image_buffer, image_buffer + image_size)));
        }
        printf("%s %d load data done.\n", filename.c_str(), nimages);
    }

    void load_label(const std::string &filename)
    {
        std::ifstream is;
        guard<std::ifstream> g(is);
        is.open(filename, std::ifstream::binary);
        if (is.is_open() == false)
        {
            throw std::logic_error(filename);
        }
        uint32_t magic = 0, nitems = 0;
        read_nbytes(is, &magic, sizeof(magic));
        read_nbytes(is, &nitems, sizeof(nitems));
        magic = convert_little_endian(magic);
        nitems = convert_little_endian(nitems);
        if (magic != 2049U)
        {
            throw std::logic_error("invalid magic number");
        }
        byte label;
        for (uint32_t i = 0; i < nitems; ++i)
        {
            read_nbytes(is, &label, 1);
            std::get<1>(tagged_samples[i]) = label;
        }
        printf("%s %d load labels done.\n", filename.c_str(), nitems);
    }
};

int main()
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

    knn trainer;
    trainer.set_k(60);
    trainer.set_training_data(dh.get_training_data());
    trainer.set_test_data(dh.get_test_data());
    trainer.set_validation_data(dh.get_validataion_data());
    trainer.test_performance();
    return 0;
}