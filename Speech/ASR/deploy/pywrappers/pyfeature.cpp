#include "pyfeature.hpp"

namespace ASR
{
    void* Feature_create()
    {
        return new Feature();
    }

    void *Feature_create_samples(int data_len_samples, int feature_freq)
    {
        return new Feature(data_len_samples, feature_freq);
    }

    void Feature_delete(void* feature)
    {
        Feature* obj = static_cast<Feature*>(feature);
        delete obj;
    }

    int Feature_data_mat_time(void *feature)
    {
        Feature* obj = static_cast<Feature*>(feature);
        return obj->data_mat_time();
    }

    int Feature_feature_time(void *feature)
    {
        Feature* obj = static_cast<Feature*>(feature);
        return obj->feature_time();
    }

    int Feature_feature_freq(void *feature)
    {
        Feature* obj = static_cast<Feature*>(feature);
        return obj->feature_freq();
    }

    int Feature_check_data_length(void *feature, int data_len_samples)
    {
        Feature* obj = static_cast<Feature*>(feature);
        return obj->check_data_length(data_len_samples);
    }

    void Feature_get_mel_feature(void *feature, short *pdata, int data_len_samples)
    {
        Feature* obj = static_cast<Feature*>(feature);
        obj->get_mel_feature(pdata, data_len_samples);
    }

    void Feature_get_mel_int_feature(void *feature, short *pdata, int data_len_samples)
    {
        Feature* obj = static_cast<Feature*>(feature);
        obj->get_mel_int_feature(pdata, data_len_samples);
    }

    void Feature_copy_mfsc_feature_to(void *feature, float *feature_data)
    {
        Feature* obj = static_cast<Feature*>(feature);
        obj->copy_mfsc_feature_to(feature_data);
    }

    void Feature_copy_mfsc_feature_int_to(void *feature, unsigned char *feature_data)
    {
        Feature* obj = static_cast<Feature*>(feature);
        obj->copy_mfsc_feature_int_to(feature_data);
    }
} // namespace ASR