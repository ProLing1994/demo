#ifndef _ASR_PYFEATURE_H_
#define _ASR_PYFEATURE_H_

#include "../common/feature.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

namespace ASR
{
    void *Feature_create();
    void Feature_delete(void *feature);
    int Feature_feature_time(void *feature);
    int Feature_feature_freq(void *feature);
    int Feature_check_data_length(void *feature, int data_len_samples);
    void Feature_get_mel_int_feature(void *feature, short *pdata, int data_len_samples);
    void Feature_copy_mfsc_feature_int_to(void *feature, unsigned char *feature_data);
} // namespace ASR

#ifdef __cplusplus
}
#endif

#endif // _ASR_PYFEATURE_H_