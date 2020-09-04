#include "main.h"
#include "student.h"

#ifndef TEST_TIME
#include <sys/time.h>
#define TEST_TIME(times) do{\
        struct timeval cur_time;\
	    gettimeofday(&cur_time, NULL);\
	    times = (cur_time.tv_sec * 1000000llu + cur_time.tv_usec) / 1000llu;\
	}while(0)
#endif
using namespace fst;

int main() {
    std::string hello_world = "HELLO WORLD !!";
    std::cout << hello_world << std::endl;

    // student
    student student_a = student();
    student_a.id = 1;
    student_a.age = 26;
    student_a.name = "huanyuan";

    student student_b = student();
    student_b.id = 2;
    student_b.age = 27;
    student_b.name = "ling";

    max_id(&student_a, &student_b);
    std::cout << "id: " << student_a.id << ", age: " << student_a.age << ", name: " << student_a.name << std::endl;

    // Eigen
    Eigen::Matrix2d matrix_a;
    matrix_a << 1, 2, 3, 4;
    Eigen::MatrixXd matrix_b(2, 2);
    matrix_b << 2, 3, 1, 4;
    std::cout << "matrix_a + matrix_b = \n"
              << matrix_a + matrix_b << std::endl;

    // Openfst: fst
    StdVectorFst fst;
    
    // Adds state 0 to the initially empty FST and make it the start state.
    fst.AddState();   // 1st state will be state 0 (returned by AddState)
    fst.SetStart(0);  // arg is state ID
    
    // Adds two arcs exiting state 0.
    // Arc constructor args: ilabel, olabel, weight, dest state ID.
    fst.AddArc(0, StdArc(1, 1, 0.5, 1));  // 1st arg is src state ID
    fst.AddArc(0, StdArc(2, 2, 1.5, 1));
    
    // Adds state 1 and its arc.
    fst.AddState();
    fst.AddArc(1, StdArc(3, 3, 2.5, 2));
    
    // Adds state 2 and set its final weight.
    fst.AddState();
    fst.SetFinal(2, 3.5);  // 1st arg is state ID, 2nd arg weight
    
    fst.Write("binary.fst");

    // openblas
    double openblas_a[6] = {1.0,3.0,1.0,-3.0,4.0,-1.0};
    double openblas_b[6] = {1.0,4.0,1.0,-3.0,4.0,-1.0};
    double openblas_c[9] = {.5,.5,.5,1.5,.5,2.5,.5,.5,.5};

    int openblas_m = 3; // row of A and C
    int openblas_n = 3; // col of B and C
    int openblas_k = 2; // col of A and row of B
 
    double alpha = 1.0;
    double beta = 0.0;
 
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, openblas_m, openblas_n, openblas_k, 
            alpha, openblas_a, openblas_k, openblas_b, openblas_n, beta, openblas_c, openblas_n);

    std::cout << "OpenBlas result: " << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << openblas_c[i] << " ";
    }
    std::cout << std::endl;

    // alsa
    printf("\nALSA library version: %s\n", SND_LIB_VERSION_STR);
 
    printf("\nPCM stream types:\n");
    for (int val = 0; val <= SND_PCM_STREAM_LAST; val++)
        printf(" %s\n",snd_pcm_stream_name((snd_pcm_stream_t)val));
 
    printf("\nPCM access types:\n");
    for (int val = 0; val <= SND_PCM_ACCESS_LAST; val++) {
        printf(" %s\n", snd_pcm_access_name((snd_pcm_access_t)val));
    }
 
    printf("\nPCM formats:\n");
    for (int val = 0; val <= SND_PCM_FORMAT_LAST; val++) {
        if (snd_pcm_format_name((snd_pcm_format_t)val)!= NULL) {
            printf(" %s (%s)\n",
                    snd_pcm_format_name((snd_pcm_format_t)val),
                    snd_pcm_format_description((snd_pcm_format_t)val));
        }
    }
 
    printf("\nPCM subformats:\n");
    for (int val = 0; val <= SND_PCM_SUBFORMAT_LAST;val++) {
        printf(" %s (%s)\n",
                snd_pcm_subformat_name((snd_pcm_subformat_t)val),
                snd_pcm_subformat_description((
                    snd_pcm_subformat_t)val));
    }
 
    printf("\nPCM states:\n");
    for (int val = 0; val <= SND_PCM_STATE_LAST; val++)
        printf(" %s\n",snd_pcm_state_name((snd_pcm_state_t)val));
    
    std::vector<TokenList> vec;
    unsigned long long start_time = 0, end_time = 0;
    unsigned long long read_size_time = 0, resize_time = 0;
    for(int i = 0; i < 2000; i++) {
        TEST_TIME(start_time);
        int size = vec.size();
        TEST_TIME(end_time);
        read_size_time += end_time - start_time;

        TEST_TIME(start_time);
        vec.resize(size + 1);
        TEST_TIME(end_time);
        resize_time += end_time - start_time;
    }
    std::cout <<"\033[0;32mRead size time: " << read_size_time << " ms. \033[0;39m" << std::endl;
    std::cout <<"\033[0;32mResize time: " << resize_time << " ms. \033[0;39m" << std::endl;

    return 1;
}