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

int main(int argc, char *argv[]) {
    std::string hello_world = "HELLO WORLD !!";
    std::cout << hello_world << std::endl;

    // student
    student A = student();
    A.id = 1;
    A.age = 26;
    A.name = "huanyuan";

    student B = student();
    B.id = 2;
    B.age = 27;
    B.name = "ling";

    max_id(&A, &B);
    std::cout << "id: " << A.id << ", age: " << A.age << ", name: " << A.name << std::endl;

    // Eigen
    Eigen::Matrix2d a;
    a << 1, 2, 3, 4;
    Eigen::MatrixXd b(2, 2);
    b << 2, 3, 1, 4;
    std::cout << "a + b = \n"
              << a + b << std::endl;

    std::vector<student> vec;
    unsigned long long start_time = 0, end_time = 0;
    unsigned long long time = 0;
    for(int i = 0; i < 2000; i++) {
        TEST_TIME(start_time);
        vec.resize(i + 1);
        TEST_TIME(end_time);
        time += end_time - start_time;
    }
    std::cout <<"\033[0;32mResize time: " << time << " ms. \033[0;39m" << std::endl;
    return 1;
}