#include "main.h"
#include "student.h"

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

    return 1;
}