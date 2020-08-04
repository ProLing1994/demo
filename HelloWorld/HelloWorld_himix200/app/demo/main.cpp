#include "main.h"
#include "student.h"

using namespace fst;

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
    return 1;
}