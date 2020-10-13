#include "student.h"

void max_id(student *A, student *B) {
    student temp = student();
    if (A->id < B->id) {
        temp = *A;
        *A = *B;
        *B = temp;
    }
    return;
}