#ifndef _STUDENT_H_
#define _STUDENT_H_

#include <string>
struct student
{   
    student():
     id(0),
     age(1),
     name("") {}

    int id;
    int age;
    std::string name;
};

void max_id(student* A, student* B);
#endif