#ifndef __TEST_UNIQUE_PTR_H__
#define __TEST_UNIQUE_PTR_H__

class ClassA{
  public:
    ClassA() {
      a = 5;
      b = 10;
      c = 0;
    };
    ~ClassA() {};

    int Add() { 
      c = a + b;
    };
    
    int printc() {
      return c;
    }

  private:
    int a;
    int b;
    int c;
};

#endif // __TEST_UNIQUE_PTR_H__