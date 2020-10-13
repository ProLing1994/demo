#include <memory>
#include <iostream>
#include "test_unique_ptr.hpp"

int func_unique_ptr(void **p) {
  // wronge
  // std::unique_ptr<ClassA> a;      
  // a.reset(new ClassA());
  // a->Add();
  // *p = static_cast<void *>(&a);
  // std::cout << "c: " << a->printc() << std::endl;
  // std::unique_ptr<ClassA> *b = static_cast<std::unique_ptr<ClassA> *>(*p);
  // std::cout << "c: " << (*b)->printc() << std::endl;

  // use static
  // static std::unique_ptr<ClassA> a;
  // a.reset(new ClassA());
  // a->Add();
  // *p = static_cast<void *>(&a);
  // std::cout << "c: " << a->printc() << std::endl;
  // std::unique_ptr<ClassA> *b = static_cast<std::unique_ptr<ClassA> *>(*p);
  // std::cout << "c: " << (*b)->printc() << std::endl;

  // use release() func
  std::unique_ptr<ClassA> a;
  a.reset(new ClassA());
  a->Add();
  std::cout << "c: " << a->printc() << std::endl;
  *p = static_cast<void *>(a.release());
  ClassA *b = static_cast<ClassA *>(*p);
  std::cout << "c: " << (*b).printc() << std::endl;
  return 0;
}

int func_new(void **p) {
  *p = new ClassA();
  ClassA *a = static_cast<ClassA*>(*p);
  a->Add();
  std::cout << "c: " << a->printc() << std::endl;
  return 0;
}

int main() {
  void * p = nullptr;
  
  // // unique_ptr
  // func_unique_ptr(&p);
  // std::unique_ptr<ClassA> *c = static_cast<std::unique_ptr<ClassA> *>(p);
  // std::cout << "c: " << (*c)->printc() << std::endl;

  // unique_ptr, use release() func
  func_unique_ptr(&p);
  ClassA *c = static_cast<ClassA *>(p);
  std::cout << "c: " << (*c).printc() << std::endl;
  delete c;
  p = nullptr;
  
  // new
  func_new(&p);
  ClassA *b = static_cast<ClassA*>(p);
  std::cout << "c: " << b->printc() << std::endl;
  delete b;
  p = nullptr;
}
