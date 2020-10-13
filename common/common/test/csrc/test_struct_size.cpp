#include <iostream>

using namespace std;

#pragma pack(1)

struct StructA {
  char a;
};

struct StructB {
  short a;
};

struct StructC {
  int a;
};

struct StructD {
  char a;
  char b;
  short c;
	int d;
};

struct StructE {
  char a;
  short c;
  char b;
	int d;
};

struct StructF {
  char a;
  char b;
  int d;
  short c;
};

int main()
{
	StructA a;
	StructB b;
	StructC c;
	StructD d;
	StructE e;
	StructF f;
	cout << sizeof(a) << endl;
	cout << sizeof(b) << endl;
	cout << sizeof(c) << endl;
	cout << sizeof(d) << endl;
	cout << sizeof(e) << endl;
	cout << sizeof(f) << endl;

	return 0;
}