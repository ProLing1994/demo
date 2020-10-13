#include <iostream>

using namespace std;

int test = 5;

struct StructA
{
	int a = test;
};

struct StructB
{
	int &b = test;
};

struct StructC
{
	int a = test;
	int &b = test;
};

struct StructD
{
	StructC a;
};

struct StructE
{
	StructC a;
	StructC &b = a;
};

int main()
{
	StructA a;
	StructB b;
	StructC c;
	StructC &d = c;
	StructA *e = &a;
	StructC *f = &c;
	const StructC *g = &c;
	StructC * const h = &c;
	const StructC * const i = &c;
	const StructC *j;
	StructD k;
	StructE l;
	cout << sizeof(a) << endl;
	cout << sizeof(b) << endl;
	cout << sizeof(c) << endl;
	cout << sizeof(d) << endl;
	cout << sizeof(e) << endl;
	cout << sizeof(f) << endl;
	cout << sizeof(g) << endl;
	cout << sizeof(h) << endl;
	cout << sizeof(i) << endl;
	cout << sizeof(j) << endl;
	cout << sizeof(k) << endl;
	cout << sizeof(l) << endl;

	return 0;
}