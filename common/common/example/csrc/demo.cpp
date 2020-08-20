#include <iostream>
using namespace std;
 
int main(int argc, char* argv[]){
    int n, s=0, t=0, x=0, y=0;
    // scanf("%d", &n);
    n = atoi(argv[1]);
    int a[n];
    for(int i=0;i<n;i++){
        // scanf("%d", &a[i]);
        a[i] = atoi(argv[1 + i + 1]);
        s ^= a[i];
    }
    t = s & (-s);
    for(int i=0;i<n;i++)
        if(a[i] & t)
            x ^= a[i];
    x = min(x, x^s);
    y = max(x, x^s);
    printf("%d %d\n", x, y);
    return 0;
}
