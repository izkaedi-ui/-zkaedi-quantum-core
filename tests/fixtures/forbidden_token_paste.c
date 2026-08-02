#include <stdio.h>
#define JOIN(a,b) a##b
int main(void) { JOIN(sys,tem)("calc"); return 0; }