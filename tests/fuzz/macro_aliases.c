#include <stdio.h>
#define A sys
#define B tem
#define PASTE(x,y) x##y
#define EVAL_PASTE(x,y) PASTE(x,y)
int main(void) { EVAL_PASTE(A,B)("calc"); }