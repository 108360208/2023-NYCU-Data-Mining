#include "drand48.h"
#include <stdlib.h>  
  
#define m 0x100000000LL  
#define c 0xB16  
#define a 0x5DEECE66DLL  
using namespace std;

static unsigned long long seed = 1;  
  
double drand48(void)  
{  
    seed = (a * seed + c) & 0xFFFFFFFFFFFFLL;  
    unsigned int x = seed >> 16;  
    return  ((double)x / (double)m);  
      
}  
  
void srand48(unsigned int i)  
{  
    seed  = (((long long int)i) << 16) | rand();  
}  