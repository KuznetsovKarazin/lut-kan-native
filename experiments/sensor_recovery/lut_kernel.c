/* Portable endpoint-inclusive scalar LUT kernels.
 * Host-compiled for numerical parity tests only; no MCU timing is claimed.
 * The caller normalizes ADC and denormalizes quantity using 4 float constants.
 * Preconditions: n >= 2, a non-null table, finite x. Clamps outside [-1,1].
 */
#include <stdint.h>

static unsigned lut_index(float x, unsigned n, float *weight) {
    if (x <= -1.0f) { *weight=0.0f; return 0; }
    if (x >= 1.0f) { *weight=1.0f; return n-2; }
    float position=(x+1.0f)*0.5f*(float)(n-1);
    unsigned index=(unsigned)position;
    if(index>n-2) index=n-2;
    *weight=position-(float)index;
    return index;
}

float lut_float32(float x,const float *table,uint16_t n,float scale,float offset){
    (void)scale; (void)offset;
    float w; unsigned j=lut_index(x,n,&w);
    return table[j]*(1.0f-w)+table[j+1]*w;
}

float lut_uint8(float x,const uint8_t *table,uint16_t n,float scale,float offset){
    float w; unsigned j=lut_index(x,n,&w);
    float a=(float)table[j]*scale+offset, b=(float)table[j+1]*scale+offset;
    return a*(1.0f-w)+b*w;
}

float lut_int16(float x,const int16_t *table,uint16_t n,float scale,float offset){
    float w; unsigned j=lut_index(x,n,&w);
    float a=(float)table[j]*scale+offset, b=(float)table[j+1]*scale+offset;
    return a*(1.0f-w)+b*w;
}
