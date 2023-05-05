#pragma once


#if !defined(CU_COMPLEX_H_)
#define CU_COMPLEX_H_


#include <math.h>       /* import fabsf, sqrt */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


typedef    struct { float x; float y; }float2;
typedef float2 cuFloatComplex;

 float cuCrealf(cuFloatComplex x)
{
    return x.x;
}

float cuCimagf(cuFloatComplex x)
{
    return x.y;
}

cuFloatComplex make_cuFloatComplex(float r, float i)
{
        cuFloatComplex res;
        res.x = r;
        res.y = i;
        return res;
}

cuFloatComplex cuConjf(cuFloatComplex x)
{
        return make_cuFloatComplex(cuCrealf(x), -cuCimagf(x));
}
cuFloatComplex cuCaddf(cuFloatComplex x, cuFloatComplex y)
{
        return make_cuFloatComplex(cuCrealf(x) + cuCrealf(y),
            cuCimagf(x) + cuCimagf(y));
}

cuFloatComplex cuCsubf(cuFloatComplex x,
        cuFloatComplex y)
{
        return make_cuFloatComplex(cuCrealf(x) - cuCrealf(y),
            cuCimagf(x) - cuCimagf(y));
}

    /* This implementation could suffer from intermediate overflow even though
     * the final result would be in range. However, various implementations do
     * not guard against this (presumably to avoid losing performance), so we
     * don't do it either to stay competitive.
     */
cuFloatComplex cuCmulf(cuFloatComplex x,
        cuFloatComplex y)
    {
        cuFloatComplex prod;
        prod = make_cuFloatComplex((cuCrealf(x) * cuCrealf(y)) -
            (cuCimagf(x) * cuCimagf(y)),
            (cuCrealf(x) * cuCimagf(y)) +
            (cuCimagf(x) * cuCrealf(y)));
        return prod;
    }

    /* This implementation guards against intermediate underflow and overflow
     * by scaling. Such guarded implementations are usually the default for
     * complex library implementations, with some also offering an unguarded,
     * faster version.
     */
cuFloatComplex cuCdivf(cuFloatComplex x,
        cuFloatComplex y)
    {
        cuFloatComplex quot;
        float s = fabsf(cuCrealf(y)) + fabsf(cuCimagf(y));
        float oos = 1.0f / s;
        float ars = cuCrealf(x) * oos;
        float ais = cuCimagf(x) * oos;
        float brs = cuCrealf(y) * oos;
        float bis = cuCimagf(y) * oos;
        s = (brs * brs) + (bis * bis);
        oos = 1.0f / s;
        quot = make_cuFloatComplex(((ars * brs) + (ais * bis)) * oos,
            ((ais * brs) - (ars * bis)) * oos);
        return quot;
    }

    /*
     * We would like to call hypotf(), but it's not available on all platforms.
     * This discrete implementation guards against intermediate underflow and
     * overflow by scaling. Otherwise we would lose half the exponent range.
     * There are various ways of doing guarded computation. For now chose the
     * simplest and fastest solution, however this may suffer from inaccuracies
     * if sqrt and division are not IEEE compliant.
     */
float cuCabsf(cuFloatComplex x)
    {
        float a = cuCrealf(x);
        float b = cuCimagf(x);
        float v, w, t;
        a = fabsf(a);
        b = fabsf(b);
        if (a > b) {
            v = a;
            w = b;
        }
        else {
            v = b;
            w = a;
        }
        t = w / v;
        t = 1.0f + t * t;
        t = v * sqrtf(t);
        if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
            t = v + w;
        }
        return t;
    }

    /* Double precision */
typedef struct { double x; double y;}double2;
    typedef double2 cuDoubleComplex;

double cuCreal(cuDoubleComplex x)
    {
        return x.x;
    }

double cuCimag(cuDoubleComplex x)
    {
        return x.y;
    }

cuDoubleComplex make_cuDoubleComplex(double r, double i)
    {
        cuDoubleComplex res;
        res.x = r;
        res.y = i;
        return res;
    }

cuDoubleComplex cuConj(cuDoubleComplex x)
    {
        return make_cuDoubleComplex(cuCreal(x), -cuCimag(x));
    }

cuDoubleComplex cuCadd(cuDoubleComplex x,
        cuDoubleComplex y)
    {
        return make_cuDoubleComplex(cuCreal(x) + cuCreal(y),
            cuCimag(x) + cuCimag(y));
    }

cuDoubleComplex cuCsub(cuDoubleComplex x,
        cuDoubleComplex y)
    {
        return make_cuDoubleComplex(cuCreal(x) - cuCreal(y),
            cuCimag(x) - cuCimag(y));
    }

    /* This implementation could suffer from intermediate overflow even though
     * the final result would be in range. However, various implementations do
     * not guard against this (presumably to avoid losing performance), so we
     * don't do it either to stay competitive.
     */
cuDoubleComplex cuCmul(cuDoubleComplex x,
        cuDoubleComplex y)
    {
        cuDoubleComplex prod;
        prod = make_cuDoubleComplex((cuCreal(x) * cuCreal(y)) -
            (cuCimag(x) * cuCimag(y)),
            (cuCreal(x) * cuCimag(y)) +
            (cuCimag(x) * cuCreal(y)));
        return prod;
    }

    /* This implementation guards against intermediate underflow and overflow
     * by scaling. Such guarded implementations are usually the default for
     * complex library implementations, with some also offering an unguarded,
     * faster version.
     */
cuDoubleComplex cuCdiv(cuDoubleComplex x,
        cuDoubleComplex y)
    {
        cuDoubleComplex quot;
        double s = (fabs(cuCreal(y))) + (fabs(cuCimag(y)));
        double oos = 1.0 / s;
        double ars = cuCreal(x) * oos;
        double ais = cuCimag(x) * oos;
        double brs = cuCreal(y) * oos;
        double bis = cuCimag(y) * oos;
        s = (brs * brs) + (bis * bis);
        oos = 1.0 / s;
        quot = make_cuDoubleComplex(((ars * brs) + (ais * bis)) * oos,
            ((ais * brs) - (ars * bis)) * oos);
        return quot;
    }

    /* This implementation guards against intermediate underflow and overflow
     * by scaling. Otherwise we would lose half the exponent range. There are
     * various ways of doing guarded computation. For now chose the simplest
     * and fastest solution, however this may suffer from inaccuracies if sqrt
     * and division are not IEEE compliant.
     */
double cuCabs(cuDoubleComplex x)
    {
        double a = cuCreal(x);
        double b = cuCimag(x);
        double v, w, t;
        a = fabs(a);
        b = fabs(b);
        if (a > b) {
            v = a;
            w = b;
        }
        else {
            v = b;
            w = a;
        }
        t = w / v;
        t = 1.0 + t * t;
        t = v * sqrt(t);
        if ((v == 0.0) ||
            (v > 1.79769313486231570e+308) || (w > 1.79769313486231570e+308)) {
            t = v + w;
        }
        return t;
    }

#if defined(__cplusplus)
}
#endif /* __cplusplus */

/* aliases */
typedef cuFloatComplex cuComplex;
cuComplex make_cuComplex(float x,
    float y)
{
    return make_cuFloatComplex(x, y);
}

/* float-to-double promotion */
cuDoubleComplex cuComplexFloatToDouble
(cuFloatComplex c)
{
    return make_cuDoubleComplex((double)cuCrealf(c), (double)cuCimagf(c));
}

cuFloatComplex cuComplexDoubleToFloat
(cuDoubleComplex c)
{
    return make_cuFloatComplex((float)cuCreal(c), (float)cuCimag(c));
}


cuComplex cuCfmaf(cuComplex x, cuComplex y, cuComplex d)
{
    float real_res;
    float imag_res;

    real_res = (cuCrealf(x) * cuCrealf(y)) + cuCrealf(d);
    imag_res = (cuCrealf(x) * cuCimagf(y)) + cuCimagf(d);

    real_res = -(cuCimagf(x) * cuCimagf(y)) + real_res;
    imag_res = (cuCimagf(x) * cuCrealf(y)) + imag_res;

    return make_cuComplex(real_res, imag_res);
}

cuDoubleComplex cuCfma(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d)
{
    double real_res;
    double imag_res;

    real_res = (cuCreal(x) * cuCreal(y)) + cuCreal(d);
    imag_res = (cuCreal(x) * cuCimag(y)) + cuCimag(d);

    real_res = -(cuCimag(x) * cuCimag(y)) + real_res;
    imag_res = (cuCimag(x) * cuCreal(y)) + imag_res;

    return make_cuDoubleComplex(real_res, imag_res);
}

#endif /* !defined(CU_COMPLEX_H_) */

