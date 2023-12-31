void ff_imdct_calc_sse(MDCTContext *s, FFTSample *output,

                       const FFTSample *input, FFTSample *tmp)

{

    x86_reg k;

    long n8, n4, n2, n;

    const uint16_t *revtab = s->fft.revtab;

    const FFTSample *tcos = s->tcos;

    const FFTSample *tsin = s->tsin;

    const FFTSample *in1, *in2;

    FFTComplex *z = (FFTComplex *)tmp;



    n = 1 << s->nbits;

    n2 = n >> 1;

    n4 = n >> 2;

    n8 = n >> 3;



#ifdef ARCH_X86_64

    asm volatile ("movaps %0, %%xmm8\n\t"::"m"(*p1m1p1m1));

#define P1M1P1M1 "%%xmm8"

#else

#define P1M1P1M1 "%4"

#endif



    /* pre rotation */

    in1 = input;

    in2 = input + n2 - 4;



    /* Complex multiplication */

    for (k = 0; k < n4; k += 4) {

        asm volatile (

            "movaps          %0, %%xmm0 \n\t"   // xmm0 = r0 X  r1 X : in2

            "movaps          %1, %%xmm3 \n\t"   // xmm3 = X  i1 X  i0: in1

            "movaps    -16+1*%0, %%xmm4 \n\t"   // xmm4 = r0 X  r1 X : in2

            "movaps     16+1*%1, %%xmm7 \n\t"   // xmm7 = X  i1 X  i0: in1

            "movlps          %2, %%xmm1 \n\t"   // xmm1 = X  X  R1 R0: tcos

            "movlps          %3, %%xmm2 \n\t"   // xmm2 = X  X  I1 I0: tsin

            "movlps      8+1*%2, %%xmm5 \n\t"   // xmm5 = X  X  R1 R0: tcos

            "movlps      8+1*%3, %%xmm6 \n\t"   // xmm6 = X  X  I1 I0: tsin

            "shufps $95, %%xmm0, %%xmm0 \n\t"   // xmm0 = r1 r1 r0 r0

            "shufps $160,%%xmm3, %%xmm3 \n\t"   // xmm3 = i1 i1 i0 i0

            "shufps $95, %%xmm4, %%xmm4 \n\t"   // xmm4 = r1 r1 r0 r0

            "shufps $160,%%xmm7, %%xmm7 \n\t"   // xmm7 = i1 i1 i0 i0

            "unpcklps    %%xmm2, %%xmm1 \n\t"   // xmm1 = I1 R1 I0 R0

            "unpcklps    %%xmm6, %%xmm5 \n\t"   // xmm5 = I1 R1 I0 R0

            "movaps      %%xmm1, %%xmm2 \n\t"   // xmm2 = I1 R1 I0 R0

            "movaps      %%xmm5, %%xmm6 \n\t"   // xmm6 = I1 R1 I0 R0

            "xorps   "P1M1P1M1", %%xmm2 \n\t"   // xmm2 = -I1 R1 -I0 R0

            "xorps   "P1M1P1M1", %%xmm6 \n\t"   // xmm6 = -I1 R1 -I0 R0

            "mulps       %%xmm1, %%xmm0 \n\t"   // xmm0 = rI rR rI rR

            "mulps       %%xmm5, %%xmm4 \n\t"   // xmm4 = rI rR rI rR

            "shufps $177,%%xmm2, %%xmm2 \n\t"   // xmm2 = R1 -I1 R0 -I0

            "shufps $177,%%xmm6, %%xmm6 \n\t"   // xmm6 = R1 -I1 R0 -I0

            "mulps       %%xmm2, %%xmm3 \n\t"   // xmm3 = Ri -Ii Ri -Ii

            "mulps       %%xmm6, %%xmm7 \n\t"   // xmm7 = Ri -Ii Ri -Ii

            "addps       %%xmm3, %%xmm0 \n\t"   // xmm0 = result

            "addps       %%xmm7, %%xmm4 \n\t"   // xmm4 = result

            ::"m"(in2[-2*k]), "m"(in1[2*k]),

              "m"(tcos[k]), "m"(tsin[k])

#ifndef ARCH_X86_64

              ,"m"(*p1m1p1m1)

#endif

        );

        /* Should be in the same block, hack for gcc2.95 & gcc3 */

        asm (

            "movlps      %%xmm0, %0     \n\t"

            "movhps      %%xmm0, %1     \n\t"

            "movlps      %%xmm4, %2     \n\t"

            "movhps      %%xmm4, %3     \n\t"

            :"=m"(z[revtab[k]]), "=m"(z[revtab[k + 1]]),

             "=m"(z[revtab[k + 2]]), "=m"(z[revtab[k + 3]])

        );

    }



    ff_fft_calc_sse(&s->fft, z);



#ifndef ARCH_X86_64

#undef P1M1P1M1

#define P1M1P1M1 "%3"

#endif



    /* post rotation + reordering */

    for (k = 0; k < n4; k += 4) {

        asm (

            "movaps          %0, %%xmm0 \n\t"   // xmm0 = i1 r1 i0 r0: z

            "movaps     16+1*%0, %%xmm4 \n\t"   // xmm4 = i1 r1 i0 r0: z

            "movlps          %1, %%xmm1 \n\t"   // xmm1 = X  X  R1 R0: tcos

            "movlps      8+1*%1, %%xmm5 \n\t"   // xmm5 = X  X  R1 R0: tcos

            "movaps      %%xmm0, %%xmm3 \n\t"   // xmm3 = i1 r1 i0 r0

            "movaps      %%xmm4, %%xmm7 \n\t"   // xmm7 = i1 r1 i0 r0

            "movlps          %2, %%xmm2 \n\t"   // xmm2 = X  X  I1 I0: tsin

            "movlps      8+1*%2, %%xmm6 \n\t"   // xmm6 = X  X  I1 I0: tsin

            "shufps $160,%%xmm0, %%xmm0 \n\t"   // xmm0 = r1 r1 r0 r0

            "shufps $245,%%xmm3, %%xmm3 \n\t"   // xmm3 = i1 i1 i0 i0

            "shufps $160,%%xmm4, %%xmm4 \n\t"   // xmm4 = r1 r1 r0 r0

            "shufps $245,%%xmm7, %%xmm7 \n\t"   // xmm7 = i1 i1 i0 i0

            "unpcklps    %%xmm2, %%xmm1 \n\t"   // xmm1 = I1 R1 I0 R0

            "unpcklps    %%xmm6, %%xmm5 \n\t"   // xmm5 = I1 R1 I0 R0

            "movaps      %%xmm1, %%xmm2 \n\t"   // xmm2 = I1 R1 I0 R0

            "movaps      %%xmm5, %%xmm6 \n\t"   // xmm6 = I1 R1 I0 R0

            "xorps   "P1M1P1M1", %%xmm2 \n\t"   // xmm2 = -I1 R1 -I0 R0

            "mulps       %%xmm1, %%xmm0 \n\t"   // xmm0 = rI rR rI rR

            "xorps   "P1M1P1M1", %%xmm6 \n\t"   // xmm6 = -I1 R1 -I0 R0

            "mulps       %%xmm5, %%xmm4 \n\t"   // xmm4 = rI rR rI rR

            "shufps $177,%%xmm2, %%xmm2 \n\t"   // xmm2 = R1 -I1 R0 -I0

            "shufps $177,%%xmm6, %%xmm6 \n\t"   // xmm6 = R1 -I1 R0 -I0

            "mulps       %%xmm2, %%xmm3 \n\t"   // xmm3 = Ri -Ii Ri -Ii

            "mulps       %%xmm6, %%xmm7 \n\t"   // xmm7 = Ri -Ii Ri -Ii

            "addps       %%xmm3, %%xmm0 \n\t"   // xmm0 = result

            "addps       %%xmm7, %%xmm4 \n\t"   // xmm4 = result

            "movaps      %%xmm0, %0     \n\t"

            "movaps      %%xmm4, 16+1*%0\n\t"

            :"+m"(z[k])

            :"m"(tcos[k]), "m"(tsin[k])

#ifndef ARCH_X86_64

             ,"m"(*p1m1p1m1)

#endif

        );

    }



    /*

       Mnemonics:

       0 = z[k].re

       1 = z[k].im

       2 = z[k + 1].re

       3 = z[k + 1].im

       4 = z[-k - 2].re

       5 = z[-k - 2].im

       6 = z[-k - 1].re

       7 = z[-k - 1].im

    */

    k = 16-n;

    asm volatile("movaps %0, %%xmm7 \n\t"::"m"(*m1m1m1m1));

    asm volatile(

        "1: \n\t"

        "movaps  -16(%4,%0), %%xmm1 \n\t"   // xmm1 = 4 5 6 7 = z[-2-k]

        "neg %0 \n\t"

        "movaps     (%4,%0), %%xmm0 \n\t"   // xmm0 = 0 1 2 3 = z[k]

        "xorps       %%xmm7, %%xmm0 \n\t"   // xmm0 = -0 -1 -2 -3

        "movaps      %%xmm0, %%xmm2 \n\t"   // xmm2 = -0 -1 -2 -3

        "shufps $141,%%xmm1, %%xmm0 \n\t"   // xmm0 = -1 -3 4 6

        "shufps $216,%%xmm1, %%xmm2 \n\t"   // xmm2 = -0 -2 5 7

        "shufps $156,%%xmm0, %%xmm0 \n\t"   // xmm0 = -1 6 -3 4 !

        "shufps $156,%%xmm2, %%xmm2 \n\t"   // xmm2 = -0 7 -2 5 !

        "movaps      %%xmm0, (%1,%0) \n\t"  // output[2*k]

        "movaps      %%xmm2, (%2,%0) \n\t"  // output[n2+2*k]

        "neg %0 \n\t"

        "shufps $27, %%xmm0, %%xmm0 \n\t"   // xmm0 = 4 -3 6 -1

        "xorps       %%xmm7, %%xmm0 \n\t"   // xmm0 = -4 3 -6 1 !

        "shufps $27, %%xmm2, %%xmm2 \n\t"   // xmm2 = 5 -2 7 -0 !

        "movaps      %%xmm0, -16(%2,%0) \n\t" // output[n2-4-2*k]

        "movaps      %%xmm2, -16(%3,%0) \n\t" // output[n-4-2*k]

        "add $16, %0 \n\t"

        "jle 1b \n\t"

        :"+r"(k)

        :"r"(output), "r"(output+n2), "r"(output+n), "r"(z+n8)

        :"memory"

    );

}
