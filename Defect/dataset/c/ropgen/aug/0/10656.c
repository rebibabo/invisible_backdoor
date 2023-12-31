void ff_llviddsp_init_x86(LLVidDSPContext *c)

{

    int cpu_flags = av_get_cpu_flags();



#if HAVE_INLINE_ASM && HAVE_7REGS && ARCH_X86_32

    if (cpu_flags & AV_CPU_FLAG_CMOV)

        c->add_median_pred = add_median_pred_cmov;

#endif



    if (ARCH_X86_32 && EXTERNAL_MMX(cpu_flags)) {

        c->add_bytes = ff_add_bytes_mmx;

    }



    if (ARCH_X86_32 && EXTERNAL_MMXEXT(cpu_flags)) {

        /* slower than cmov version on AMD */

        if (!(cpu_flags & AV_CPU_FLAG_3DNOW))

            c->add_median_pred = ff_add_median_pred_mmxext;

    }



    if (EXTERNAL_SSE2(cpu_flags)) {

        c->add_bytes       = ff_add_bytes_sse2;

        c->add_median_pred = ff_add_median_pred_sse2;

    }



    if (EXTERNAL_SSSE3(cpu_flags)) {

        c->add_left_pred = ff_add_left_pred_ssse3;

        c->add_left_pred_int16 = ff_add_left_pred_int16_ssse3;

        c->add_gradient_pred   = ff_add_gradient_pred_ssse3;

    }



    if (EXTERNAL_SSSE3_FAST(cpu_flags)) {

        c->add_left_pred = ff_add_left_pred_unaligned_ssse3;

    }



    if (EXTERNAL_SSE4(cpu_flags)) {

        c->add_left_pred_int16 = ff_add_left_pred_int16_sse4;

    }

    if (EXTERNAL_AVX2_FAST(cpu_flags)) {

        c->add_bytes       = ff_add_bytes_avx2;

        c->add_left_pred   = ff_add_left_pred_unaligned_avx2;

        c->add_gradient_pred = ff_add_gradient_pred_avx2;

    }

}
