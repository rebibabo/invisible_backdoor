static inline void compute_hflags(CPUMIPSState *env)

{

    env->hflags &= ~(MIPS_HFLAG_COP1X | MIPS_HFLAG_64 | MIPS_HFLAG_CP0 |

                     MIPS_HFLAG_F64 | MIPS_HFLAG_FPU | MIPS_HFLAG_KSU |

                     MIPS_HFLAG_UX);

    if (!(env->CP0_Status & (1 << CP0St_EXL)) &&

        !(env->CP0_Status & (1 << CP0St_ERL)) &&

        !(env->hflags & MIPS_HFLAG_DM)) {

        env->hflags |= (env->CP0_Status >> CP0St_KSU) & MIPS_HFLAG_KSU;

    }

#if defined(TARGET_MIPS64)

    if (((env->hflags & MIPS_HFLAG_KSU) != MIPS_HFLAG_UM) ||

        (env->CP0_Status & (1 << CP0St_PX)) ||

        (env->CP0_Status & (1 << CP0St_UX))) {

        env->hflags |= MIPS_HFLAG_64;

    }

    if (env->CP0_Status & (1 << CP0St_UX)) {

        env->hflags |= MIPS_HFLAG_UX;

    }

#endif

    if ((env->CP0_Status & (1 << CP0St_CU0)) ||

        !(env->hflags & MIPS_HFLAG_KSU)) {

        env->hflags |= MIPS_HFLAG_CP0;

    }

    if (env->CP0_Status & (1 << CP0St_CU1)) {

        env->hflags |= MIPS_HFLAG_FPU;

    }

    if (env->CP0_Status & (1 << CP0St_FR)) {

        env->hflags |= MIPS_HFLAG_F64;

    }

    if (env->insn_flags & ISA_MIPS32R2) {

        if (env->active_fpu.fcr0 & (1 << FCR0_F64)) {

            env->hflags |= MIPS_HFLAG_COP1X;

        }

    } else if (env->insn_flags & ISA_MIPS32) {

        if (env->hflags & MIPS_HFLAG_64) {

            env->hflags |= MIPS_HFLAG_COP1X;

        }

    } else if (env->insn_flags & ISA_MIPS4) {

        /* All supported MIPS IV CPUs use the XX (CU3) to enable

           and disable the MIPS IV extensions to the MIPS III ISA.

           Some other MIPS IV CPUs ignore the bit, so the check here

           would be too restrictive for them.  */

        if (env->CP0_Status & (1 << CP0St_CU3)) {

            env->hflags |= MIPS_HFLAG_COP1X;

        }

    }

}
