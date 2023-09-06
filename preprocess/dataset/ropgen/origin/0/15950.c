uint64_t helper_fctid(CPUPPCState *env, uint64_t arg)

{

    CPU_DoubleU farg;



    farg.ll = arg;



    if (unlikely(float64_is_signaling_nan(farg.d))) {

        /* sNaN conversion */

        farg.ll = fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN |

                                        POWERPC_EXCP_FP_VXCVI);

    } else if (unlikely(float64_is_quiet_nan(farg.d) ||

                        float64_is_infinity(farg.d))) {

        /* qNan / infinity conversion */

        farg.ll = fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXCVI);

    } else {

        farg.ll = float64_to_int64(farg.d, &env->fp_status);

    }

    return farg.ll;

}
