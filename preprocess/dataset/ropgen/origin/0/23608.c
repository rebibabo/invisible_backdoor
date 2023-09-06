static inline int check_physical(CPUPPCState *env, mmu_ctx_t *ctx,

                                 target_ulong eaddr, int rw)

{

    int in_plb, ret;



    ctx->raddr = eaddr;

    ctx->prot = PAGE_READ | PAGE_EXEC;

    ret = 0;

    switch (env->mmu_model) {

    case POWERPC_MMU_32B:

    case POWERPC_MMU_601:

    case POWERPC_MMU_SOFT_6xx:

    case POWERPC_MMU_SOFT_74xx:

    case POWERPC_MMU_SOFT_4xx:

    case POWERPC_MMU_REAL:

    case POWERPC_MMU_BOOKE:

        ctx->prot |= PAGE_WRITE;

        break;

#if defined(TARGET_PPC64)

    case POWERPC_MMU_64B:

    case POWERPC_MMU_2_06:

    case POWERPC_MMU_2_06d:

        /* Real address are 60 bits long */

        ctx->raddr &= 0x0FFFFFFFFFFFFFFFULL;

        ctx->prot |= PAGE_WRITE;

        break;

#endif

    case POWERPC_MMU_SOFT_4xx_Z:

        if (unlikely(msr_pe != 0)) {

            /* 403 family add some particular protections,

             * using PBL/PBU registers for accesses with no translation.

             */

            in_plb =

                /* Check PLB validity */

                (env->pb[0] < env->pb[1] &&

                 /* and address in plb area */

                 eaddr >= env->pb[0] && eaddr < env->pb[1]) ||

                (env->pb[2] < env->pb[3] &&

                 eaddr >= env->pb[2] && eaddr < env->pb[3]) ? 1 : 0;

            if (in_plb ^ msr_px) {

                /* Access in protected area */

                if (rw == 1) {

                    /* Access is not allowed */

                    ret = -2;

                }

            } else {

                /* Read-write access is allowed */

                ctx->prot |= PAGE_WRITE;

            }

        }

        break;

    case POWERPC_MMU_MPC8xx:

        /* XXX: TODO */

        cpu_abort(env, "MPC8xx MMU model is not implemented\n");

        break;

    case POWERPC_MMU_BOOKE206:

        cpu_abort(env, "BookE 2.06 MMU doesn't have physical real mode\n");

        break;

    default:

        cpu_abort(env, "Unknown or invalid MMU model\n");

        return -1;

    }



    return ret;

}
