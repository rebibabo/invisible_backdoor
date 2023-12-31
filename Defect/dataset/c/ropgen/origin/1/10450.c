int x86_cpu_handle_mmu_fault(CPUState *cs, vaddr addr,

                             int is_write1, int mmu_idx)

{

    X86CPU *cpu = X86_CPU(cs);

    CPUX86State *env = &cpu->env;

    uint64_t ptep, pte;

    target_ulong pde_addr, pte_addr;

    int error_code = 0;

    int is_dirty, prot, page_size, is_write, is_user;

    hwaddr paddr;

    uint64_t rsvd_mask = PG_HI_RSVD_MASK;

    uint32_t page_offset;

    target_ulong vaddr;



    is_user = mmu_idx == MMU_USER_IDX;

#if defined(DEBUG_MMU)

    printf("MMU fault: addr=%" VADDR_PRIx " w=%d u=%d eip=" TARGET_FMT_lx "\n",

           addr, is_write1, is_user, env->eip);

#endif

    is_write = is_write1 & 1;



    if (!(env->cr[0] & CR0_PG_MASK)) {

        pte = addr;

#ifdef TARGET_X86_64

        if (!(env->hflags & HF_LMA_MASK)) {

            /* Without long mode we can only address 32bits in real mode */

            pte = (uint32_t)pte;

        }

#endif

        prot = PAGE_READ | PAGE_WRITE | PAGE_EXEC;

        page_size = 4096;

        goto do_mapping;

    }



    if (!(env->efer & MSR_EFER_NXE)) {

        rsvd_mask |= PG_NX_MASK;

    }



    if (env->cr[4] & CR4_PAE_MASK) {

        uint64_t pde, pdpe;

        target_ulong pdpe_addr;



#ifdef TARGET_X86_64

        if (env->hflags & HF_LMA_MASK) {

            uint64_t pml4e_addr, pml4e;

            int32_t sext;



            /* test virtual address sign extension */

            sext = (int64_t)addr >> 47;

            if (sext != 0 && sext != -1) {

                env->error_code = 0;

                cs->exception_index = EXCP0D_GPF;

                return 1;

            }



            pml4e_addr = ((env->cr[3] & ~0xfff) + (((addr >> 39) & 0x1ff) << 3)) &

                env->a20_mask;

            pml4e = ldq_phys(cs->as, pml4e_addr);

            if (!(pml4e & PG_PRESENT_MASK)) {

                goto do_fault;

            }

            if (pml4e & (rsvd_mask | PG_PSE_MASK)) {

                goto do_fault_rsvd;

            }

            if (!(pml4e & PG_ACCESSED_MASK)) {

                pml4e |= PG_ACCESSED_MASK;

                stl_phys_notdirty(cs->as, pml4e_addr, pml4e);

            }

            ptep = pml4e ^ PG_NX_MASK;

            pdpe_addr = ((pml4e & PG_ADDRESS_MASK) + (((addr >> 30) & 0x1ff) << 3)) &

                env->a20_mask;

            pdpe = ldq_phys(cs->as, pdpe_addr);

            if (!(pdpe & PG_PRESENT_MASK)) {

                goto do_fault;

            }

            if (pdpe & rsvd_mask) {

                goto do_fault_rsvd;

            }

            ptep &= pdpe ^ PG_NX_MASK;

            if (!(pdpe & PG_ACCESSED_MASK)) {

                pdpe |= PG_ACCESSED_MASK;

                stl_phys_notdirty(cs->as, pdpe_addr, pdpe);

            }

            if (pdpe & PG_PSE_MASK) {

                /* 1 GB page */

                page_size = 1024 * 1024 * 1024;

                pte_addr = pdpe_addr;

                pte = pdpe;

                goto do_check_protect;

            }

        } else

#endif

        {

            /* XXX: load them when cr3 is loaded ? */

            pdpe_addr = ((env->cr[3] & ~0x1f) + ((addr >> 27) & 0x18)) &

                env->a20_mask;

            pdpe = ldq_phys(cs->as, pdpe_addr);

            if (!(pdpe & PG_PRESENT_MASK)) {

                goto do_fault;

            }

            rsvd_mask |= PG_HI_USER_MASK | PG_NX_MASK;

            if (pdpe & rsvd_mask) {

                goto do_fault_rsvd;

            }

            ptep = PG_NX_MASK | PG_USER_MASK | PG_RW_MASK;

        }



        pde_addr = ((pdpe & PG_ADDRESS_MASK) + (((addr >> 21) & 0x1ff) << 3)) &

            env->a20_mask;

        pde = ldq_phys(cs->as, pde_addr);

        if (!(pde & PG_PRESENT_MASK)) {

            goto do_fault;

        }

        if (pde & rsvd_mask) {

            goto do_fault_rsvd;

        }

        ptep &= pde ^ PG_NX_MASK;

        if (pde & PG_PSE_MASK) {

            /* 2 MB page */

            page_size = 2048 * 1024;

            pte_addr = pde_addr;

            pte = pde;

            goto do_check_protect;

        }

        /* 4 KB page */

        if (!(pde & PG_ACCESSED_MASK)) {

            pde |= PG_ACCESSED_MASK;

            stl_phys_notdirty(cs->as, pde_addr, pde);

        }

        pte_addr = ((pde & PG_ADDRESS_MASK) + (((addr >> 12) & 0x1ff) << 3)) &

            env->a20_mask;

        pte = ldq_phys(cs->as, pte_addr);

        if (!(pte & PG_PRESENT_MASK)) {

            goto do_fault;

        }

        if (pte & rsvd_mask) {

            goto do_fault_rsvd;

        }

        /* combine pde and pte nx, user and rw protections */

        ptep &= pte ^ PG_NX_MASK;

        page_size = 4096;

    } else {

        uint32_t pde;



        /* page directory entry */

        pde_addr = ((env->cr[3] & ~0xfff) + ((addr >> 20) & 0xffc)) &

            env->a20_mask;

        pde = ldl_phys(cs->as, pde_addr);

        if (!(pde & PG_PRESENT_MASK)) {

            goto do_fault;

        }

        ptep = pde | PG_NX_MASK;



        /* if PSE bit is set, then we use a 4MB page */

        if ((pde & PG_PSE_MASK) && (env->cr[4] & CR4_PSE_MASK)) {

            page_size = 4096 * 1024;

            pte_addr = pde_addr;



            /* Bits 20-13 provide bits 39-32 of the address, bit 21 is reserved.

             * Leave bits 20-13 in place for setting accessed/dirty bits below.

             */

            pte = pde | ((pde & 0x1fe000) << (32 - 13));

            rsvd_mask = 0x200000;

            goto do_check_protect_pse36;

        }



        if (!(pde & PG_ACCESSED_MASK)) {

            pde |= PG_ACCESSED_MASK;

            stl_phys_notdirty(cs->as, pde_addr, pde);

        }



        /* page directory entry */

        pte_addr = ((pde & ~0xfff) + ((addr >> 10) & 0xffc)) &

            env->a20_mask;

        pte = ldl_phys(cs->as, pte_addr);

        if (!(pte & PG_PRESENT_MASK)) {

            goto do_fault;

        }

        /* combine pde and pte user and rw protections */

        ptep &= pte | PG_NX_MASK;

        page_size = 4096;

        rsvd_mask = 0;

    }



do_check_protect:

    rsvd_mask |= (page_size - 1) & PG_ADDRESS_MASK & ~PG_PSE_PAT_MASK;

do_check_protect_pse36:

    if (pte & rsvd_mask) {

        goto do_fault_rsvd;

    }

    ptep ^= PG_NX_MASK;

    if ((ptep & PG_NX_MASK) && is_write1 == 2) {

        goto do_fault_protect;

    }

    switch (mmu_idx) {

    case MMU_USER_IDX:

        if (!(ptep & PG_USER_MASK)) {

            goto do_fault_protect;

        }

        if (is_write && !(ptep & PG_RW_MASK)) {

            goto do_fault_protect;

        }

        break;



    case MMU_KSMAP_IDX:

        if (is_write1 != 2 && (ptep & PG_USER_MASK)) {

            goto do_fault_protect;

        }

        /* fall through */

    case MMU_KNOSMAP_IDX:

        if (is_write1 == 2 && (env->cr[4] & CR4_SMEP_MASK) &&

            (ptep & PG_USER_MASK)) {

            goto do_fault_protect;

        }

        if ((env->cr[0] & CR0_WP_MASK) &&

            is_write && !(ptep & PG_RW_MASK)) {

            goto do_fault_protect;

        }

        break;



    default: /* cannot happen */

        break;

    }

    is_dirty = is_write && !(pte & PG_DIRTY_MASK);

    if (!(pte & PG_ACCESSED_MASK) || is_dirty) {

        pte |= PG_ACCESSED_MASK;

        if (is_dirty) {

            pte |= PG_DIRTY_MASK;

        }

        stl_phys_notdirty(cs->as, pte_addr, pte);

    }



    /* the page can be put in the TLB */

    prot = PAGE_READ;

    if (!(ptep & PG_NX_MASK))

        prot |= PAGE_EXEC;

    if (pte & PG_DIRTY_MASK) {

        /* only set write access if already dirty... otherwise wait

           for dirty access */

        if (is_user) {

            if (ptep & PG_RW_MASK)

                prot |= PAGE_WRITE;

        } else {

            if (!(env->cr[0] & CR0_WP_MASK) ||

                (ptep & PG_RW_MASK))

                prot |= PAGE_WRITE;

        }

    }

 do_mapping:

    pte = pte & env->a20_mask;



    /* align to page_size */

    pte &= PG_ADDRESS_MASK & ~(page_size - 1);



    /* Even if 4MB pages, we map only one 4KB page in the cache to

       avoid filling it too fast */

    vaddr = addr & TARGET_PAGE_MASK;

    page_offset = vaddr & (page_size - 1);

    paddr = pte + page_offset;



    tlb_set_page(cs, vaddr, paddr, prot, mmu_idx, page_size);

    return 0;

 do_fault_rsvd:

    error_code |= PG_ERROR_RSVD_MASK;

 do_fault_protect:

    error_code |= PG_ERROR_P_MASK;

 do_fault:

    error_code |= (is_write << PG_ERROR_W_BIT);

    if (is_user)

        error_code |= PG_ERROR_U_MASK;

    if (is_write1 == 2 &&

        (((env->efer & MSR_EFER_NXE) &&

          (env->cr[4] & CR4_PAE_MASK)) ||

         (env->cr[4] & CR4_SMEP_MASK)))

        error_code |= PG_ERROR_I_D_MASK;

    if (env->intercept_exceptions & (1 << EXCP0E_PAGE)) {

        /* cr2 is not modified in case of exceptions */

        stq_phys(cs->as,

                 env->vm_vmcb + offsetof(struct vmcb, control.exit_info_2),

                 addr);

    } else {

        env->cr[2] = addr;

    }

    env->error_code = error_code;

    cs->exception_index = EXCP0E_PAGE;

    return 1;

}
