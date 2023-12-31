tcg_target_ulong tcg_qemu_tb_exec(CPUArchState *cpustate, uint8_t *tb_ptr)

{

    tcg_target_ulong next_tb = 0;



    env = cpustate;

    tci_reg[TCG_AREG0] = (tcg_target_ulong)env;

    assert(tb_ptr);



    for (;;) {

#if defined(GETPC)

        tci_tb_ptr = (uintptr_t)tb_ptr;

#endif

        TCGOpcode opc = tb_ptr[0];

#if !defined(NDEBUG)

        uint8_t op_size = tb_ptr[1];

        uint8_t *old_code_ptr = tb_ptr;

#endif

        tcg_target_ulong t0;

        tcg_target_ulong t1;

        tcg_target_ulong t2;

        tcg_target_ulong label;

        TCGCond condition;

        target_ulong taddr;

#ifndef CONFIG_SOFTMMU

        tcg_target_ulong host_addr;

#endif

        uint8_t tmp8;

        uint16_t tmp16;

        uint32_t tmp32;

        uint64_t tmp64;

#if TCG_TARGET_REG_BITS == 32

        uint64_t v64;

#endif



        /* Skip opcode and size entry. */

        tb_ptr += 2;



        switch (opc) {

        case INDEX_op_end:

        case INDEX_op_nop:

            break;

        case INDEX_op_nop1:

        case INDEX_op_nop2:

        case INDEX_op_nop3:

        case INDEX_op_nopn:

        case INDEX_op_discard:

            TODO();

            break;

        case INDEX_op_set_label:

            TODO();

            break;

        case INDEX_op_call:

            t0 = tci_read_ri(&tb_ptr);

#if TCG_TARGET_REG_BITS == 32

            tmp64 = ((helper_function)t0)(tci_read_reg(TCG_REG_R0),

                                          tci_read_reg(TCG_REG_R1),

                                          tci_read_reg(TCG_REG_R2),

                                          tci_read_reg(TCG_REG_R3),

                                          tci_read_reg(TCG_REG_R5),

                                          tci_read_reg(TCG_REG_R6),

                                          tci_read_reg(TCG_REG_R7),

                                          tci_read_reg(TCG_REG_R8),

                                          tci_read_reg(TCG_REG_R9),

                                          tci_read_reg(TCG_REG_R10));

            tci_write_reg(TCG_REG_R0, tmp64);

            tci_write_reg(TCG_REG_R1, tmp64 >> 32);

#else

            tmp64 = ((helper_function)t0)(tci_read_reg(TCG_REG_R0),

                                          tci_read_reg(TCG_REG_R1),

                                          tci_read_reg(TCG_REG_R2),

                                          tci_read_reg(TCG_REG_R3),

                                          tci_read_reg(TCG_REG_R5));

            tci_write_reg(TCG_REG_R0, tmp64);

#endif

            break;

        case INDEX_op_br:

            label = tci_read_label(&tb_ptr);

            assert(tb_ptr == old_code_ptr + op_size);

            tb_ptr = (uint8_t *)label;

            continue;

        case INDEX_op_setcond_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            condition = *tb_ptr++;

            tci_write_reg32(t0, tci_compare32(t1, t2, condition));

            break;

#if TCG_TARGET_REG_BITS == 32

        case INDEX_op_setcond2_i32:

            t0 = *tb_ptr++;

            tmp64 = tci_read_r64(&tb_ptr);

            v64 = tci_read_ri64(&tb_ptr);

            condition = *tb_ptr++;

            tci_write_reg32(t0, tci_compare64(tmp64, v64, condition));

            break;

#elif TCG_TARGET_REG_BITS == 64

        case INDEX_op_setcond_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            condition = *tb_ptr++;

            tci_write_reg64(t0, tci_compare64(t1, t2, condition));

            break;

#endif

        case INDEX_op_mov_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r32(&tb_ptr);

            tci_write_reg32(t0, t1);

            break;

        case INDEX_op_movi_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_i32(&tb_ptr);

            tci_write_reg32(t0, t1);

            break;



            /* Load/store operations (32 bit). */



        case INDEX_op_ld8u_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            tci_write_reg8(t0, *(uint8_t *)(t1 + t2));

            break;

        case INDEX_op_ld8s_i32:

        case INDEX_op_ld16u_i32:

            TODO();

            break;

        case INDEX_op_ld16s_i32:

            TODO();

            break;

        case INDEX_op_ld_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            tci_write_reg32(t0, *(uint32_t *)(t1 + t2));

            break;

        case INDEX_op_st8_i32:

            t0 = tci_read_r8(&tb_ptr);

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            *(uint8_t *)(t1 + t2) = t0;

            break;

        case INDEX_op_st16_i32:

            t0 = tci_read_r16(&tb_ptr);

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            *(uint16_t *)(t1 + t2) = t0;

            break;

        case INDEX_op_st_i32:

            t0 = tci_read_r32(&tb_ptr);

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            *(uint32_t *)(t1 + t2) = t0;

            break;



            /* Arithmetic operations (32 bit). */



        case INDEX_op_add_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 + t2);

            break;

        case INDEX_op_sub_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 - t2);

            break;

        case INDEX_op_mul_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 * t2);

            break;

#if TCG_TARGET_HAS_div_i32

        case INDEX_op_div_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, (int32_t)t1 / (int32_t)t2);

            break;

        case INDEX_op_divu_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 / t2);

            break;

        case INDEX_op_rem_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, (int32_t)t1 % (int32_t)t2);

            break;

        case INDEX_op_remu_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 % t2);

            break;

#elif TCG_TARGET_HAS_div2_i32

        case INDEX_op_div2_i32:

        case INDEX_op_divu2_i32:

            TODO();

            break;

#endif

        case INDEX_op_and_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 & t2);

            break;

        case INDEX_op_or_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 | t2);

            break;

        case INDEX_op_xor_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 ^ t2);

            break;



            /* Shift/rotate operations (32 bit). */



        case INDEX_op_shl_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 << t2);

            break;

        case INDEX_op_shr_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, t1 >> t2);

            break;

        case INDEX_op_sar_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, ((int32_t)t1 >> t2));

            break;

#if TCG_TARGET_HAS_rot_i32

        case INDEX_op_rotl_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, (t1 << t2) | (t1 >> (32 - t2)));

            break;

        case INDEX_op_rotr_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_ri32(&tb_ptr);

            t2 = tci_read_ri32(&tb_ptr);

            tci_write_reg32(t0, (t1 >> t2) | (t1 << (32 - t2)));

            break;

#endif

#if TCG_TARGET_HAS_deposit_i32

        case INDEX_op_deposit_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r32(&tb_ptr);

            t2 = tci_read_r32(&tb_ptr);

            tmp16 = *tb_ptr++;

            tmp8 = *tb_ptr++;

            tmp32 = (((1 << tmp8) - 1) << tmp16);

            tci_write_reg32(t0, (t1 & ~tmp32) | ((t2 << tmp16) & tmp32));

            break;

#endif

        case INDEX_op_brcond_i32:

            t0 = tci_read_r32(&tb_ptr);

            t1 = tci_read_ri32(&tb_ptr);

            condition = *tb_ptr++;

            label = tci_read_label(&tb_ptr);

            if (tci_compare32(t0, t1, condition)) {

                assert(tb_ptr == old_code_ptr + op_size);

                tb_ptr = (uint8_t *)label;

                continue;

            }

            break;

#if TCG_TARGET_REG_BITS == 32

        case INDEX_op_add2_i32:

            t0 = *tb_ptr++;

            t1 = *tb_ptr++;

            tmp64 = tci_read_r64(&tb_ptr);

            tmp64 += tci_read_r64(&tb_ptr);

            tci_write_reg64(t1, t0, tmp64);

            break;

        case INDEX_op_sub2_i32:

            t0 = *tb_ptr++;

            t1 = *tb_ptr++;

            tmp64 = tci_read_r64(&tb_ptr);

            tmp64 -= tci_read_r64(&tb_ptr);

            tci_write_reg64(t1, t0, tmp64);

            break;

        case INDEX_op_brcond2_i32:

            tmp64 = tci_read_r64(&tb_ptr);

            v64 = tci_read_ri64(&tb_ptr);

            condition = *tb_ptr++;

            label = tci_read_label(&tb_ptr);

            if (tci_compare64(tmp64, v64, condition)) {

                assert(tb_ptr == old_code_ptr + op_size);

                tb_ptr = (uint8_t *)label;

                continue;

            }

            break;

        case INDEX_op_mulu2_i32:

            t0 = *tb_ptr++;

            t1 = *tb_ptr++;

            t2 = tci_read_r32(&tb_ptr);

            tmp64 = tci_read_r32(&tb_ptr);

            tci_write_reg64(t1, t0, t2 * tmp64);

            break;

#endif /* TCG_TARGET_REG_BITS == 32 */

#if TCG_TARGET_HAS_ext8s_i32

        case INDEX_op_ext8s_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r8s(&tb_ptr);

            tci_write_reg32(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_ext16s_i32

        case INDEX_op_ext16s_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r16s(&tb_ptr);

            tci_write_reg32(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_ext8u_i32

        case INDEX_op_ext8u_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r8(&tb_ptr);

            tci_write_reg32(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_ext16u_i32

        case INDEX_op_ext16u_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r16(&tb_ptr);

            tci_write_reg32(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_bswap16_i32

        case INDEX_op_bswap16_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r16(&tb_ptr);

            tci_write_reg32(t0, bswap16(t1));

            break;

#endif

#if TCG_TARGET_HAS_bswap32_i32

        case INDEX_op_bswap32_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r32(&tb_ptr);

            tci_write_reg32(t0, bswap32(t1));

            break;

#endif

#if TCG_TARGET_HAS_not_i32

        case INDEX_op_not_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r32(&tb_ptr);

            tci_write_reg32(t0, ~t1);

            break;

#endif

#if TCG_TARGET_HAS_neg_i32

        case INDEX_op_neg_i32:

            t0 = *tb_ptr++;

            t1 = tci_read_r32(&tb_ptr);

            tci_write_reg32(t0, -t1);

            break;

#endif

#if TCG_TARGET_REG_BITS == 64

        case INDEX_op_mov_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r64(&tb_ptr);

            tci_write_reg64(t0, t1);

            break;

        case INDEX_op_movi_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_i64(&tb_ptr);

            tci_write_reg64(t0, t1);

            break;



            /* Load/store operations (64 bit). */



        case INDEX_op_ld8u_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            tci_write_reg8(t0, *(uint8_t *)(t1 + t2));

            break;

        case INDEX_op_ld8s_i64:

        case INDEX_op_ld16u_i64:

        case INDEX_op_ld16s_i64:

            TODO();

            break;

        case INDEX_op_ld32u_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            tci_write_reg32(t0, *(uint32_t *)(t1 + t2));

            break;

        case INDEX_op_ld32s_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            tci_write_reg32s(t0, *(int32_t *)(t1 + t2));

            break;

        case INDEX_op_ld_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            tci_write_reg64(t0, *(uint64_t *)(t1 + t2));

            break;

        case INDEX_op_st8_i64:

            t0 = tci_read_r8(&tb_ptr);

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            *(uint8_t *)(t1 + t2) = t0;

            break;

        case INDEX_op_st16_i64:

            t0 = tci_read_r16(&tb_ptr);

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            *(uint16_t *)(t1 + t2) = t0;

            break;

        case INDEX_op_st32_i64:

            t0 = tci_read_r32(&tb_ptr);

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            *(uint32_t *)(t1 + t2) = t0;

            break;

        case INDEX_op_st_i64:

            t0 = tci_read_r64(&tb_ptr);

            t1 = tci_read_r(&tb_ptr);

            t2 = tci_read_i32(&tb_ptr);

            *(uint64_t *)(t1 + t2) = t0;

            break;



            /* Arithmetic operations (64 bit). */



        case INDEX_op_add_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, t1 + t2);

            break;

        case INDEX_op_sub_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, t1 - t2);

            break;

        case INDEX_op_mul_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, t1 * t2);

            break;

#if TCG_TARGET_HAS_div_i64

        case INDEX_op_div_i64:

        case INDEX_op_divu_i64:

        case INDEX_op_rem_i64:

        case INDEX_op_remu_i64:

            TODO();

            break;

#elif TCG_TARGET_HAS_div2_i64

        case INDEX_op_div2_i64:

        case INDEX_op_divu2_i64:

            TODO();

            break;

#endif

        case INDEX_op_and_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, t1 & t2);

            break;

        case INDEX_op_or_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, t1 | t2);

            break;

        case INDEX_op_xor_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, t1 ^ t2);

            break;



            /* Shift/rotate operations (64 bit). */



        case INDEX_op_shl_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, t1 << t2);

            break;

        case INDEX_op_shr_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, t1 >> t2);

            break;

        case INDEX_op_sar_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_ri64(&tb_ptr);

            t2 = tci_read_ri64(&tb_ptr);

            tci_write_reg64(t0, ((int64_t)t1 >> t2));

            break;

#if TCG_TARGET_HAS_rot_i64

        case INDEX_op_rotl_i64:

        case INDEX_op_rotr_i64:

            TODO();

            break;

#endif

#if TCG_TARGET_HAS_deposit_i64

        case INDEX_op_deposit_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r64(&tb_ptr);

            t2 = tci_read_r64(&tb_ptr);

            tmp16 = *tb_ptr++;

            tmp8 = *tb_ptr++;

            tmp64 = (((1ULL << tmp8) - 1) << tmp16);

            tci_write_reg64(t0, (t1 & ~tmp64) | ((t2 << tmp16) & tmp64));

            break;

#endif

        case INDEX_op_brcond_i64:

            t0 = tci_read_r64(&tb_ptr);

            t1 = tci_read_ri64(&tb_ptr);

            condition = *tb_ptr++;

            label = tci_read_label(&tb_ptr);

            if (tci_compare64(t0, t1, condition)) {

                assert(tb_ptr == old_code_ptr + op_size);

                tb_ptr = (uint8_t *)label;

                continue;

            }

            break;

#if TCG_TARGET_HAS_ext8u_i64

        case INDEX_op_ext8u_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r8(&tb_ptr);

            tci_write_reg64(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_ext8s_i64

        case INDEX_op_ext8s_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r8s(&tb_ptr);

            tci_write_reg64(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_ext16s_i64

        case INDEX_op_ext16s_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r16s(&tb_ptr);

            tci_write_reg64(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_ext16u_i64

        case INDEX_op_ext16u_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r16(&tb_ptr);

            tci_write_reg64(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_ext32s_i64

        case INDEX_op_ext32s_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r32s(&tb_ptr);

            tci_write_reg64(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_ext32u_i64

        case INDEX_op_ext32u_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r32(&tb_ptr);

            tci_write_reg64(t0, t1);

            break;

#endif

#if TCG_TARGET_HAS_bswap16_i64

        case INDEX_op_bswap16_i64:

            TODO();

            t0 = *tb_ptr++;

            t1 = tci_read_r16(&tb_ptr);

            tci_write_reg64(t0, bswap16(t1));

            break;

#endif

#if TCG_TARGET_HAS_bswap32_i64

        case INDEX_op_bswap32_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r32(&tb_ptr);

            tci_write_reg64(t0, bswap32(t1));

            break;

#endif

#if TCG_TARGET_HAS_bswap64_i64

        case INDEX_op_bswap64_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r64(&tb_ptr);

            tci_write_reg64(t0, bswap64(t1));

            break;

#endif

#if TCG_TARGET_HAS_not_i64

        case INDEX_op_not_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r64(&tb_ptr);

            tci_write_reg64(t0, ~t1);

            break;

#endif

#if TCG_TARGET_HAS_neg_i64

        case INDEX_op_neg_i64:

            t0 = *tb_ptr++;

            t1 = tci_read_r64(&tb_ptr);

            tci_write_reg64(t0, -t1);

            break;

#endif

#endif /* TCG_TARGET_REG_BITS == 64 */



            /* QEMU specific operations. */



#if TARGET_LONG_BITS > TCG_TARGET_REG_BITS

        case INDEX_op_debug_insn_start:

            TODO();

            break;

#else

        case INDEX_op_debug_insn_start:

            TODO();

            break;

#endif

        case INDEX_op_exit_tb:

            next_tb = *(uint64_t *)tb_ptr;

            goto exit;

            break;

        case INDEX_op_goto_tb:

            t0 = tci_read_i32(&tb_ptr);

            assert(tb_ptr == old_code_ptr + op_size);

            tb_ptr += (int32_t)t0;

            continue;

        case INDEX_op_qemu_ld8u:

            t0 = *tb_ptr++;

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            tmp8 = helper_ldb_mmu(env, taddr, tci_read_i(&tb_ptr));

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            tmp8 = *(uint8_t *)(host_addr + GUEST_BASE);

#endif

            tci_write_reg8(t0, tmp8);

            break;

        case INDEX_op_qemu_ld8s:

            t0 = *tb_ptr++;

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            tmp8 = helper_ldb_mmu(env, taddr, tci_read_i(&tb_ptr));

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            tmp8 = *(uint8_t *)(host_addr + GUEST_BASE);

#endif

            tci_write_reg8s(t0, tmp8);

            break;

        case INDEX_op_qemu_ld16u:

            t0 = *tb_ptr++;

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            tmp16 = helper_ldw_mmu(env, taddr, tci_read_i(&tb_ptr));

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            tmp16 = tswap16(*(uint16_t *)(host_addr + GUEST_BASE));

#endif

            tci_write_reg16(t0, tmp16);

            break;

        case INDEX_op_qemu_ld16s:

            t0 = *tb_ptr++;

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            tmp16 = helper_ldw_mmu(env, taddr, tci_read_i(&tb_ptr));

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            tmp16 = tswap16(*(uint16_t *)(host_addr + GUEST_BASE));

#endif

            tci_write_reg16s(t0, tmp16);

            break;

#if TCG_TARGET_REG_BITS == 64

        case INDEX_op_qemu_ld32u:

            t0 = *tb_ptr++;

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            tmp32 = helper_ldl_mmu(env, taddr, tci_read_i(&tb_ptr));

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            tmp32 = tswap32(*(uint32_t *)(host_addr + GUEST_BASE));

#endif

            tci_write_reg32(t0, tmp32);

            break;

        case INDEX_op_qemu_ld32s:

            t0 = *tb_ptr++;

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            tmp32 = helper_ldl_mmu(env, taddr, tci_read_i(&tb_ptr));

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            tmp32 = tswap32(*(uint32_t *)(host_addr + GUEST_BASE));

#endif

            tci_write_reg32s(t0, tmp32);

            break;

#endif /* TCG_TARGET_REG_BITS == 64 */

        case INDEX_op_qemu_ld32:

            t0 = *tb_ptr++;

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            tmp32 = helper_ldl_mmu(env, taddr, tci_read_i(&tb_ptr));

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            tmp32 = tswap32(*(uint32_t *)(host_addr + GUEST_BASE));

#endif

            tci_write_reg32(t0, tmp32);

            break;

        case INDEX_op_qemu_ld64:

            t0 = *tb_ptr++;

#if TCG_TARGET_REG_BITS == 32

            t1 = *tb_ptr++;

#endif

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            tmp64 = helper_ldq_mmu(env, taddr, tci_read_i(&tb_ptr));

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            tmp64 = tswap64(*(uint64_t *)(host_addr + GUEST_BASE));

#endif

            tci_write_reg(t0, tmp64);

#if TCG_TARGET_REG_BITS == 32

            tci_write_reg(t1, tmp64 >> 32);

#endif

            break;

        case INDEX_op_qemu_st8:

            t0 = tci_read_r8(&tb_ptr);

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            t2 = tci_read_i(&tb_ptr);

            helper_stb_mmu(env, taddr, t0, t2);

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            *(uint8_t *)(host_addr + GUEST_BASE) = t0;

#endif

            break;

        case INDEX_op_qemu_st16:

            t0 = tci_read_r16(&tb_ptr);

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            t2 = tci_read_i(&tb_ptr);

            helper_stw_mmu(env, taddr, t0, t2);

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            *(uint16_t *)(host_addr + GUEST_BASE) = tswap16(t0);

#endif

            break;

        case INDEX_op_qemu_st32:

            t0 = tci_read_r32(&tb_ptr);

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            t2 = tci_read_i(&tb_ptr);

            helper_stl_mmu(env, taddr, t0, t2);

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            *(uint32_t *)(host_addr + GUEST_BASE) = tswap32(t0);

#endif

            break;

        case INDEX_op_qemu_st64:

            tmp64 = tci_read_r64(&tb_ptr);

            taddr = tci_read_ulong(&tb_ptr);

#ifdef CONFIG_SOFTMMU

            t2 = tci_read_i(&tb_ptr);

            helper_stq_mmu(env, taddr, tmp64, t2);

#else

            host_addr = (tcg_target_ulong)taddr;

            assert(taddr == host_addr);

            *(uint64_t *)(host_addr + GUEST_BASE) = tswap64(tmp64);

#endif

            break;

        default:

            TODO();

            break;

        }

        assert(tb_ptr == old_code_ptr + op_size);

    }

exit:

    return next_tb;

}
