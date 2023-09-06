static void patch_instruction(VAPICROMState *s, X86CPU *cpu, target_ulong ip)

{

    CPUState *cs = CPU(cpu);

    CPUX86State *env = &cpu->env;

    VAPICHandlers *handlers;

    uint8_t opcode[2];

    uint32_t imm32 = 0;

    target_ulong current_pc = 0;

    target_ulong current_cs_base = 0;

    uint32_t current_flags = 0;



    if (smp_cpus == 1) {

        handlers = &s->rom_state.up;

    } else {

        handlers = &s->rom_state.mp;

    }



    if (!kvm_enabled()) {

        cpu_get_tb_cpu_state(env, &current_pc, &current_cs_base,

                             &current_flags);

        /* Account this instruction, because we will exit the tb.

           This is the first instruction in the block. Therefore

           there is no need in restoring CPU state. */

        if (use_icount) {

            --cs->icount_decr.u16.low;

        }

    }



    pause_all_vcpus();



    cpu_memory_rw_debug(cs, ip, opcode, sizeof(opcode), 0);



    switch (opcode[0]) {

    case 0x89: /* mov r32 to r/m32 */

        patch_byte(cpu, ip, 0x50 + modrm_reg(opcode[1]));  /* push reg */

        patch_call(s, cpu, ip + 1, handlers->set_tpr);

        break;

    case 0x8b: /* mov r/m32 to r32 */

        patch_byte(cpu, ip, 0x90);

        patch_call(s, cpu, ip + 1, handlers->get_tpr[modrm_reg(opcode[1])]);

        break;

    case 0xa1: /* mov abs to eax */

        patch_call(s, cpu, ip, handlers->get_tpr[0]);

        break;

    case 0xa3: /* mov eax to abs */

        patch_call(s, cpu, ip, handlers->set_tpr_eax);

        break;

    case 0xc7: /* mov imm32, r/m32 (c7/0) */

        patch_byte(cpu, ip, 0x68);  /* push imm32 */

        cpu_memory_rw_debug(cs, ip + 6, (void *)&imm32, sizeof(imm32), 0);

        cpu_memory_rw_debug(cs, ip + 1, (void *)&imm32, sizeof(imm32), 1);

        patch_call(s, cpu, ip + 5, handlers->set_tpr);

        break;

    case 0xff: /* push r/m32 */

        patch_byte(cpu, ip, 0x50); /* push eax */

        patch_call(s, cpu, ip + 1, handlers->get_tpr_stack);

        break;

    default:

        abort();

    }



    resume_all_vcpus();



    if (!kvm_enabled()) {

        /* tb_lock will be reset when cpu_loop_exit_noexc longjmps

         * back into the cpu_exec loop. */

        tb_lock();

        tb_gen_code(cs, current_pc, current_cs_base, current_flags, 1);

        cpu_loop_exit_noexc(cs);

    }

}
