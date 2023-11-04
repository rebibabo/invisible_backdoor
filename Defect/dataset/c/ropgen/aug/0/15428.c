static void vapic_write(void *opaque, target_phys_addr_t addr, uint64_t data,

                        unsigned int size)

{

    CPUX86State *env = cpu_single_env;

    target_phys_addr_t rom_paddr;

    VAPICROMState *s = opaque;



    cpu_synchronize_state(env);



    /*

     * The VAPIC supports two PIO-based hypercalls, both via port 0x7E.

     *  o 16-bit write access:

     *    Reports the option ROM initialization to the hypervisor. Written

     *    value is the offset of the state structure in the ROM.

     *  o 8-bit write access:

     *    Reactivates the VAPIC after a guest hibernation, i.e. after the

     *    option ROM content has been re-initialized by a guest power cycle.

     *  o 32-bit write access:

     *    Poll for pending IRQs, considering the current VAPIC state.

     */

    switch (size) {

    case 2:

        if (s->state == VAPIC_INACTIVE) {

            rom_paddr = (env->segs[R_CS].base + env->eip) & ROM_BLOCK_MASK;

            s->rom_state_paddr = rom_paddr + data;



            s->state = VAPIC_STANDBY;

        }

        if (vapic_prepare(s) < 0) {

            s->state = VAPIC_INACTIVE;

            break;

        }

        break;

    case 1:

        if (kvm_enabled()) {

            /*

             * Disable triggering instruction in ROM by writing a NOP.

             *

             * We cannot do this in TCG mode as the reported IP is not

             * accurate.

             */

            pause_all_vcpus();

            patch_byte(env, env->eip - 2, 0x66);

            patch_byte(env, env->eip - 1, 0x90);

            resume_all_vcpus();

        }



        if (s->state == VAPIC_ACTIVE) {

            break;

        }

        if (update_rom_mapping(s, env, env->eip) < 0) {

            break;

        }

        if (find_real_tpr_addr(s, env) < 0) {

            break;

        }

        vapic_enable(s, env);

        break;

    default:

    case 4:

        if (!kvm_irqchip_in_kernel()) {

            apic_poll_irq(env->apic_state);

        }

        break;

    }

}