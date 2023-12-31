static int xen_pt_cmd_reg_write(XenPCIPassthroughState *s, XenPTReg *cfg_entry,

                                uint16_t *val, uint16_t dev_value,

                                uint16_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    uint16_t writable_mask = 0;

    uint16_t throughable_mask = 0;

    uint16_t emu_mask = reg->emu_mask;



    if (s->is_virtfn) {

        emu_mask |= PCI_COMMAND_MEMORY;

    }



    /* modify emulate register */

    writable_mask = ~reg->ro_mask & valid_mask;

    cfg_entry->data = XEN_PT_MERGE_VALUE(*val, cfg_entry->data, writable_mask);



    /* create value for writing to I/O device register */

    throughable_mask = ~emu_mask & valid_mask;



    if (*val & PCI_COMMAND_INTX_DISABLE) {

        throughable_mask |= PCI_COMMAND_INTX_DISABLE;

    } else {

        if (s->machine_irq) {

            throughable_mask |= PCI_COMMAND_INTX_DISABLE;

        }

    }



    *val = XEN_PT_MERGE_VALUE(*val, dev_value, throughable_mask);



    return 0;

}
