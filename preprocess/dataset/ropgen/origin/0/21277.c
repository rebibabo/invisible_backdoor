static int xen_pt_msgctrl_reg_write(XenPCIPassthroughState *s,

                                    XenPTReg *cfg_entry, uint16_t *val,

                                    uint16_t dev_value, uint16_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    XenPTMSI *msi = s->msi;

    uint16_t writable_mask = 0;

    uint16_t throughable_mask = get_throughable_mask(s, reg, valid_mask);



    /* Currently no support for multi-vector */

    if (*val & PCI_MSI_FLAGS_QSIZE) {

        XEN_PT_WARN(&s->dev, "Tries to set more than 1 vector ctrl %x\n", *val);

    }



    /* modify emulate register */

    writable_mask = reg->emu_mask & ~reg->ro_mask & valid_mask;

    cfg_entry->data = XEN_PT_MERGE_VALUE(*val, cfg_entry->data, writable_mask);

    msi->flags |= cfg_entry->data & ~PCI_MSI_FLAGS_ENABLE;



    /* create value for writing to I/O device register */

    *val = XEN_PT_MERGE_VALUE(*val, dev_value, throughable_mask);



    /* update MSI */

    if (*val & PCI_MSI_FLAGS_ENABLE) {

        /* setup MSI pirq for the first time */

        if (!msi->initialized) {

            /* Init physical one */

            XEN_PT_LOG(&s->dev, "setup MSI (register: %x).\n", *val);

            if (xen_pt_msi_setup(s)) {

                /* We do not broadcast the error to the framework code, so

                 * that MSI errors are contained in MSI emulation code and

                 * QEMU can go on running.

                 * Guest MSI would be actually not working.

                 */

                *val &= ~PCI_MSI_FLAGS_ENABLE;

                XEN_PT_WARN(&s->dev, "Can not map MSI (register: %x)!\n", *val);

                return 0;

            }

            if (xen_pt_msi_update(s)) {

                *val &= ~PCI_MSI_FLAGS_ENABLE;

                XEN_PT_WARN(&s->dev, "Can not bind MSI (register: %x)!\n", *val);

                return 0;

            }

            msi->initialized = true;

            msi->mapped = true;

        }

        msi->flags |= PCI_MSI_FLAGS_ENABLE;

    } else if (msi->mapped) {

        xen_pt_msi_disable(s);

    }



    return 0;

}
