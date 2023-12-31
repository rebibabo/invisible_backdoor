static int xen_pt_initfn(PCIDevice *d)

{

    XenPCIPassthroughState *s = XEN_PT_DEVICE(d);

    int rc = 0;

    uint8_t machine_irq = 0, scratch;

    uint16_t cmd = 0;

    int pirq = XEN_PT_UNASSIGNED_PIRQ;



    /* register real device */

    XEN_PT_LOG(d, "Assigning real physical device %02x:%02x.%d"

               " to devfn %#x\n",

               s->hostaddr.bus, s->hostaddr.slot, s->hostaddr.function,

               s->dev.devfn);



    rc = xen_host_pci_device_get(&s->real_device,

                                 s->hostaddr.domain, s->hostaddr.bus,

                                 s->hostaddr.slot, s->hostaddr.function);

    if (rc) {

        XEN_PT_ERR(d, "Failed to \"open\" the real pci device. rc: %i\n", rc);

        return -1;

    }



    s->is_virtfn = s->real_device.is_virtfn;

    if (s->is_virtfn) {

        XEN_PT_LOG(d, "%04x:%02x:%02x.%d is a SR-IOV Virtual Function\n",

                   s->real_device.domain, s->real_device.bus,

                   s->real_device.dev, s->real_device.func);

    }



    /* Initialize virtualized PCI configuration (Extended 256 Bytes) */

    if (xen_host_pci_get_block(&s->real_device, 0, d->config,

                               PCI_CONFIG_SPACE_SIZE) < 0) {

        xen_host_pci_device_put(&s->real_device);

        return -1;

    }



    s->memory_listener = xen_pt_memory_listener;

    s->io_listener = xen_pt_io_listener;



    /* Setup VGA bios for passthrough GFX */

    if ((s->real_device.domain == 0) && (s->real_device.bus == 0) &&

        (s->real_device.dev == 2) && (s->real_device.func == 0)) {

        if (!is_igd_vga_passthrough(&s->real_device)) {

            XEN_PT_ERR(d, "Need to enable igd-passthru if you're trying"

                       " to passthrough IGD GFX.\n");

            xen_host_pci_device_put(&s->real_device);

            return -1;

        }



        if (xen_pt_setup_vga(s, &s->real_device) < 0) {

            XEN_PT_ERR(d, "Setup VGA BIOS of passthrough GFX failed!\n");

            xen_host_pci_device_put(&s->real_device);

            return -1;

        }



        /* Register ISA bridge for passthrough GFX. */

        xen_igd_passthrough_isa_bridge_create(s, &s->real_device);

    }



    /* Handle real device's MMIO/PIO BARs */

    xen_pt_register_regions(s, &cmd);



    /* reinitialize each config register to be emulated */

    if (xen_pt_config_init(s)) {

        XEN_PT_ERR(d, "PCI Config space initialisation failed.\n");

        xen_host_pci_device_put(&s->real_device);

        return -1;

    }



    /* Bind interrupt */

    rc = xen_host_pci_get_byte(&s->real_device, PCI_INTERRUPT_PIN, &scratch);

    if (rc) {

        XEN_PT_ERR(d, "Failed to read PCI_INTERRUPT_PIN! (rc:%d)\n", rc);

        scratch = 0;

    }

    if (!scratch) {

        XEN_PT_LOG(d, "no pin interrupt\n");

        goto out;

    }



    machine_irq = s->real_device.irq;

    rc = xc_physdev_map_pirq(xen_xc, xen_domid, machine_irq, &pirq);



    if (rc < 0) {

        XEN_PT_ERR(d, "Mapping machine irq %u to pirq %i failed, (err: %d)\n",

                   machine_irq, pirq, errno);



        /* Disable PCI intx assertion (turn on bit10 of devctl) */

        cmd |= PCI_COMMAND_INTX_DISABLE;

        machine_irq = 0;

        s->machine_irq = 0;

    } else {

        machine_irq = pirq;

        s->machine_irq = pirq;

        xen_pt_mapped_machine_irq[machine_irq]++;

    }



    /* bind machine_irq to device */

    if (machine_irq != 0) {

        uint8_t e_intx = xen_pt_pci_intx(s);



        rc = xc_domain_bind_pt_pci_irq(xen_xc, xen_domid, machine_irq,

                                       pci_bus_num(d->bus),

                                       PCI_SLOT(d->devfn),

                                       e_intx);

        if (rc < 0) {

            XEN_PT_ERR(d, "Binding of interrupt %i failed! (err: %d)\n",

                       e_intx, errno);



            /* Disable PCI intx assertion (turn on bit10 of devctl) */

            cmd |= PCI_COMMAND_INTX_DISABLE;

            xen_pt_mapped_machine_irq[machine_irq]--;



            if (xen_pt_mapped_machine_irq[machine_irq] == 0) {

                if (xc_physdev_unmap_pirq(xen_xc, xen_domid, machine_irq)) {

                    XEN_PT_ERR(d, "Unmapping of machine interrupt %i failed!"

                               " (err: %d)\n", machine_irq, errno);

                }

            }

            s->machine_irq = 0;

        }

    }



out:

    if (cmd) {

        uint16_t val;



        rc = xen_host_pci_get_word(&s->real_device, PCI_COMMAND, &val);

        if (rc) {

            XEN_PT_ERR(d, "Failed to read PCI_COMMAND! (rc: %d)\n", rc);

        } else {

            val |= cmd;

            rc = xen_host_pci_set_word(&s->real_device, PCI_COMMAND, val);

            if (rc) {

                XEN_PT_ERR(d, "Failed to write PCI_COMMAND val=0x%x!(rc: %d)\n",

                           val, rc);

            }

        }

    }



    memory_listener_register(&s->memory_listener, &s->dev.bus_master_as);

    memory_listener_register(&s->io_listener, &address_space_io);

    s->listener_set = true;

    XEN_PT_LOG(d,

               "Real physical device %02x:%02x.%d registered successfully!\n",

               s->hostaddr.bus, s->hostaddr.slot, s->hostaddr.function);



    return 0;

}
