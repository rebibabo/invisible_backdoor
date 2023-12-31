static int xen_pt_register_regions(XenPCIPassthroughState *s)

{

    int i = 0;

    XenHostPCIDevice *d = &s->real_device;



    /* Register PIO/MMIO BARs */

    for (i = 0; i < PCI_ROM_SLOT; i++) {

        XenHostPCIIORegion *r = &d->io_regions[i];

        uint8_t type;



        if (r->base_addr == 0 || r->size == 0) {

            continue;

        }



        s->bases[i].access.u = r->base_addr;



        if (r->type & XEN_HOST_PCI_REGION_TYPE_IO) {

            type = PCI_BASE_ADDRESS_SPACE_IO;

        } else {

            type = PCI_BASE_ADDRESS_SPACE_MEMORY;

            if (r->type & XEN_HOST_PCI_REGION_TYPE_PREFETCH) {

                type |= PCI_BASE_ADDRESS_MEM_PREFETCH;

            }

            if (r->type & XEN_HOST_PCI_REGION_TYPE_MEM_64) {

                type |= PCI_BASE_ADDRESS_MEM_TYPE_64;

            }

        }



        memory_region_init_io(&s->bar[i], OBJECT(s), &ops, &s->dev,

                              "xen-pci-pt-bar", r->size);

        pci_register_bar(&s->dev, i, type, &s->bar[i]);



        XEN_PT_LOG(&s->dev, "IO region %i registered (size=0x%08"PRIx64

                   " base_addr=0x%08"PRIx64" type: %#x)\n",

                   i, r->size, r->base_addr, type);

    }



    /* Register expansion ROM address */

    if (d->rom.base_addr && d->rom.size) {

        uint32_t bar_data = 0;



        /* Re-set BAR reported by OS, otherwise ROM can't be read. */

        if (xen_host_pci_get_long(d, PCI_ROM_ADDRESS, &bar_data)) {

            return 0;

        }

        if ((bar_data & PCI_ROM_ADDRESS_MASK) == 0) {

            bar_data |= d->rom.base_addr & PCI_ROM_ADDRESS_MASK;

            xen_host_pci_set_long(d, PCI_ROM_ADDRESS, bar_data);

        }



        s->bases[PCI_ROM_SLOT].access.maddr = d->rom.base_addr;



        memory_region_init_io(&s->rom, OBJECT(s), &ops, &s->dev,

                              "xen-pci-pt-rom", d->rom.size);

        pci_register_bar(&s->dev, PCI_ROM_SLOT, PCI_BASE_ADDRESS_MEM_PREFETCH,

                         &s->rom);



        XEN_PT_LOG(&s->dev, "Expansion ROM registered (size=0x%08"PRIx64

                   " base_addr=0x%08"PRIx64")\n",

                   d->rom.size, d->rom.base_addr);

    }



    return 0;

}
