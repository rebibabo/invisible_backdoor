static void pciej_write(void *opaque, uint32_t addr, uint32_t val)

{

    BusState *bus = opaque;

    DeviceState *qdev, *next;

    PCIDevice *dev;

    int slot = ffs(val) - 1;



    QLIST_FOREACH_SAFE(qdev, &bus->children, sibling, next) {

        dev = DO_UPCAST(PCIDevice, qdev, qdev);

        if (PCI_SLOT(dev->devfn) == slot) {

            qdev_free(qdev);

        }

    }





    PIIX4_DPRINTF("pciej write %x <== %d\n", addr, val);

}
