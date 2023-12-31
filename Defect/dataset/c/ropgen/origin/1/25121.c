static void i6300esb_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->config_read = i6300esb_config_read;

    k->config_write = i6300esb_config_write;

    k->realize = i6300esb_realize;


    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_ESB_9;

    k->class_id = PCI_CLASS_SYSTEM_OTHER;

    dc->reset = i6300esb_reset;

    dc->vmsd = &vmstate_i6300esb;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

}