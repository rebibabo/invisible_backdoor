static void es1370_class_init (ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS (klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS (klass);



    k->realize = es1370_realize;


    k->vendor_id = PCI_VENDOR_ID_ENSONIQ;

    k->device_id = PCI_DEVICE_ID_ENSONIQ_ES1370;

    k->class_id = PCI_CLASS_MULTIMEDIA_AUDIO;

    k->subsystem_vendor_id = 0x4942;

    k->subsystem_id = 0x4c4c;

    set_bit(DEVICE_CATEGORY_SOUND, dc->categories);

    dc->desc = "ENSONIQ AudioPCI ES1370";

    dc->vmsd = &vmstate_es1370;

}