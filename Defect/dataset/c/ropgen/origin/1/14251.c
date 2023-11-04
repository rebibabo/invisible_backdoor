void pcie_cap_slot_hot_unplug_cb(HotplugHandler *hotplug_dev, DeviceState *dev,

                                 Error **errp)

{

    uint8_t *exp_cap;



    pcie_cap_slot_hotplug_common(PCI_DEVICE(hotplug_dev), dev, &exp_cap, errp);



    object_unparent(OBJECT(dev));

    pci_word_test_and_clear_mask(exp_cap + PCI_EXP_SLTSTA,

                                 PCI_EXP_SLTSTA_PDS);

    pcie_cap_slot_event(PCI_DEVICE(hotplug_dev), PCI_EXP_HP_EV_PDC);

}