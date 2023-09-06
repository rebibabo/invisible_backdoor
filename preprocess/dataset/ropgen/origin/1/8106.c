static int stellaris_enet_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    stellaris_enet_state *s = STELLARIS_ENET(dev);



    memory_region_init_io(&s->mmio, OBJECT(s), &stellaris_enet_ops, s,

                          "stellaris_enet", 0x1000);

    sysbus_init_mmio(sbd, &s->mmio);

    sysbus_init_irq(sbd, &s->irq);

    qemu_macaddr_default_if_unset(&s->conf.macaddr);



    s->nic = qemu_new_nic(&net_stellaris_enet_info, &s->conf,

                          object_get_typename(OBJECT(dev)), dev->id, s);

    qemu_format_nic_info_str(qemu_get_queue(s->nic), s->conf.macaddr.a);



    stellaris_enet_reset(s);

    register_savevm(dev, "stellaris_enet", -1, 1,

                    stellaris_enet_save, stellaris_enet_load, s);

    return 0;

}
