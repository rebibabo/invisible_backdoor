static void stellaris_enet_unrealize(DeviceState *dev, Error **errp)

{

    stellaris_enet_state *s = STELLARIS_ENET(dev);



    unregister_savevm(DEVICE(s), "stellaris_enet", s);



    memory_region_destroy(&s->mmio);

}
