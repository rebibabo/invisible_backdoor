USBDevice *usb_msd_init(const char *filename)

{

    MSDState *s;

    BlockDriverState *bdrv;

    BlockDriver *drv = NULL;

    const char *p1;

    char fmt[32];



    p1 = strchr(filename, ':');

    if (p1++) {

        const char *p2;



        if (strstart(filename, "format=", &p2)) {

            int len = MIN(p1 - p2, sizeof(fmt));

            pstrcpy(fmt, len, p2);



            drv = bdrv_find_format(fmt);

            if (!drv) {

                printf("invalid format %s\n", fmt);

                return NULL;

            }

        } else if (*filename != ':') {

            printf("unrecognized USB mass-storage option %s\n", filename);

            return NULL;

        }



        filename = p1;

    }



    if (!*filename) {

        printf("block device specification needed\n");

        return NULL;

    }



    s = qemu_mallocz(sizeof(MSDState));



    bdrv = bdrv_new("usb");

    if (bdrv_open2(bdrv, filename, 0, drv) < 0)

        goto fail;

    if (qemu_key_check(bdrv, filename))

        goto fail;

    s->bs = bdrv;



    s->dev.speed = USB_SPEED_FULL;

    s->dev.handle_packet = usb_generic_handle_packet;



    s->dev.handle_reset = usb_msd_handle_reset;

    s->dev.handle_control = usb_msd_handle_control;

    s->dev.handle_data = usb_msd_handle_data;

    s->dev.handle_destroy = usb_msd_handle_destroy;



    snprintf(s->dev.devname, sizeof(s->dev.devname), "QEMU USB MSD(%.16s)",

             filename);



    s->scsi_dev = scsi_disk_init(bdrv, 0, usb_msd_command_complete, s);

    usb_msd_handle_reset((USBDevice *)s);

    return (USBDevice *)s;

 fail:

    qemu_free(s);

    return NULL;

}