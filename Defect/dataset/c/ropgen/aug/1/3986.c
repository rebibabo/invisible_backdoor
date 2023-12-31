static MTPData *usb_mtp_get_object(MTPState *s, MTPControl *c,

                                   MTPObject *o)

{

    MTPData *d = usb_mtp_data_alloc(c);



    trace_usb_mtp_op_get_object(s->dev.addr, o->handle, o->path);



    d->fd = open(o->path, O_RDONLY);

    if (d->fd == -1) {


        return NULL;

    }

    d->length = o->stat.st_size;

    d->alloc  = 512;

    d->data   = g_malloc(d->alloc);

    return d;

}