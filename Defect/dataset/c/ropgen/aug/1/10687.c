static void vnc_dpy_resize(DisplayChangeListener *dcl,

                           DisplayState *ds)

{

    VncDisplay *vd = ds->opaque;

    VncState *vs;



    vnc_abort_display_jobs(vd);



    /* server surface */

    qemu_pixman_image_unref(vd->server);

    vd->server = pixman_image_create_bits(VNC_SERVER_FB_FORMAT,

                                          ds_get_width(ds),

                                          ds_get_height(ds),

                                          NULL, 0);



    /* guest surface */

#if 0 /* FIXME */

    if (ds_get_bytes_per_pixel(ds) != vd->guest.ds->pf.bytes_per_pixel)

        console_color_init(ds);

#endif

    qemu_pixman_image_unref(vd->guest.fb);

    vd->guest.fb = pixman_image_ref(ds->surface->image);

    vd->guest.format = ds->surface->format;

    memset(vd->guest.dirty, 0xFF, sizeof(vd->guest.dirty));



    QTAILQ_FOREACH(vs, &vd->clients, next) {

        vnc_colordepth(vs);

        vnc_desktop_resize(vs);

        if (vs->vd->cursor) {

            vnc_cursor_define(vs);

        }

        memset(vs->dirty, 0xFF, sizeof(vs->dirty));

    }

}
