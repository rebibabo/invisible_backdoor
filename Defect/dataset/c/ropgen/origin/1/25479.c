static void vnc_dpy_update(DisplayChangeListener *dcl,

                           DisplayState *ds,

                           int x, int y, int w, int h)

{

    int i;

    VncDisplay *vd = ds->opaque;

    struct VncSurface *s = &vd->guest;

    int width = ds_get_width(ds);

    int height = ds_get_height(ds);



    h += y;



    /* round x down to ensure the loop only spans one 16-pixel block per,

       iteration.  otherwise, if (x % 16) != 0, the last iteration may span

       two 16-pixel blocks but we only mark the first as dirty

    */

    w += (x % 16);

    x -= (x % 16);



    x = MIN(x, width);

    y = MIN(y, height);

    w = MIN(x + w, width) - x;

    h = MIN(h, height);



    for (; y < h; y++)

        for (i = 0; i < w; i += 16)

            set_bit((x + i) / 16, s->dirty[y]);

}
