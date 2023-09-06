static void vnc_update(VncState *vs, int x, int y, int w, int h)

{

    int i;



    h += y;



    /* round x down to ensure the loop only spans one 16-pixel block per,

       iteration.  otherwise, if (x % 16) != 0, the last iteration may span

       two 16-pixel blocks but we only mark the first as dirty

    */

    w += (x % 16);

    x -= (x % 16);



    x = MIN(x, vs->serverds.width);

    y = MIN(y, vs->serverds.height);

    w = MIN(x + w, vs->serverds.width) - x;

    h = MIN(h, vs->serverds.height);



    for (; y < h; y++)

        for (i = 0; i < w; i += 16)

            vnc_set_bit(vs->dirty_row[y], (x + i) / 16);

}
