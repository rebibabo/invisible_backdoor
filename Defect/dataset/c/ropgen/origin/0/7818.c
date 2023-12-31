static int vnc_refresh_server_surface(VncDisplay *vd)

{

    int width = pixman_image_get_width(vd->guest.fb);

    int height = pixman_image_get_height(vd->guest.fb);

    int y;

    uint8_t *guest_row;

    uint8_t *server_row;

    int cmp_bytes;

    VncState *vs;

    int has_dirty = 0;

    pixman_image_t *tmpbuf = NULL;



    struct timeval tv = { 0, 0 };



    if (!vd->non_adaptive) {

        gettimeofday(&tv, NULL);

        has_dirty = vnc_update_stats(vd, &tv);

    }



    /*

     * Walk through the guest dirty map.

     * Check and copy modified bits from guest to server surface.

     * Update server dirty map.

     */

    cmp_bytes = 64;

    if (cmp_bytes > vnc_server_fb_stride(vd)) {

        cmp_bytes = vnc_server_fb_stride(vd);

    }

    if (vd->guest.format != VNC_SERVER_FB_FORMAT) {

        int width = pixman_image_get_width(vd->server);

        tmpbuf = qemu_pixman_linebuf_create(VNC_SERVER_FB_FORMAT, width);

    }

    guest_row = (uint8_t *)pixman_image_get_data(vd->guest.fb);

    server_row = (uint8_t *)pixman_image_get_data(vd->server);

    for (y = 0; y < height; y++) {

        if (!bitmap_empty(vd->guest.dirty[y], VNC_DIRTY_BITS)) {

            int x;

            uint8_t *guest_ptr;

            uint8_t *server_ptr;



            if (vd->guest.format != VNC_SERVER_FB_FORMAT) {

                qemu_pixman_linebuf_fill(tmpbuf, vd->guest.fb, width, y);

                guest_ptr = (uint8_t *)pixman_image_get_data(tmpbuf);

            } else {

                guest_ptr = guest_row;

            }

            server_ptr = server_row;



            for (x = 0; x + 15 < width;

                    x += 16, guest_ptr += cmp_bytes, server_ptr += cmp_bytes) {

                if (!test_and_clear_bit((x / 16), vd->guest.dirty[y]))

                    continue;

                if (memcmp(server_ptr, guest_ptr, cmp_bytes) == 0)

                    continue;

                memcpy(server_ptr, guest_ptr, cmp_bytes);

                if (!vd->non_adaptive)

                    vnc_rect_updated(vd, x, y, &tv);

                QTAILQ_FOREACH(vs, &vd->clients, next) {

                    set_bit((x / 16), vs->dirty[y]);

                }

                has_dirty++;

            }

        }

        guest_row  += pixman_image_get_stride(vd->guest.fb);

        server_row += pixman_image_get_stride(vd->server);

    }

    qemu_pixman_image_unref(tmpbuf);

    return has_dirty;

}
