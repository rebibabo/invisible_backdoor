static int send_palette_rect(VncState *vs, int w, int h, struct QDict *palette)

{

    int stream = 2;

    int level = tight_conf[vs->tight_compression].idx_zlib_level;

    int colors;

    size_t bytes;



    colors = qdict_size(palette);



    vnc_write_u8(vs, (stream | VNC_TIGHT_EXPLICIT_FILTER) << 4);

    vnc_write_u8(vs, VNC_TIGHT_FILTER_PALETTE);

    vnc_write_u8(vs, colors - 1);



    switch(vs->clientds.pf.bytes_per_pixel) {

    case 4:

    {

        size_t old_offset, offset;

        uint32_t header[qdict_size(palette)];

        struct palette_cb_priv priv = { vs, (uint8_t *)header };



        old_offset = vs->output.offset;

        qdict_iter(palette, write_palette, &priv);

        vnc_write(vs, header, sizeof(header));



        if (vs->tight_pixel24) {

            tight_pack24(vs, vs->output.buffer + old_offset, colors, &offset);

            vs->output.offset = old_offset + offset;

        }



        tight_encode_indexed_rect32(vs->tight.buffer, w * h, palette);

        break;

    }

    case 2:

    {

        uint16_t header[qdict_size(palette)];

        struct palette_cb_priv priv = { vs, (uint8_t *)header };



        qdict_iter(palette, write_palette, &priv);

        vnc_write(vs, header, sizeof(header));

        tight_encode_indexed_rect16(vs->tight.buffer, w * h, palette);

        break;

    }

    default:

        return -1; /* No palette for 8bits colors */

        break;

    }

    bytes = w * h;

    vs->tight.offset = bytes;



    bytes = tight_compress_data(vs, stream, bytes,

                                level, Z_DEFAULT_STRATEGY);

    return (bytes >= 0);

}
