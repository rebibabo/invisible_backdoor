static void vmd_decode(VmdVideoContext *s)

{

    int i;

    unsigned int *palette32;

    unsigned char r, g, b;



    /* point to the start of the encoded data */

    unsigned char *p = s->buf + 16;



    unsigned char *pb;

    unsigned char meth;

    unsigned char *dp;   /* pointer to current frame */

    unsigned char *pp;   /* pointer to previous frame */

    unsigned char len;

    int ofs;



    int frame_x, frame_y;

    int frame_width, frame_height;



    frame_x = LE_16(&s->buf[6]);

    frame_y = LE_16(&s->buf[8]);

    frame_width = LE_16(&s->buf[10]) - frame_x + 1;

    frame_height = LE_16(&s->buf[12]) - frame_y + 1;



    /* if only a certain region will be updated, copy the entire previous

     * frame before the decode */

    if (frame_x || frame_y || (frame_width != s->avctx->width) ||

        (frame_height != s->avctx->height)) {



        memcpy(s->frame.data[0], s->prev_frame.data[0], 

            s->avctx->height * s->frame.linesize[0]);

    }



    /* check if there is a new palette */

    if (s->buf[15] & 0x02) {

        p += 2;

        palette32 = (unsigned int *)s->palette;

        for (i = 0; i < PALETTE_COUNT; i++) {

            r = *p++ * 4;

            g = *p++ * 4;

            b = *p++ * 4;

            palette32[i] = (r << 16) | (g << 8) | (b);

        }

        s->size -= (256 * 3 + 2);

    }

    if (s->size >= 0) {

        /* originally UnpackFrame in VAG's code */

        pb = p;

        meth = *pb++;

        if (meth & 0x80) {

            lz_unpack(pb, s->unpack_buffer);

            meth &= 0x7F;

            pb = s->unpack_buffer;

        }



        dp = &s->frame.data[0][frame_y * s->frame.linesize[0] + frame_x];

        pp = &s->prev_frame.data[0][frame_y * s->prev_frame.linesize[0] + frame_x];

        switch (meth) {

        case 1:

            for (i = 0; i < frame_height; i++) {

                ofs = 0;

                do {

                    len = *pb++;

                    if (len & 0x80) {

                        len = (len & 0x7F) + 1;

                        memcpy(&dp[ofs], pb, len);

                        pb += len;

                        ofs += len;

                    } else {

                        /* interframe pixel copy */

                        memcpy(&dp[ofs], &pp[ofs], len + 1);

                        ofs += len + 1;

                    }

                } while (ofs < frame_width);

                if (ofs > frame_width) {

                    av_log(s->avctx, AV_LOG_ERROR, "VMD video: offset > width (%d > %d)\n",

                        ofs, frame_width);

                    break;

                }

                dp += s->frame.linesize[0];

                pp += s->prev_frame.linesize[0];

            }

            break;



        case 2:

            for (i = 0; i < frame_height; i++) {

                memcpy(dp, pb, frame_width);

                pb += frame_width;

                dp += s->frame.linesize[0];

                pp += s->prev_frame.linesize[0];

            }

            break;



        case 3:

            for (i = 0; i < frame_height; i++) {

                ofs = 0;

                do {

                    len = *pb++;

                    if (len & 0x80) {

                        len = (len & 0x7F) + 1;

                        if (*pb++ == 0xFF)

                            len = rle_unpack(pb, &dp[ofs], len);

                        else

                            memcpy(&dp[ofs], pb, len);

                        pb += len;

                        ofs += len;

                    } else {

                        /* interframe pixel copy */

                        memcpy(&dp[ofs], &pp[ofs], len + 1);

                        ofs += len + 1;

                    }

                } while (ofs < frame_width);

                if (ofs > frame_width) {

                    av_log(s->avctx, AV_LOG_ERROR, "VMD video: offset > width (%d > %d)\n",

                        ofs, frame_width);

                }

                dp += s->frame.linesize[0];

                pp += s->prev_frame.linesize[0];

            }

            break;

        }

    }

}
