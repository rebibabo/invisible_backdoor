static int sunrast_decode_frame(AVCodecContext *avctx, void *data,

                                int *data_size, AVPacket *avpkt) {

    const uint8_t *buf = avpkt->data;

    const uint8_t *buf_end = avpkt->data + avpkt->size;

    SUNRASTContext * const s = avctx->priv_data;

    AVFrame *picture = data;

    AVFrame * const p = &s->picture;

    unsigned int w, h, depth, type, maptype, maplength, stride, x, y, len, alen;

    uint8_t *ptr;

    const uint8_t *bufstart = buf;



    if (avpkt->size < 32)

        return AVERROR_INVALIDDATA;



    if (AV_RB32(buf) != 0x59a66a95) {

        av_log(avctx, AV_LOG_ERROR, "this is not sunras encoded data\n");

        return -1;

    }



    w         = AV_RB32(buf+4);

    h         = AV_RB32(buf+8);

    depth     = AV_RB32(buf+12);

    type      = AV_RB32(buf+20);

    maptype   = AV_RB32(buf+24);

    maplength = AV_RB32(buf+28);

    buf      += 32;



    if (type == RT_FORMAT_TIFF || type == RT_FORMAT_IFF) {

        av_log(avctx, AV_LOG_ERROR, "unsupported (compression) type\n");

        return -1;

    }

    if (type < RT_OLD || type > RT_FORMAT_IFF) {

        av_log(avctx, AV_LOG_ERROR, "invalid (compression) type\n");

        return -1;

    }

    if (av_image_check_size(w, h, 0, avctx)) {

        av_log(avctx, AV_LOG_ERROR, "invalid image size\n");

        return -1;

    }

    if (maptype & ~1) {

        av_log(avctx, AV_LOG_ERROR, "invalid colormap type\n");

        return -1;

    }





    switch (depth) {

        case 1:

            avctx->pix_fmt = PIX_FMT_MONOWHITE;

            break;

        case 8:

            avctx->pix_fmt = PIX_FMT_PAL8;

            break;

        case 24:

            avctx->pix_fmt = (type == RT_FORMAT_RGB) ? PIX_FMT_RGB24 : PIX_FMT_BGR24;

            break;

        default:

            av_log(avctx, AV_LOG_ERROR, "invalid depth\n");

            return -1;

    }



    if (p->data[0])

        avctx->release_buffer(avctx, p);



    if (w != avctx->width || h != avctx->height)

        avcodec_set_dimensions(avctx, w, h);

    if (avctx->get_buffer(avctx, p) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    p->pict_type = AV_PICTURE_TYPE_I;



    if (buf_end - buf < maplength)

        return AVERROR_INVALIDDATA;



    if (depth != 8 && maplength) {

        av_log(avctx, AV_LOG_WARNING, "useless colormap found or file is corrupted, trying to recover\n");



    } else if (depth == 8) {

        unsigned int len = maplength / 3;



        if (!maplength) {

            av_log(avctx, AV_LOG_ERROR, "colormap expected\n");

            return -1;

        }

        if (maplength % 3 || maplength > 768) {

            av_log(avctx, AV_LOG_WARNING, "invalid colormap length\n");

            return -1;

        }



        ptr = p->data[1];

        for (x=0; x<len; x++, ptr+=4)

            *(uint32_t *)ptr = (buf[x]<<16) + (buf[len+x]<<8) + buf[len+len+x];

    }



    buf += maplength;



    ptr    = p->data[0];

    stride = p->linesize[0];



    /* scanlines are aligned on 16 bit boundaries */

    len  = (depth * w + 7) >> 3;

    alen = len + (len&1);



    if (type == RT_BYTE_ENCODED) {

        int value, run;

        uint8_t *end = ptr + h*stride;



        x = 0;

        while (ptr != end && buf < buf_end) {

            run = 1;

            if (buf_end - buf < 1)

                return AVERROR_INVALIDDATA;



            if ((value = *buf++) == 0x80) {

                run = *buf++ + 1;

                if (run != 1)

                    value = *buf++;

            }

            while (run--) {

                if (x < len)

                    ptr[x] = value;

                if (++x >= alen) {

                    x = 0;

                    ptr += stride;

                    if (ptr == end)

                        break;

                }

            }

        }

    } else {

        for (y=0; y<h; y++) {

            if (buf_end - buf < len)

                break;

            memcpy(ptr, buf, len);

            ptr += stride;

            buf += alen;

        }

    }



    *picture = s->picture;

    *data_size = sizeof(AVFrame);



    return buf - bufstart;

}
