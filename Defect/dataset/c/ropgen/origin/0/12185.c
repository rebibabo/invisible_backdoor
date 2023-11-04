static int lag_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    LagarithContext *l = avctx->priv_data;

    AVFrame *const p = &l->picture;

    uint8_t frametype = 0;

    uint32_t offset_gu = 0, offset_bv = 0, offset_ry = 9;

    int offs[4];

    uint8_t *srcs[4], *dst;

    int i, j, planes = 3;



    AVFrame *picture = data;



    if (p->data[0])

        avctx->release_buffer(avctx, p);



    p->reference = 0;

    p->key_frame = 1;



    frametype = buf[0];



    offset_gu = AV_RL32(buf + 1);

    offset_bv = AV_RL32(buf + 5);



    switch (frametype) {

    case FRAME_SOLID_RGBA:

        avctx->pix_fmt = PIX_FMT_RGB32;



        if (avctx->get_buffer(avctx, p) < 0) {

            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            return -1;

        }



        dst = p->data[0];

        for (j = 0; j < avctx->height; j++) {

            for (i = 0; i < avctx->width; i++)

                AV_WN32(dst + i * 4, offset_gu);

            dst += p->linesize[0];

        }

        break;

    case FRAME_ARITH_RGBA:

        avctx->pix_fmt = PIX_FMT_RGB32;

        planes = 4;

        offset_ry += 4;

        offs[3] = AV_RL32(buf + 9);

    case FRAME_ARITH_RGB24:

        if (frametype == FRAME_ARITH_RGB24)

            avctx->pix_fmt = PIX_FMT_RGB24;



        if (avctx->get_buffer(avctx, p) < 0) {

            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            return -1;

        }



        offs[0] = offset_bv;

        offs[1] = offset_gu;

        offs[2] = offset_ry;



        if (!l->rgb_planes) {

            l->rgb_stride = FFALIGN(avctx->width, 16);

            l->rgb_planes = av_malloc(l->rgb_stride * avctx->height * planes + 16);

            if (!l->rgb_planes) {

                av_log(avctx, AV_LOG_ERROR, "cannot allocate temporary buffer\n");

                return AVERROR(ENOMEM);

            }

        }

        for (i = 0; i < planes; i++)

            srcs[i] = l->rgb_planes + (i + 1) * l->rgb_stride * avctx->height - l->rgb_stride;

        for (i = 0; i < planes; i++)

            lag_decode_arith_plane(l, srcs[i],

                                   avctx->width, avctx->height,

                                   -l->rgb_stride, buf + offs[i],

                                   buf_size);

        dst = p->data[0];

        for (i = 0; i < planes; i++)

            srcs[i] = l->rgb_planes + i * l->rgb_stride * avctx->height;

        for (j = 0; j < avctx->height; j++) {

            for (i = 0; i < avctx->width; i++) {

                uint8_t r, g, b, a;

                r = srcs[0][i];

                g = srcs[1][i];

                b = srcs[2][i];

                r += g;

                b += g;

                if (frametype == FRAME_ARITH_RGBA) {

                    a = srcs[3][i];

                    AV_WN32(dst + i * 4, MKBETAG(a, r, g, b));

                } else {

                    dst[i * 3 + 0] = r;

                    dst[i * 3 + 1] = g;

                    dst[i * 3 + 2] = b;

                }

            }

            dst += p->linesize[0];

            for (i = 0; i < planes; i++)

                srcs[i] += l->rgb_stride;

        }

        break;

    case FRAME_ARITH_YV12:

        avctx->pix_fmt = PIX_FMT_YUV420P;



        if (avctx->get_buffer(avctx, p) < 0) {

            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            return -1;

        }



        lag_decode_arith_plane(l, p->data[0], avctx->width, avctx->height,

                               p->linesize[0], buf + offset_ry,

                               buf_size);

        lag_decode_arith_plane(l, p->data[2], avctx->width / 2,

                               avctx->height / 2, p->linesize[2],

                               buf + offset_gu, buf_size);

        lag_decode_arith_plane(l, p->data[1], avctx->width / 2,

                               avctx->height / 2, p->linesize[1],

                               buf + offset_bv, buf_size);

        break;

    default:

        av_log(avctx, AV_LOG_ERROR,

               "Unsupported Lagarith frame type: %#x\n", frametype);

        return -1;

    }



    *picture = *p;

    *data_size = sizeof(AVFrame);



    return buf_size;

}