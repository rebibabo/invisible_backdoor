static int h264_mp4toannexb_filter(AVBitStreamFilterContext *bsfc,

                                   AVCodecContext *avctx, const char *args,

                                   uint8_t  **poutbuf, int *poutbuf_size,

                                   const uint8_t *buf, int      buf_size,

                                   int keyframe) {

    H264BSFContext *ctx = bsfc->priv_data;

    uint8_t unit_type;

    int32_t nal_size;

    uint32_t cumul_size = 0;

    const uint8_t *buf_end = buf + buf_size;



    /* nothing to filter */

    if (!avctx->extradata || avctx->extradata_size < 6) {

        *poutbuf = (uint8_t*) buf;

        *poutbuf_size = buf_size;

        return 0;

    }



    /* retrieve sps and pps NAL units from extradata */

    if (!ctx->extradata_parsed) {

        uint16_t unit_size;

        uint64_t total_size = 0;

        uint8_t *out = NULL, unit_nb, sps_done = 0;

        const uint8_t *extradata = avctx->extradata+4;

        static const uint8_t nalu_header[4] = {0, 0, 0, 1};



        /* retrieve length coded size */

        ctx->length_size = (*extradata++ & 0x3) + 1;

        if (ctx->length_size == 3)

            return AVERROR(EINVAL);



        /* retrieve sps and pps unit(s) */

        unit_nb = *extradata++ & 0x1f; /* number of sps unit(s) */

        if (!unit_nb) {

            unit_nb = *extradata++; /* number of pps unit(s) */

            sps_done++;

        }

        while (unit_nb--) {

            void *tmp;



            unit_size = AV_RB16(extradata);

            total_size += unit_size+4;

            if (total_size > INT_MAX - FF_INPUT_BUFFER_PADDING_SIZE ||

                extradata+2+unit_size > avctx->extradata+avctx->extradata_size) {

                av_free(out);

                return AVERROR(EINVAL);

            }

            tmp = av_realloc(out, total_size + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!tmp) {

                av_free(out);

                return AVERROR(ENOMEM);

            }

            out = tmp;

            memcpy(out+total_size-unit_size-4, nalu_header, 4);

            memcpy(out+total_size-unit_size,   extradata+2, unit_size);

            extradata += 2+unit_size;



            if (!unit_nb && !sps_done++)

                unit_nb = *extradata++; /* number of pps unit(s) */

        }



        memset(out + total_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);

        av_free(avctx->extradata);

        avctx->extradata      = out;

        avctx->extradata_size = total_size;

        ctx->first_idr        = 1;

        ctx->extradata_parsed = 1;

    }



    *poutbuf_size = 0;

    *poutbuf = NULL;

    do {

        if (buf + ctx->length_size > buf_end)

            goto fail;



        if (ctx->length_size == 1) {

            nal_size = buf[0];

        } else if (ctx->length_size == 2) {

            nal_size = AV_RB16(buf);

        } else

            nal_size = AV_RB32(buf);



        buf += ctx->length_size;

        unit_type = *buf & 0x1f;



        if (buf + nal_size > buf_end || nal_size < 0)

            goto fail;



        /* prepend only to the first type 5 NAL unit of an IDR picture */

        if (ctx->first_idr && unit_type == 5) {

            if (alloc_and_copy(poutbuf, poutbuf_size,

                               avctx->extradata, avctx->extradata_size,

                               buf, nal_size) < 0)

                goto fail;

            ctx->first_idr = 0;

        } else {

            if (alloc_and_copy(poutbuf, poutbuf_size,

                               NULL, 0,

                               buf, nal_size) < 0)

                goto fail;

            if (!ctx->first_idr && unit_type == 1)

                ctx->first_idr = 1;

        }



        buf += nal_size;

        cumul_size += nal_size + ctx->length_size;

    } while (cumul_size < buf_size);



    return 1;



fail:

    av_freep(poutbuf);

    *poutbuf_size = 0;

    return AVERROR(EINVAL);

}
