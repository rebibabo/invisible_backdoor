static int decode_frame(AVCodecContext * avctx, void *data, int *data_size,

                        AVPacket *avpkt)

{

    const uint8_t *buf  = avpkt->data;

    int buf_size        = avpkt->size;

    MPADecodeContext *s = avctx->priv_data;

    uint32_t header;

    int out_size;

    OUT_INT *out_samples = data;



    if (buf_size < HEADER_SIZE)

        return AVERROR_INVALIDDATA;



    header = AV_RB32(buf);

    if (ff_mpa_check_header(header) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Header missing\n");

        return AVERROR_INVALIDDATA;

    }



    if (avpriv_mpegaudio_decode_header((MPADecodeHeader *)s, header) == 1) {

        /* free format: prepare to compute frame size */

        s->frame_size = -1;

        return AVERROR_INVALIDDATA;

    }

    /* update codec info */

    avctx->channels       = s->nb_channels;

    avctx->channel_layout = s->nb_channels == 1 ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;

    if (!avctx->bit_rate)

        avctx->bit_rate = s->bit_rate;

    avctx->sub_id = s->layer;



    if (*data_size < 1152 * avctx->channels * sizeof(OUT_INT))

        return AVERROR(EINVAL);

    *data_size = 0;



    if (s->frame_size <= 0 || s->frame_size > buf_size) {

        av_log(avctx, AV_LOG_ERROR, "incomplete frame\n");

        return AVERROR_INVALIDDATA;

    } else if (s->frame_size < buf_size) {

        av_log(avctx, AV_LOG_ERROR, "incorrect frame size\n");

        buf_size= s->frame_size;

    }



    out_size = mp_decode_frame(s, out_samples, buf, buf_size);

    if (out_size >= 0) {

        *data_size         = out_size;

        avctx->sample_rate = s->sample_rate;

        //FIXME maybe move the other codec info stuff from above here too

    } else {

        av_log(avctx, AV_LOG_ERROR, "Error while decoding MPEG audio frame.\n");

        /* Only return an error if the bad frame makes up the whole packet.

           If there is more data in the packet, just consume the bad frame

           instead of returning an error, which would discard the whole

           packet. */

        if (buf_size == avpkt->size)

            return out_size;

    }

    s->frame_size = 0;

    return buf_size;

}
