static int decode_frame_adu(AVCodecContext *avctx, void *data,

                            int *got_frame_ptr, AVPacket *avpkt)

{

    const uint8_t *buf  = avpkt->data;

    int buf_size        = avpkt->size;

    MPADecodeContext *s = avctx->priv_data;

    uint32_t header;

    int len, ret;



    len = buf_size;



    // Discard too short frames

    if (buf_size < HEADER_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "Packet is too small\n");

        return AVERROR_INVALIDDATA;

    }





    if (len > MPA_MAX_CODED_FRAME_SIZE)

        len = MPA_MAX_CODED_FRAME_SIZE;



    // Get header and restore sync word

    header = AV_RB32(buf) | 0xffe00000;



    if (ff_mpa_check_header(header) < 0) { // Bad header, discard frame

        av_log(avctx, AV_LOG_ERROR, "Invalid frame header\n");

        return AVERROR_INVALIDDATA;

    }



    avpriv_mpegaudio_decode_header((MPADecodeHeader *)s, header);

    /* update codec info */

    avctx->sample_rate = s->sample_rate;

    avctx->channels    = s->nb_channels;

    avctx->channel_layout = s->nb_channels == 1 ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;

    if (!avctx->bit_rate)

        avctx->bit_rate = s->bit_rate;



    s->frame_size = len;



    s->frame = data;



    ret = mp_decode_frame(s, NULL, buf, buf_size);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error while decoding MPEG audio frame.\n");

        return ret;

    }



    *got_frame_ptr = 1;



    return buf_size;

}
