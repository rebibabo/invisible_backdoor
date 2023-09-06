static int gif_decode_frame(AVCodecContext *avctx, void *data, int *got_frame, AVPacket *avpkt)
{
    GifState *s = avctx->priv_data;
    AVFrame *picture = data;
    int ret;
    bytestream2_init(&s->gb, avpkt->data, avpkt->size);
    s->picture.pts          = avpkt->pts;
    s->picture.pkt_pts      = avpkt->pts;
    s->picture.pkt_dts      = avpkt->dts;
    s->picture.pkt_duration = avpkt->duration;
    if (avpkt->size >= 6) {
        s->keyframe = memcmp(avpkt->data, gif87a_sig, 6) == 0 ||
                      memcmp(avpkt->data, gif89a_sig, 6) == 0;
    } else {
        s->keyframe = 0;
    if (s->keyframe) {
        s->keyframe_ok = 0;
        if ((ret = gif_read_header1(s)) < 0)
            return ret;
        if ((ret = av_image_check_size(s->screen_width, s->screen_height, 0, avctx)) < 0)
            return ret;
        avcodec_set_dimensions(avctx, s->screen_width, s->screen_height);
        if (s->picture.data[0])
            avctx->release_buffer(avctx, &s->picture);
        if ((ret = ff_get_buffer(avctx, &s->picture)) < 0) {
            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");
            return ret;
        s->picture.pict_type = AV_PICTURE_TYPE_I;
        s->picture.key_frame = 1;
    } else {
        if ((ret = avctx->reget_buffer(avctx, &s->picture)) < 0) {
            av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");
            return ret;
        s->picture.pict_type = AV_PICTURE_TYPE_P;
        s->picture.key_frame = 0;
    ret = gif_parse_next_image(s, got_frame);
    if (ret < 0)
        return ret;
    else if (*got_frame)
        *picture = s->picture;
    return avpkt->size;