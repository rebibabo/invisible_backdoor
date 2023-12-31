static av_cold int gif_encode_init(AVCodecContext *avctx)

{

    GIFContext *s = avctx->priv_data;



    if (avctx->width > 65535 || avctx->height > 65535) {

        av_log(avctx, AV_LOG_ERROR, "GIF does not support resolutions above 65535x65535\n");

        return AVERROR(EINVAL);

    }

#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    s->transparent_index = -1;



    s->lzw = av_mallocz(ff_lzw_encode_state_size);

    s->buf = av_malloc(avctx->width*avctx->height*2);

    s->tmpl = av_malloc(avctx->width);

    if (!s->tmpl || !s->buf || !s->lzw)

        return AVERROR(ENOMEM);



    if (avpriv_set_systematic_pal2(s->palette, avctx->pix_fmt) < 0)

        av_assert0(avctx->pix_fmt == AV_PIX_FMT_PAL8);



    return 0;

}
