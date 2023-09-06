static int process_frame_obj(SANMVideoContext *ctx)
{
    uint16_t codec, top, left, w, h;
    codec = bytestream2_get_le16u(&ctx->gb);
    left  = bytestream2_get_le16u(&ctx->gb);
    top   = bytestream2_get_le16u(&ctx->gb);
    w     = bytestream2_get_le16u(&ctx->gb);
    h     = bytestream2_get_le16u(&ctx->gb);
    if (ctx->width < left + w || ctx->height < top + h) {
        if (av_image_check_size(FFMAX(left + w, ctx->width),
                                FFMAX(top  + h, ctx->height), 0, ctx->avctx) < 0)
        avcodec_set_dimensions(ctx->avctx, FFMAX(left + w, ctx->width),
                                           FFMAX(top  + h, ctx->height));
        init_sizes(ctx, FFMAX(left + w, ctx->width),
                        FFMAX(top  + h, ctx->height));
        if (init_buffers(ctx)) {
            av_log(ctx->avctx, AV_LOG_ERROR, "error resizing buffers\n");
            return AVERROR(ENOMEM);
    bytestream2_skip(&ctx->gb, 4);
    av_dlog(ctx->avctx, "subcodec %d\n", codec);
    switch (codec) {
    case 1:
    case 3:
        return old_codec1(ctx, top, left, w, h);
        break;
    case 37:
        return old_codec37(ctx, top, left, w, h);
        break;
    case 47:
        return old_codec47(ctx, top, left, w, h);
        break;
    default:
        avpriv_request_sample(ctx->avctx, "unknown subcodec %d", codec);
        return AVERROR_PATCHWELCOME;