int ff_load_image(uint8_t *data[4], int linesize[4],

                  int *w, int *h, enum AVPixelFormat *pix_fmt,

                  const char *filename, void *log_ctx)

{

    AVInputFormat *iformat = NULL;

    AVFormatContext *format_ctx = NULL;

    AVCodec *codec;

    AVCodecContext *codec_ctx;

    AVFrame *frame;

    int frame_decoded, ret = 0;

    AVPacket pkt;



    av_register_all();



    iformat = av_find_input_format("image2");

    if ((ret = avformat_open_input(&format_ctx, filename, iformat, NULL)) < 0) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Failed to open input file '%s'\n", filename);

        return ret;

    }



    codec_ctx = format_ctx->streams[0]->codec;

    codec = avcodec_find_decoder(codec_ctx->codec_id);

    if (!codec) {

        av_log(log_ctx, AV_LOG_ERROR, "Failed to find codec\n");

        ret = AVERROR(EINVAL);

        goto end;

    }



    if ((ret = avcodec_open2(codec_ctx, codec, NULL)) < 0) {

        av_log(log_ctx, AV_LOG_ERROR, "Failed to open codec\n");

        goto end;

    }



    if (!(frame = avcodec_alloc_frame()) ) {

        av_log(log_ctx, AV_LOG_ERROR, "Failed to alloc frame\n");

        ret = AVERROR(ENOMEM);

        goto end;

    }



    ret = av_read_frame(format_ctx, &pkt);

    if (ret < 0) {

        av_log(log_ctx, AV_LOG_ERROR, "Failed to read frame from file\n");

        goto end;

    }



    ret = avcodec_decode_video2(codec_ctx, frame, &frame_decoded, &pkt);

    if (ret < 0 || !frame_decoded) {

        av_log(log_ctx, AV_LOG_ERROR, "Failed to decode image from file\n");

        goto end;

    }

    ret = 0;



    *w       = frame->width;

    *h       = frame->height;

    *pix_fmt = frame->format;



    if ((ret = av_image_alloc(data, linesize, *w, *h, *pix_fmt, 16)) < 0)

        goto end;

    ret = 0;



    av_image_copy(data, linesize, (const uint8_t **)frame->data, frame->linesize, *pix_fmt, *w, *h);



end:

    if (codec_ctx)

        avcodec_close(codec_ctx);

    if (format_ctx)

        avformat_close_input(&format_ctx);

    av_freep(&frame);



    if (ret < 0)

        av_log(log_ctx, AV_LOG_ERROR, "Error loading image file '%s'\n", filename);

    return ret;

}
