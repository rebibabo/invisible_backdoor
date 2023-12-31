static int decode_frame_headers(Indeo3DecodeContext *ctx, AVCodecContext *avctx,

                                const uint8_t *buf, int buf_size)

{

    const uint8_t   *buf_ptr = buf, *bs_hdr;

    uint32_t        frame_num, word2, check_sum, data_size;

    uint32_t        y_offset, u_offset, v_offset, starts[3], ends[3];

    uint16_t        height, width;

    int             i, j;



    /* parse and check the OS header */

    frame_num = bytestream_get_le32(&buf_ptr);

    word2     = bytestream_get_le32(&buf_ptr);

    check_sum = bytestream_get_le32(&buf_ptr);

    data_size = bytestream_get_le32(&buf_ptr);



    if ((frame_num ^ word2 ^ data_size ^ OS_HDR_ID) != check_sum) {

        av_log(avctx, AV_LOG_ERROR, "OS header checksum mismatch!\n");

        return AVERROR_INVALIDDATA;

    }



    /* parse the bitstream header */

    bs_hdr = buf_ptr;



    if (bytestream_get_le16(&buf_ptr) != 32) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported codec version!\n");

        return AVERROR_INVALIDDATA;

    }



    ctx->frame_num   =  frame_num;

    ctx->frame_flags =  bytestream_get_le16(&buf_ptr);

    ctx->data_size   = (bytestream_get_le32(&buf_ptr) + 7) >> 3;

    ctx->cb_offset   = *buf_ptr++;



    if (ctx->data_size == 16)

        return 4;

    if (ctx->data_size > buf_size)

        ctx->data_size = buf_size;



    buf_ptr += 3; // skip reserved byte and checksum



    /* check frame dimensions */

    height = bytestream_get_le16(&buf_ptr);

    width  = bytestream_get_le16(&buf_ptr);

    if (av_image_check_size(width, height, 0, avctx))

        return AVERROR_INVALIDDATA;



    if (width != ctx->width || height != ctx->height) {

        av_dlog(avctx, "Frame dimensions changed!\n");



        ctx->width  = width;

        ctx->height = height;



        free_frame_buffers(ctx);

        allocate_frame_buffers(ctx, avctx);

        avcodec_set_dimensions(avctx, width, height);

    }



    y_offset = bytestream_get_le32(&buf_ptr);

    v_offset = bytestream_get_le32(&buf_ptr);

    u_offset = bytestream_get_le32(&buf_ptr);



    /* unfortunately there is no common order of planes in the buffer */

    /* so we use that sorting algo for determining planes data sizes  */

    starts[0] = y_offset;

    starts[1] = v_offset;

    starts[2] = u_offset;



    for (j = 0; j < 3; j++) {

        ends[j] = ctx->data_size;

        for (i = 2; i >= 0; i--)

            if (starts[i] < ends[j] && starts[i] > starts[j])

                ends[j] = starts[i];

    }



    ctx->y_data_size = ends[0] - starts[0];

    ctx->v_data_size = ends[1] - starts[1];

    ctx->u_data_size = ends[2] - starts[2];

    if (FFMAX3(y_offset, v_offset, u_offset) >= ctx->data_size - 16 ||

        FFMIN3(ctx->y_data_size, ctx->v_data_size, ctx->u_data_size) <= 0) {

        av_log(avctx, AV_LOG_ERROR, "One of the y/u/v offsets is invalid\n");

        return AVERROR_INVALIDDATA;

    }



    ctx->y_data_ptr = bs_hdr + y_offset;

    ctx->v_data_ptr = bs_hdr + v_offset;

    ctx->u_data_ptr = bs_hdr + u_offset;

    ctx->alt_quant  = buf_ptr + sizeof(uint32_t);



    if (ctx->data_size == 16) {

        av_log(avctx, AV_LOG_DEBUG, "Sync frame encountered!\n");

        return 16;

    }



    if (ctx->frame_flags & BS_8BIT_PEL) {

        av_log_ask_for_sample(avctx, "8-bit pixel format\n");

        return AVERROR_PATCHWELCOME;

    }



    if (ctx->frame_flags & BS_MV_X_HALF || ctx->frame_flags & BS_MV_Y_HALF) {

        av_log_ask_for_sample(avctx, "halfpel motion vectors\n");

        return AVERROR_PATCHWELCOME;

    }



    return 0;

}
