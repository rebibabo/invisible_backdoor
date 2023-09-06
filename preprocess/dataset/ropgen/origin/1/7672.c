static int flashsv_decode_frame(AVCodecContext *avctx, void *data,

                                int *data_size, AVPacket *avpkt)

{

    int buf_size       = avpkt->size;

    FlashSVContext *s  = avctx->priv_data;

    int h_blocks, v_blocks, h_part, v_part, i, j;

    GetBitContext gb;



    /* no supplementary picture */

    if (buf_size == 0)

        return 0;

    if (buf_size < 4)

        return -1;



    init_get_bits(&gb, avpkt->data, buf_size * 8);



    /* start to parse the bitstream */

    s->block_width  = 16 * (get_bits(&gb,  4) + 1);

    s->image_width  =       get_bits(&gb, 12);

    s->block_height = 16 * (get_bits(&gb,  4) + 1);

    s->image_height =       get_bits(&gb, 12);



    if (s->ver == 2) {

        skip_bits(&gb, 6);

        if (get_bits1(&gb)) {

            av_log_missing_feature(avctx, "iframe", 1);

            return AVERROR_PATCHWELCOME;


        if (get_bits1(&gb)) {

            av_log_missing_feature(avctx, "Custom palette", 1);

            return AVERROR_PATCHWELCOME;





    /* calculate number of blocks and size of border (partial) blocks */

    h_blocks = s->image_width  / s->block_width;

    h_part   = s->image_width  % s->block_width;

    v_blocks = s->image_height / s->block_height;

    v_part   = s->image_height % s->block_height;



    /* the block size could change between frames, make sure the buffer

     * is large enough, if not, get a larger one */

    if (s->block_size < s->block_width * s->block_height) {

        int tmpblock_size = 3 * s->block_width * s->block_height;



        s->tmpblock = av_realloc(s->tmpblock, tmpblock_size);

        if (!s->tmpblock) {

            av_log(avctx, AV_LOG_ERROR, "Can't allocate decompression buffer.\n");

            return AVERROR(ENOMEM);


        if (s->ver == 2) {

            s->deflate_block_size = calc_deflate_block_size(tmpblock_size);

            if (s->deflate_block_size <= 0) {

                av_log(avctx, AV_LOG_ERROR, "Can't determine deflate buffer size.\n");

                return -1;


            s->deflate_block = av_realloc(s->deflate_block, s->deflate_block_size);

            if (!s->deflate_block) {

                av_log(avctx, AV_LOG_ERROR, "Can't allocate deflate buffer.\n");

                return AVERROR(ENOMEM);




    s->block_size = s->block_width * s->block_height;



    /* initialize the image size once */

    if (avctx->width == 0 && avctx->height == 0) {

        avctx->width  = s->image_width;

        avctx->height = s->image_height;




    /* check for changes of image width and image height */

    if (avctx->width != s->image_width || avctx->height != s->image_height) {


               "Frame width or height differs from first frame!\n");

        av_log(avctx, AV_LOG_ERROR, "fh = %d, fv %d  vs  ch = %d, cv = %d\n",

               avctx->height, avctx->width, s->image_height, s->image_width);





    /* we care for keyframes only in Screen Video v2 */

    s->is_keyframe = (avpkt->flags & AV_PKT_FLAG_KEY) && (s->ver == 2);

    if (s->is_keyframe) {

        s->keyframedata = av_realloc(s->keyframedata, avpkt->size);

        memcpy(s->keyframedata, avpkt->data, avpkt->size);

        s->blocks = av_realloc(s->blocks,

                               (v_blocks + !!v_part) * (h_blocks + !!h_part)

                               * sizeof(s->blocks[0]));




    av_dlog(avctx, "image: %dx%d block: %dx%d num: %dx%d part: %dx%d\n",

            s->image_width, s->image_height, s->block_width, s->block_height,

            h_blocks, v_blocks, h_part, v_part);



    s->frame.reference    = 3;

    s->frame.buffer_hints = FF_BUFFER_HINTS_VALID    |

                            FF_BUFFER_HINTS_PRESERVE |

                            FF_BUFFER_HINTS_REUSABLE;

    if (avctx->reget_buffer(avctx, &s->frame) < 0) {

        av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

        return -1;




    /* loop over all block columns */

    for (j = 0; j < v_blocks + (v_part ? 1 : 0); j++) {



        int y_pos  = j * s->block_height; // vertical position in frame

        int cur_blk_height = (j < v_blocks) ? s->block_height : v_part;



        /* loop over all block rows */

        for (i = 0; i < h_blocks + (h_part ? 1 : 0); i++) {

            int x_pos = i * s->block_width; // horizontal position in frame

            int cur_blk_width = (i < h_blocks) ? s->block_width : h_part;

            int has_diff = 0;



            /* get the size of the compressed zlib chunk */

            int size = get_bits(&gb, 16);



            s->color_depth    = 0;

            s->zlibprime_curr = 0;

            s->zlibprime_prev = 0;

            s->diff_start     = 0;

            s->diff_height    = cur_blk_height;



            if (8 * size > get_bits_left(&gb)) {

                avctx->release_buffer(avctx, &s->frame);

                s->frame.data[0] = NULL;





            if (s->ver == 2 && size) {

                skip_bits(&gb, 3);

                s->color_depth    = get_bits(&gb, 2);

                has_diff          = get_bits1(&gb);

                s->zlibprime_curr = get_bits1(&gb);

                s->zlibprime_prev = get_bits1(&gb);



                if (s->color_depth != 0 && s->color_depth != 2) {


                           "%dx%d invalid color depth %d\n", i, j, s->color_depth);





                if (has_diff) {






                    s->diff_start  = get_bits(&gb, 8);

                    s->diff_height = get_bits(&gb, 8);

                    av_log(avctx, AV_LOG_DEBUG,

                           "%dx%d diff start %d height %d\n",

                           i, j, s->diff_start, s->diff_height);

                    size -= 2;




                if (s->zlibprime_prev)

                    av_log(avctx, AV_LOG_DEBUG, "%dx%d zlibprime_prev\n", i, j);



                if (s->zlibprime_curr) {

                    int col = get_bits(&gb, 8);

                    int row = get_bits(&gb, 8);

                    av_log(avctx, AV_LOG_DEBUG, "%dx%d zlibprime_curr %dx%d\n", i, j, col, row);

                    size -= 2;

                    av_log_missing_feature(avctx, "zlibprime_curr", 1);

                    return AVERROR_PATCHWELCOME;


                size--; // account for flags byte




            if (has_diff) {

                int k;

                int off = (s->image_height - y_pos - 1) * s->frame.linesize[0];



                for (k = 0; k < cur_blk_height; k++)

                    memcpy(s->frame.data[0] + off - k*s->frame.linesize[0] + x_pos*3,

                           s->keyframe + off - k*s->frame.linesize[0] + x_pos*3,

                           cur_blk_width * 3);




            /* skip unchanged blocks, which have size 0 */

            if (size) {

                if (flashsv_decode_block(avctx, avpkt, &gb, size,

                                         cur_blk_width, cur_blk_height,

                                         x_pos, y_pos,

                                         i + j * (h_blocks + !!h_part)))


                           "error in decompression of block %dx%d\n", i, j);




    if (s->is_keyframe && s->ver == 2) {


            s->keyframe = av_malloc(s->frame.linesize[0] * avctx->height);


                av_log(avctx, AV_LOG_ERROR, "Cannot allocate image data\n");

                return AVERROR(ENOMEM);



        memcpy(s->keyframe, s->frame.data[0], s->frame.linesize[0] * avctx->height);




    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->frame;



    if ((get_bits_count(&gb) / 8) != buf_size)

        av_log(avctx, AV_LOG_ERROR, "buffer not fully consumed (%d != %d)\n",

               buf_size, (get_bits_count(&gb) / 8));



    /* report that the buffer was completely consumed */

    return buf_size;