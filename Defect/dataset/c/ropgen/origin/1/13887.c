static int vp3_decode_frame(AVCodecContext *avctx,

                            void *data, int *got_frame,

                            AVPacket *avpkt)

{

    const uint8_t *buf  = avpkt->data;

    int buf_size        = avpkt->size;

    Vp3DecodeContext *s = avctx->priv_data;

    GetBitContext gb;

    int i, ret;



    if ((ret = init_get_bits8(&gb, buf, buf_size)) < 0)

        return ret;



#if CONFIG_THEORA_DECODER

    if (s->theora && get_bits1(&gb)) {

        int type = get_bits(&gb, 7);

        skip_bits_long(&gb, 6*8); /* "theora" */



        if (s->avctx->active_thread_type&FF_THREAD_FRAME) {

            av_log(avctx, AV_LOG_ERROR, "midstream reconfiguration with multithreading is unsupported, try -threads 1\n");

            return AVERROR_PATCHWELCOME;

        }

        if (type == 0) {

            vp3_decode_end(avctx);

            ret = theora_decode_header(avctx, &gb);



            if (ret < 0) {

                vp3_decode_end(avctx);

            } else

                ret = vp3_decode_init(avctx);

            return ret;

        } else if (type == 2) {

            ret = theora_decode_tables(avctx, &gb);

            if (ret < 0) {

                vp3_decode_end(avctx);

            } else

                ret = vp3_decode_init(avctx);

            return ret;

        }



        av_log(avctx, AV_LOG_ERROR,

               "Header packet passed to frame decoder, skipping\n");

        return -1;

    }

#endif



    s->keyframe = !get_bits1(&gb);

    if (!s->all_fragments) {

        av_log(avctx, AV_LOG_ERROR, "Data packet without prior valid headers\n");

        return -1;

    }

    if (!s->theora)

        skip_bits(&gb, 1);

    for (i = 0; i < 3; i++)

        s->last_qps[i] = s->qps[i];



    s->nqps = 0;

    do {

        s->qps[s->nqps++] = get_bits(&gb, 6);

    } while (s->theora >= 0x030200 && s->nqps < 3 && get_bits1(&gb));

    for (i = s->nqps; i < 3; i++)

        s->qps[i] = -1;



    if (s->avctx->debug & FF_DEBUG_PICT_INFO)

        av_log(s->avctx, AV_LOG_INFO, " VP3 %sframe #%d: Q index = %d\n",

               s->keyframe ? "key" : "", avctx->frame_number + 1, s->qps[0]);



    s->skip_loop_filter = !s->filter_limit_values[s->qps[0]] ||

                          avctx->skip_loop_filter >= (s->keyframe ? AVDISCARD_ALL

                                                                  : AVDISCARD_NONKEY);



    if (s->qps[0] != s->last_qps[0])

        init_loop_filter(s);



    for (i = 0; i < s->nqps; i++)

        // reinit all dequantizers if the first one changed, because

        // the DC of the first quantizer must be used for all matrices

        if (s->qps[i] != s->last_qps[i] || s->qps[0] != s->last_qps[0])

            init_dequantizer(s, i);



    if (avctx->skip_frame >= AVDISCARD_NONKEY && !s->keyframe)

        return buf_size;



    s->current_frame.f->pict_type = s->keyframe ? AV_PICTURE_TYPE_I

                                                : AV_PICTURE_TYPE_P;

    s->current_frame.f->key_frame = s->keyframe;

    if (ff_thread_get_buffer(avctx, &s->current_frame, AV_GET_BUFFER_FLAG_REF) < 0)

        goto error;



    if (!s->edge_emu_buffer)

        s->edge_emu_buffer = av_malloc(9 * FFABS(s->current_frame.f->linesize[0]));



    if (s->keyframe) {

        if (!s->theora) {

            skip_bits(&gb, 4); /* width code */

            skip_bits(&gb, 4); /* height code */

            if (s->version) {

                s->version = get_bits(&gb, 5);

                if (avctx->frame_number == 0)

                    av_log(s->avctx, AV_LOG_DEBUG,

                           "VP version: %d\n", s->version);

            }

        }

        if (s->version || s->theora) {

            if (get_bits1(&gb))

                av_log(s->avctx, AV_LOG_ERROR,

                       "Warning, unsupported keyframe coding type?!\n");

            skip_bits(&gb, 2); /* reserved? */

        }

    } else {

        if (!s->golden_frame.f->data[0]) {

            av_log(s->avctx, AV_LOG_WARNING,

                   "vp3: first frame not a keyframe\n");



            s->golden_frame.f->pict_type = AV_PICTURE_TYPE_I;

            if (ff_thread_get_buffer(avctx, &s->golden_frame,

                                     AV_GET_BUFFER_FLAG_REF) < 0)

                goto error;

            ff_thread_release_buffer(avctx, &s->last_frame);

            if ((ret = ff_thread_ref_frame(&s->last_frame,

                                           &s->golden_frame)) < 0)

                goto error;

            ff_thread_report_progress(&s->last_frame, INT_MAX, 0);

        }

    }



    memset(s->all_fragments, 0, s->fragment_count * sizeof(Vp3Fragment));

    ff_thread_finish_setup(avctx);



    if (unpack_superblocks(s, &gb)) {

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_superblocks\n");

        goto error;

    }

    if (unpack_modes(s, &gb)) {

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_modes\n");

        goto error;

    }

    if (unpack_vectors(s, &gb)) {

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_vectors\n");

        goto error;

    }

    if (unpack_block_qpis(s, &gb)) {

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_block_qpis\n");

        goto error;

    }

    if (unpack_dct_coeffs(s, &gb)) {

        av_log(s->avctx, AV_LOG_ERROR, "error in unpack_dct_coeffs\n");

        goto error;

    }



    for (i = 0; i < 3; i++) {

        int height = s->height >> (i && s->chroma_y_shift);

        if (s->flipped_image)

            s->data_offset[i] = 0;

        else

            s->data_offset[i] = (height - 1) * s->current_frame.f->linesize[i];

    }



    s->last_slice_end = 0;

    for (i = 0; i < s->c_superblock_height; i++)

        render_slice(s, i);



    // filter the last row

    for (i = 0; i < 3; i++) {

        int row = (s->height >> (3 + (i && s->chroma_y_shift))) - 1;

        apply_loop_filter(s, i, row, row + 1);

    }

    vp3_draw_horiz_band(s, s->height);



    /* output frame, offset as needed */

    if ((ret = av_frame_ref(data, s->current_frame.f)) < 0)

        return ret;

    for (i = 0; i < 3; i++) {

        AVFrame *dst = data;

        int off = (s->offset_x >> (i && s->chroma_y_shift)) +

                  (s->offset_y >> (i && s->chroma_y_shift)) * dst->linesize[i];

        dst->data[i] += off;

    }

    *got_frame = 1;



    if (!HAVE_THREADS || !(s->avctx->active_thread_type & FF_THREAD_FRAME)) {

        ret = update_frames(avctx);

        if (ret < 0)

            return ret;

    }



    return buf_size;



error:

    ff_thread_report_progress(&s->current_frame, INT_MAX, 0);



    if (!HAVE_THREADS || !(s->avctx->active_thread_type & FF_THREAD_FRAME))

        av_frame_unref(s->current_frame.f);



    return -1;

}
