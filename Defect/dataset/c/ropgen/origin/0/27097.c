static int h264_slice_header_init(H264Context *h)

{

    int nb_slices = (HAVE_THREADS &&

                     h->avctx->active_thread_type & FF_THREAD_SLICE) ?

                    h->avctx->thread_count : 1;

    int i, ret;



    ff_set_sar(h->avctx, h->sps.sar);

    av_pix_fmt_get_chroma_sub_sample(h->avctx->pix_fmt,

                                     &h->chroma_x_shift, &h->chroma_y_shift);



    if (h->sps.timing_info_present_flag) {

        int64_t den = h->sps.time_scale;

        if (h->x264_build < 44U)

            den *= 2;

        av_reduce(&h->avctx->framerate.den, &h->avctx->framerate.num,

                  h->sps.num_units_in_tick, den, 1 << 30);

    }



    ff_h264_free_tables(h);



    h->first_field           = 0;

    h->prev_interlaced_frame = 1;



    init_scan_tables(h);

    ret = ff_h264_alloc_tables(h);

    if (ret < 0) {

        av_log(h->avctx, AV_LOG_ERROR, "Could not allocate memory\n");

        return ret;

    }



    if (h->sps.bit_depth_luma < 8 || h->sps.bit_depth_luma > 10) {

        av_log(h->avctx, AV_LOG_ERROR, "Unsupported bit depth %d\n",

               h->sps.bit_depth_luma);

        return AVERROR_INVALIDDATA;

    }



    h->avctx->bits_per_raw_sample = h->sps.bit_depth_luma;

    h->pixel_shift                = h->sps.bit_depth_luma > 8;

    h->chroma_format_idc          = h->sps.chroma_format_idc;

    h->bit_depth_luma             = h->sps.bit_depth_luma;



    ff_h264dsp_init(&h->h264dsp, h->sps.bit_depth_luma,

                    h->sps.chroma_format_idc);

    ff_h264chroma_init(&h->h264chroma, h->sps.bit_depth_chroma);

    ff_h264qpel_init(&h->h264qpel, h->sps.bit_depth_luma);

    ff_h264_pred_init(&h->hpc, h->avctx->codec_id, h->sps.bit_depth_luma,

                      h->sps.chroma_format_idc);

    ff_videodsp_init(&h->vdsp, h->sps.bit_depth_luma);



    if (nb_slices > H264_MAX_THREADS || (nb_slices > h->mb_height && h->mb_height)) {

        int max_slices;

        if (h->mb_height)

            max_slices = FFMIN(H264_MAX_THREADS, h->mb_height);

        else

            max_slices = H264_MAX_THREADS;

        av_log(h->avctx, AV_LOG_WARNING, "too many threads/slices %d,"

               " reducing to %d\n", nb_slices, max_slices);

        nb_slices = max_slices;

    }

    h->slice_context_count = nb_slices;



    if (!HAVE_THREADS || !(h->avctx->active_thread_type & FF_THREAD_SLICE)) {

        ret = ff_h264_slice_context_init(h, &h->slice_ctx[0]);

        if (ret < 0) {

            av_log(h->avctx, AV_LOG_ERROR, "context_init() failed.\n");

            return ret;

        }

    } else {

        for (i = 0; i < h->slice_context_count; i++) {

            H264SliceContext *sl = &h->slice_ctx[i];



            sl->h264               = h;

            sl->intra4x4_pred_mode = h->intra4x4_pred_mode + i * 8 * 2 * h->mb_stride;

            sl->mvd_table[0]       = h->mvd_table[0]       + i * 8 * 2 * h->mb_stride;

            sl->mvd_table[1]       = h->mvd_table[1]       + i * 8 * 2 * h->mb_stride;



            if ((ret = ff_h264_slice_context_init(h, sl)) < 0) {

                av_log(h->avctx, AV_LOG_ERROR, "context_init() failed.\n");

                return ret;

            }

        }

    }



    h->context_initialized = 1;



    return 0;

}
