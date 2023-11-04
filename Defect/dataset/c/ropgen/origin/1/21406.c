int ff_h263_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MpegEncContext *s = avctx->priv_data;

    int ret;

    AVFrame *pict = data;



#ifdef PRINT_FRAME_TIME

uint64_t time= rdtsc();

#endif

    s->flags= avctx->flags;

    s->flags2= avctx->flags2;



    /* no supplementary picture */

    if (buf_size == 0) {

        /* special case for last picture */

        if (s->low_delay==0 && s->next_picture_ptr) {

            *pict = s->next_picture_ptr->f;

            s->next_picture_ptr= NULL;



            *data_size = sizeof(AVFrame);

        }



        return 0;

    }



    if(s->flags&CODEC_FLAG_TRUNCATED){

        int next;



        if(CONFIG_MPEG4_DECODER && s->codec_id==CODEC_ID_MPEG4){

            next= ff_mpeg4_find_frame_end(&s->parse_context, buf, buf_size);

        }else if(CONFIG_H263_DECODER && s->codec_id==CODEC_ID_H263){

            next= ff_h263_find_frame_end(&s->parse_context, buf, buf_size);

        }else{

            av_log(s->avctx, AV_LOG_ERROR, "this codec does not support truncated bitstreams\n");

            return -1;

        }



        if( ff_combine_frame(&s->parse_context, next, (const uint8_t **)&buf, &buf_size) < 0 )

            return buf_size;

    }





retry:



    if(s->bitstream_buffer_size && (s->divx_packed || buf_size<20)){ //divx 5.01+/xvid frame reorder

        init_get_bits(&s->gb, s->bitstream_buffer, s->bitstream_buffer_size*8);

    }else

        init_get_bits(&s->gb, buf, buf_size*8);

    s->bitstream_buffer_size=0;



    if (!s->context_initialized) {

        if (ff_MPV_common_init(s) < 0) //we need the idct permutaton for reading a custom matrix

            return -1;

    }



    /* We need to set current_picture_ptr before reading the header,

     * otherwise we cannot store anyting in there */

    if (s->current_picture_ptr == NULL || s->current_picture_ptr->f.data[0]) {

        int i= ff_find_unused_picture(s, 0);

        if (i < 0)

            return i;

        s->current_picture_ptr= &s->picture[i];

    }



    /* let's go :-) */

    if (CONFIG_WMV2_DECODER && s->msmpeg4_version==5) {

        ret= ff_wmv2_decode_picture_header(s);

    } else if (CONFIG_MSMPEG4_DECODER && s->msmpeg4_version) {

        ret = ff_msmpeg4_decode_picture_header(s);

    } else if (CONFIG_MPEG4_DECODER && s->h263_pred) {

        if(s->avctx->extradata_size && s->picture_number==0){

            GetBitContext gb;



            init_get_bits(&gb, s->avctx->extradata, s->avctx->extradata_size*8);

            ret = ff_mpeg4_decode_picture_header(s, &gb);

        }

        ret = ff_mpeg4_decode_picture_header(s, &s->gb);

    } else if (CONFIG_H263I_DECODER && s->codec_id == CODEC_ID_H263I) {

        ret = ff_intel_h263_decode_picture_header(s);

    } else if (CONFIG_FLV_DECODER && s->h263_flv) {

        ret = ff_flv_decode_picture_header(s);

    } else {

        ret = ff_h263_decode_picture_header(s);

    }



    if(ret==FRAME_SKIPPED) return get_consumed_bytes(s, buf_size);



    /* skip if the header was thrashed */

    if (ret < 0){

        av_log(s->avctx, AV_LOG_ERROR, "header damaged\n");

        return -1;

    }



    avctx->has_b_frames= !s->low_delay;



    if(s->xvid_build==-1 && s->divx_version==-1 && s->lavc_build==-1){

        if(s->stream_codec_tag == AV_RL32("XVID") ||

           s->codec_tag == AV_RL32("XVID") || s->codec_tag == AV_RL32("XVIX") ||

           s->codec_tag == AV_RL32("RMP4") ||

           s->codec_tag == AV_RL32("SIPP")

           )

            s->xvid_build= 0;

#if 0

        if(s->codec_tag == AV_RL32("DIVX") && s->vo_type==0 && s->vol_control_parameters==1

           && s->padding_bug_score > 0 && s->low_delay) // XVID with modified fourcc

            s->xvid_build= 0;

#endif

    }



    if(s->xvid_build==-1 && s->divx_version==-1 && s->lavc_build==-1){

        if(s->codec_tag == AV_RL32("DIVX") && s->vo_type==0 && s->vol_control_parameters==0)

            s->divx_version= 400; //divx 4

    }



    if(s->xvid_build>=0 && s->divx_version>=0){

        s->divx_version=

        s->divx_build= -1;

    }



    if(s->workaround_bugs&FF_BUG_AUTODETECT){

        if(s->codec_tag == AV_RL32("XVIX"))

            s->workaround_bugs|= FF_BUG_XVID_ILACE;



        if(s->codec_tag == AV_RL32("UMP4")){

            s->workaround_bugs|= FF_BUG_UMP4;

        }



        if(s->divx_version>=500 && s->divx_build<1814){

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA;

        }



        if(s->divx_version>502 && s->divx_build<1814){

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA2;

        }



        if(s->xvid_build<=3U)

            s->padding_bug_score= 256*256*256*64;



        if(s->xvid_build<=1U)

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA;



        if(s->xvid_build<=12U)

            s->workaround_bugs|= FF_BUG_EDGE;



        if(s->xvid_build<=32U)

            s->workaround_bugs|= FF_BUG_DC_CLIP;



#define SET_QPEL_FUNC(postfix1, postfix2) \

    s->dsp.put_ ## postfix1 = ff_put_ ## postfix2;\

    s->dsp.put_no_rnd_ ## postfix1 = ff_put_no_rnd_ ## postfix2;\

    s->dsp.avg_ ## postfix1 = ff_avg_ ## postfix2;



        if(s->lavc_build<4653U)

            s->workaround_bugs|= FF_BUG_STD_QPEL;



        if(s->lavc_build<4655U)

            s->workaround_bugs|= FF_BUG_DIRECT_BLOCKSIZE;



        if(s->lavc_build<4670U){

            s->workaround_bugs|= FF_BUG_EDGE;

        }



        if(s->lavc_build<=4712U)

            s->workaround_bugs|= FF_BUG_DC_CLIP;



        if(s->divx_version>=0)

            s->workaround_bugs|= FF_BUG_DIRECT_BLOCKSIZE;

//printf("padding_bug_score: %d\n", s->padding_bug_score);

        if(s->divx_version==501 && s->divx_build==20020416)

            s->padding_bug_score= 256*256*256*64;



        if(s->divx_version<500U){

            s->workaround_bugs|= FF_BUG_EDGE;

        }



        if(s->divx_version>=0)

            s->workaround_bugs|= FF_BUG_HPEL_CHROMA;

#if 0

        if(s->divx_version==500)

            s->padding_bug_score= 256*256*256*64;



        /* very ugly XVID padding bug detection FIXME/XXX solve this differently

         * Let us hope this at least works.

         */

        if(   s->resync_marker==0 && s->data_partitioning==0 && s->divx_version==-1

           && s->codec_id==CODEC_ID_MPEG4 && s->vo_type==0)

            s->workaround_bugs|= FF_BUG_NO_PADDING;



        if(s->lavc_build<4609U) //FIXME not sure about the version num but a 4609 file seems ok

            s->workaround_bugs|= FF_BUG_NO_PADDING;

#endif

    }



    if(s->workaround_bugs& FF_BUG_STD_QPEL){

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 5], qpel16_mc11_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 7], qpel16_mc31_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 9], qpel16_mc12_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][11], qpel16_mc32_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][13], qpel16_mc13_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][15], qpel16_mc33_old_c)



        SET_QPEL_FUNC(qpel_pixels_tab[1][ 5], qpel8_mc11_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][ 7], qpel8_mc31_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][ 9], qpel8_mc12_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][11], qpel8_mc32_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][13], qpel8_mc13_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][15], qpel8_mc33_old_c)

    }



    if(avctx->debug & FF_DEBUG_BUGS)

        av_log(s->avctx, AV_LOG_DEBUG, "bugs: %X lavc_build:%d xvid_build:%d divx_version:%d divx_build:%d %s\n",

               s->workaround_bugs, s->lavc_build, s->xvid_build, s->divx_version, s->divx_build,

               s->divx_packed ? "p" : "");



#if HAVE_MMX

    if (s->codec_id == CODEC_ID_MPEG4 && s->xvid_build>=0 && avctx->idct_algo == FF_IDCT_AUTO && (av_get_cpu_flags() & AV_CPU_FLAG_MMX)) {

        avctx->idct_algo= FF_IDCT_XVIDMMX;

        avctx->coded_width= 0; // force reinit

//        ff_dsputil_init(&s->dsp, avctx);

        s->picture_number=0;

    }

#endif



        /* After H263 & mpeg4 header decode we have the height, width,*/

        /* and other parameters. So then we could init the picture   */

        /* FIXME: By the way H263 decoder is evolving it should have */

        /* an H263EncContext                                         */



    if (   s->width  != avctx->coded_width

        || s->height != avctx->coded_height) {

        /* H.263 could change picture size any time */

        ParseContext pc= s->parse_context; //FIXME move these demuxng hack to avformat

        s->parse_context.buffer=0;

        ff_MPV_common_end(s);

        s->parse_context= pc;

    }

    if (!s->context_initialized) {

        avcodec_set_dimensions(avctx, s->width, s->height);



        goto retry;

    }



    if((s->codec_id==CODEC_ID_H263 || s->codec_id==CODEC_ID_H263P || s->codec_id == CODEC_ID_H263I))

        s->gob_index = ff_h263_get_gob_height(s);



    // for skipping the frame

    s->current_picture.f.pict_type = s->pict_type;

    s->current_picture.f.key_frame = s->pict_type == AV_PICTURE_TYPE_I;



    /* skip B-frames if we don't have reference frames */

    if(s->last_picture_ptr==NULL && (s->pict_type==AV_PICTURE_TYPE_B || s->dropable)) return get_consumed_bytes(s, buf_size);

    if(   (avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type==AV_PICTURE_TYPE_B)

       || (avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type!=AV_PICTURE_TYPE_I)

       ||  avctx->skip_frame >= AVDISCARD_ALL)

        return get_consumed_bytes(s, buf_size);



    if(s->next_p_frame_damaged){

        if(s->pict_type==AV_PICTURE_TYPE_B)

            return get_consumed_bytes(s, buf_size);

        else

            s->next_p_frame_damaged=0;

    }



    if((s->avctx->flags2 & CODEC_FLAG2_FAST) && s->pict_type==AV_PICTURE_TYPE_B){

        s->me.qpel_put= s->dsp.put_2tap_qpel_pixels_tab;

        s->me.qpel_avg= s->dsp.avg_2tap_qpel_pixels_tab;

    }else if((!s->no_rounding) || s->pict_type==AV_PICTURE_TYPE_B){

        s->me.qpel_put= s->dsp.put_qpel_pixels_tab;

        s->me.qpel_avg= s->dsp.avg_qpel_pixels_tab;

    }else{

        s->me.qpel_put= s->dsp.put_no_rnd_qpel_pixels_tab;

        s->me.qpel_avg= s->dsp.avg_qpel_pixels_tab;

    }



    if(ff_MPV_frame_start(s, avctx) < 0)

        return -1;



    if (!s->divx_packed) ff_thread_finish_setup(avctx);



    if (CONFIG_MPEG4_VDPAU_DECODER && (s->avctx->codec->capabilities & CODEC_CAP_HWACCEL_VDPAU)) {

        ff_vdpau_mpeg4_decode_picture(s, s->gb.buffer, s->gb.buffer_end - s->gb.buffer);

        goto frame_end;

    }



    if (avctx->hwaccel) {

        if (avctx->hwaccel->start_frame(avctx, s->gb.buffer, s->gb.buffer_end - s->gb.buffer) < 0)

            return -1;

    }



    ff_er_frame_start(s);



    //the second part of the wmv2 header contains the MB skip bits which are stored in current_picture->mb_type

    //which is not available before ff_MPV_frame_start()

    if (CONFIG_WMV2_DECODER && s->msmpeg4_version==5){

        ret = ff_wmv2_decode_secondary_picture_header(s);

        if(ret<0) return ret;

        if(ret==1) goto intrax8_decoded;

    }



    /* decode each macroblock */

    s->mb_x=0;

    s->mb_y=0;



    ret = decode_slice(s);

    while(s->mb_y<s->mb_height){

        if(s->msmpeg4_version){

            if(s->slice_height==0 || s->mb_x!=0 || (s->mb_y%s->slice_height)!=0 || get_bits_count(&s->gb) > s->gb.size_in_bits)

                break;

        }else{

            int prev_x=s->mb_x, prev_y=s->mb_y;

            if(ff_h263_resync(s)<0)

                break;

            if (prev_y * s->mb_width + prev_x < s->mb_y * s->mb_width + s->mb_x)

                s->error_occurred = 1;

        }



        if(s->msmpeg4_version<4 && s->h263_pred)

            ff_mpeg4_clean_buffers(s);



        if (decode_slice(s) < 0) ret = AVERROR_INVALIDDATA;

    }



    if (s->msmpeg4_version && s->msmpeg4_version<4 && s->pict_type==AV_PICTURE_TYPE_I)

        if(!CONFIG_MSMPEG4_DECODER || ff_msmpeg4_decode_ext_header(s, buf_size) < 0){

            s->error_status_table[s->mb_num-1]= ER_MB_ERROR;

        }



    assert(s->bitstream_buffer_size==0);

frame_end:

    /* divx 5.01+ bistream reorder stuff */

    if(s->codec_id==CODEC_ID_MPEG4 && s->divx_packed){

        int current_pos= get_bits_count(&s->gb)>>3;

        int startcode_found=0;



        if(buf_size - current_pos > 5){

            int i;

            for(i=current_pos; i<buf_size-3; i++){

                if(buf[i]==0 && buf[i+1]==0 && buf[i+2]==1 && buf[i+3]==0xB6){

                    startcode_found=1;

                    break;

                }

            }

        }

        if(s->gb.buffer == s->bitstream_buffer && buf_size>7 && s->xvid_build>=0){ //xvid style

            startcode_found=1;

            current_pos=0;

        }



        if(startcode_found){

            av_fast_malloc(

                &s->bitstream_buffer,

                &s->allocated_bitstream_buffer_size,

                buf_size - current_pos + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!s->bitstream_buffer)

                return AVERROR(ENOMEM);

            memcpy(s->bitstream_buffer, buf + current_pos, buf_size - current_pos);

            s->bitstream_buffer_size= buf_size - current_pos;

        }

    }



intrax8_decoded:

    ff_er_frame_end(s);



    if (avctx->hwaccel) {

        if (avctx->hwaccel->end_frame(avctx) < 0)

            return -1;

    }



    ff_MPV_frame_end(s);



    assert(s->current_picture.f.pict_type == s->current_picture_ptr->f.pict_type);

    assert(s->current_picture.f.pict_type == s->pict_type);

    if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {

        *pict = s->current_picture_ptr->f;

    } else if (s->last_picture_ptr != NULL) {

        *pict = s->last_picture_ptr->f;

    }



    if(s->last_picture_ptr || s->low_delay){

        *data_size = sizeof(AVFrame);

        ff_print_debug_info(s, pict);

    }



#ifdef PRINT_FRAME_TIME

av_log(avctx, AV_LOG_DEBUG, "%"PRId64"\n", rdtsc()-time);

#endif



    return (ret && (avctx->err_recognition & AV_EF_EXPLODE))?ret:get_consumed_bytes(s, buf_size);

}