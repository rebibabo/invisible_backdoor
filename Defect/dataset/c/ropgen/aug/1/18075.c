int MPV_frame_start(MpegEncContext *s, AVCodecContext *avctx)

{

    int i;

    Picture *pic;

    s->mb_skipped = 0;



    assert(s->last_picture_ptr==NULL || s->out_format != FMT_H264 || s->codec_id == CODEC_ID_SVQ3);



    /* mark&release old frames */

    if (s->pict_type != AV_PICTURE_TYPE_B && s->last_picture_ptr && s->last_picture_ptr != s->next_picture_ptr && s->last_picture_ptr->f.data[0]) {

      if(s->out_format != FMT_H264 || s->codec_id == CODEC_ID_SVQ3){

          free_frame_buffer(s, s->last_picture_ptr);



        /* release forgotten pictures */

        /* if(mpeg124/h263) */

        if(!s->encoding){

            for(i=0; i<s->picture_count; i++){

                if (s->picture[i].f.data[0] && &s->picture[i] != s->next_picture_ptr && s->picture[i].f.reference) {

                    av_log(avctx, AV_LOG_ERROR, "releasing zombie picture\n");

                    free_frame_buffer(s, &s->picture[i]);

                }

            }

        }

      }

    }



    if(!s->encoding){

        ff_release_unused_pictures(s, 1);



        if (s->current_picture_ptr && s->current_picture_ptr->f.data[0] == NULL)

            pic= s->current_picture_ptr; //we already have a unused image (maybe it was set before reading the header)

        else{

            i= ff_find_unused_picture(s, 0);

            pic= &s->picture[i];

        }



        pic->f.reference = 0;

        if (!s->dropable){

            if (s->codec_id == CODEC_ID_H264)

                pic->f.reference = s->picture_structure;

            else if (s->pict_type != AV_PICTURE_TYPE_B)

                pic->f.reference = 3;

        }



        pic->f.coded_picture_number = s->coded_picture_number++;



        if(ff_alloc_picture(s, pic, 0) < 0)

            return -1;



        s->current_picture_ptr= pic;

        //FIXME use only the vars from current_pic

        s->current_picture_ptr->f.top_field_first = s->top_field_first;

        if(s->codec_id == CODEC_ID_MPEG1VIDEO || s->codec_id == CODEC_ID_MPEG2VIDEO) {

            if(s->picture_structure != PICT_FRAME)

                s->current_picture_ptr->f.top_field_first = (s->picture_structure == PICT_TOP_FIELD) == s->first_field;

        }

        s->current_picture_ptr->f.interlaced_frame = !s->progressive_frame && !s->progressive_sequence;

        s->current_picture_ptr->field_picture = s->picture_structure != PICT_FRAME;

    }



    s->current_picture_ptr->f.pict_type = s->pict_type;

//    if(s->flags && CODEC_FLAG_QSCALE)

  //      s->current_picture_ptr->quality= s->new_picture_ptr->quality;

    s->current_picture_ptr->f.key_frame = s->pict_type == AV_PICTURE_TYPE_I;



    ff_copy_picture(&s->current_picture, s->current_picture_ptr);



    if (s->pict_type != AV_PICTURE_TYPE_B) {

        s->last_picture_ptr= s->next_picture_ptr;

        if(!s->dropable)

            s->next_picture_ptr= s->current_picture_ptr;

    }

/*    av_log(s->avctx, AV_LOG_DEBUG, "L%p N%p C%p L%p N%p C%p type:%d drop:%d\n", s->last_picture_ptr, s->next_picture_ptr,s->current_picture_ptr,

        s->last_picture_ptr    ? s->last_picture_ptr->f.data[0]    : NULL,

        s->next_picture_ptr    ? s->next_picture_ptr->f.data[0]    : NULL,

        s->current_picture_ptr ? s->current_picture_ptr->f.data[0] : NULL,

        s->pict_type, s->dropable);*/



    if(s->codec_id != CODEC_ID_H264){

        if ((s->last_picture_ptr == NULL || s->last_picture_ptr->f.data[0] == NULL) &&

           (s->pict_type!=AV_PICTURE_TYPE_I || s->picture_structure != PICT_FRAME)){

            if (s->pict_type != AV_PICTURE_TYPE_I)

                av_log(avctx, AV_LOG_ERROR, "warning: first frame is no keyframe\n");

            else if (s->picture_structure != PICT_FRAME)

                av_log(avctx, AV_LOG_INFO, "allocate dummy last picture for field based first keyframe\n");



            /* Allocate a dummy frame */

            i= ff_find_unused_picture(s, 0);

            s->last_picture_ptr= &s->picture[i];

            if(ff_alloc_picture(s, s->last_picture_ptr, 0) < 0)

                return -1;

            ff_thread_report_progress((AVFrame*)s->last_picture_ptr, INT_MAX, 0);

            ff_thread_report_progress((AVFrame*)s->last_picture_ptr, INT_MAX, 1);

        }

        if ((s->next_picture_ptr == NULL || s->next_picture_ptr->f.data[0] == NULL) && s->pict_type == AV_PICTURE_TYPE_B) {

            /* Allocate a dummy frame */

            i= ff_find_unused_picture(s, 0);

            s->next_picture_ptr= &s->picture[i];

            if(ff_alloc_picture(s, s->next_picture_ptr, 0) < 0)

                return -1;

            ff_thread_report_progress((AVFrame*)s->next_picture_ptr, INT_MAX, 0);

            ff_thread_report_progress((AVFrame*)s->next_picture_ptr, INT_MAX, 1);

        }

    }



    if(s->last_picture_ptr) ff_copy_picture(&s->last_picture, s->last_picture_ptr);

    if(s->next_picture_ptr) ff_copy_picture(&s->next_picture, s->next_picture_ptr);



    assert(s->pict_type == AV_PICTURE_TYPE_I || (s->last_picture_ptr && s->last_picture_ptr->f.data[0]));



    if(s->picture_structure!=PICT_FRAME && s->out_format != FMT_H264){

        int i;

        for(i=0; i<4; i++){

            if(s->picture_structure == PICT_BOTTOM_FIELD){

                 s->current_picture.f.data[i] += s->current_picture.f.linesize[i];

            }

            s->current_picture.f.linesize[i] *= 2;

            s->last_picture.f.linesize[i]    *= 2;

            s->next_picture.f.linesize[i]    *= 2;

        }

    }



    s->error_recognition= avctx->error_recognition;



    /* set dequantizer, we can't do it during init as it might change for mpeg4

       and we can't do it in the header decode as init is not called for mpeg4 there yet */

    if(s->mpeg_quant || s->codec_id == CODEC_ID_MPEG2VIDEO){

        s->dct_unquantize_intra = s->dct_unquantize_mpeg2_intra;

        s->dct_unquantize_inter = s->dct_unquantize_mpeg2_inter;

    }else if(s->out_format == FMT_H263 || s->out_format == FMT_H261){

        s->dct_unquantize_intra = s->dct_unquantize_h263_intra;

        s->dct_unquantize_inter = s->dct_unquantize_h263_inter;

    }else{

        s->dct_unquantize_intra = s->dct_unquantize_mpeg1_intra;

        s->dct_unquantize_inter = s->dct_unquantize_mpeg1_inter;

    }



    if(s->dct_error_sum){

        assert(s->avctx->noise_reduction && s->encoding);



        update_noise_reduction(s);

    }



    if(CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration)

        return ff_xvmc_field_start(s, avctx);



    return 0;

}
