static int mpeg_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             const uint8_t *buf, int buf_size)

{

    Mpeg1Context *s = avctx->priv_data;

    AVFrame *picture = data;

    MpegEncContext *s2 = &s->mpeg_enc_ctx;

    dprintf(avctx, "fill_buffer\n");



    if (buf_size == 0 || (buf_size == 4 && AV_RB32(buf) == SEQ_END_CODE)) {

        /* special case for last picture */

        if (s2->low_delay==0 && s2->next_picture_ptr) {

            *picture= *(AVFrame*)s2->next_picture_ptr;

            s2->next_picture_ptr= NULL;



            *data_size = sizeof(AVFrame);

        }

        return buf_size;

    }



    if(s2->flags&CODEC_FLAG_TRUNCATED){

        int next= ff_mpeg1_find_frame_end(&s2->parse_context, buf, buf_size);



        if( ff_combine_frame(&s2->parse_context, next, (const uint8_t **)&buf, &buf_size) < 0 )

            return buf_size;

    }



#if 0

    if (s->repeat_field % 2 == 1) {

        s->repeat_field++;

        //fprintf(stderr,"\nRepeating last frame: %d -> %d! pict: %d %d", avctx->frame_number-1, avctx->frame_number,

        //        s2->picture_number, s->repeat_field);

        if (avctx->flags & CODEC_FLAG_REPEAT_FIELD) {

            *data_size = sizeof(AVPicture);

            goto the_end;

        }

    }

#endif



    if(s->mpeg_enc_ctx_allocated==0 && avctx->codec_tag == AV_RL32("VCR2"))

        vcr2_init_sequence(avctx);



    s->slice_count= 0;



    if(avctx->extradata && !avctx->frame_number)

        decode_chunks(avctx, picture, data_size, avctx->extradata, avctx->extradata_size);



    return decode_chunks(avctx, picture, data_size, buf, buf_size);

}
