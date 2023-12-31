static int h263_decode_frame(AVCodecContext *avctx, 

                             void *data, int *data_size,

                             UINT8 *buf, int buf_size)

{

    MpegEncContext *s = avctx->priv_data;

    int ret;

    AVPicture *pict = data; 



#ifdef DEBUG

    printf("*****frame %d size=%d\n", avctx->frame_number, buf_size);

    printf("bytes=%x %x %x %x\n", buf[0], buf[1], buf[2], buf[3]);

#endif

    

    /* no supplementary picture */

    if (buf_size == 0) {

        *data_size = 0;

        return 0;

    }



    if(s->bitstream_buffer_size) //divx 5.01+ frame reorder

        init_get_bits(&s->gb, s->bitstream_buffer, s->bitstream_buffer_size);

    else

        init_get_bits(&s->gb, buf, buf_size);



    /* let's go :-) */

    if (s->h263_msmpeg4) {

        ret = msmpeg4_decode_picture_header(s);

    } else if (s->h263_pred) {

        ret = mpeg4_decode_picture_header(s);

        s->has_b_frames= !s->low_delay;

    } else if (s->h263_intel) {

        ret = intel_h263_decode_picture_header(s);

    } else {

        ret = h263_decode_picture_header(s);

    }



        /* After H263 & mpeg4 header decode we have the height, width,*/

        /* and other parameters. So then we could init the picture   */

        /* FIXME: By the way H263 decoder is evolving it should have */

        /* an H263EncContext                                         */

    if (!s->context_initialized) {

        avctx->width = s->width;

        avctx->height = s->height;

        avctx->aspect_ratio_info= s->aspect_ratio_info;

        if (MPV_common_init(s) < 0)

            return -1;

    } else if (s->width != avctx->width || s->height != avctx->height) {

        /* H.263 could change picture size any time */

        MPV_common_end(s);

        if (MPV_common_init(s) < 0)

            return -1;

    }



    if(ret==FRAME_SKIPED) return 0;

    if (ret < 0)

        return -1;

    /* skip b frames if we dont have reference frames */

    if(s->num_available_buffers<2 && s->pict_type==B_TYPE) return 0;

        

    MPV_frame_start(s);



#ifdef DEBUG

    printf("qscale=%d\n", s->qscale);

#endif



    /* decode each macroblock */

    s->block_wrap[0]=

    s->block_wrap[1]=

    s->block_wrap[2]=

    s->block_wrap[3]= s->mb_width*2 + 2;

    s->block_wrap[4]=

    s->block_wrap[5]= s->mb_width + 2;

    for(s->mb_y=0; s->mb_y < s->mb_height; s->mb_y++) {

        /* Check for GOB headers on H.263 */

        /* FIXME: In the future H.263+ will have intra prediction */

        /* and we are gonna need another way to detect MPEG4      */

        if (s->mb_y && !s->h263_pred) {

            s->first_gob_line = h263_decode_gob_header(s);

        }



        s->block_index[0]= s->block_wrap[0]*(s->mb_y*2 + 1) - 1;

        s->block_index[1]= s->block_wrap[0]*(s->mb_y*2 + 1);

        s->block_index[2]= s->block_wrap[0]*(s->mb_y*2 + 2) - 1;

        s->block_index[3]= s->block_wrap[0]*(s->mb_y*2 + 2);

        s->block_index[4]= s->block_wrap[4]*(s->mb_y + 1)                    + s->block_wrap[0]*(s->mb_height*2 + 2);

        s->block_index[5]= s->block_wrap[4]*(s->mb_y + 1 + s->mb_height + 2) + s->block_wrap[0]*(s->mb_height*2 + 2);

        for(s->mb_x=0; s->mb_x < s->mb_width; s->mb_x++) {

            s->block_index[0]+=2;

            s->block_index[1]+=2;

            s->block_index[2]+=2;

            s->block_index[3]+=2;

            s->block_index[4]++;

            s->block_index[5]++;

#ifdef DEBUG

            printf("**mb x=%d y=%d\n", s->mb_x, s->mb_y);

#endif

            //fprintf(stderr,"\nFrame: %d\tMB: %d",avctx->frame_number, (s->mb_y * s->mb_width) + s->mb_x);

            /* DCT & quantize */

            if (s->h263_msmpeg4) {

                msmpeg4_dc_scale(s);

            } else if (s->h263_pred) {

                h263_dc_scale(s);

            } else {

                /* default quantization values */

                s->y_dc_scale = 8;

                s->c_dc_scale = 8;

            }

            clear_blocks(s->block[0]);

            

            s->mv_dir = MV_DIR_FORWARD;

            s->mv_type = MV_TYPE_16X16; 

            if (s->h263_msmpeg4) {

		if (msmpeg4_decode_mb(s, s->block) < 0) {

		    fprintf(stderr,"\nError at MB: %d\n", (s->mb_y * s->mb_width) + s->mb_x);

                    return -1;

		}

            } else {

                if (h263_decode_mb(s, s->block) < 0) {

                    fprintf(stderr,"\nError at MB: %d\n", (s->mb_y * s->mb_width) + s->mb_x);

                    return -1;

                }

            }

            MPV_decode_mb(s, s->block);

        }

        if (    avctx->draw_horiz_band 

            && (s->num_available_buffers>=1 || (!s->has_b_frames)) ) {

            UINT8 *src_ptr[3];

            int y, h, offset;

            y = s->mb_y * 16;

            h = s->height - y;

            if (h > 16)

                h = 16;

            offset = y * s->linesize;

            if(s->pict_type==B_TYPE || (!s->has_b_frames)){

                src_ptr[0] = s->current_picture[0] + offset;

                src_ptr[1] = s->current_picture[1] + (offset >> 2);

                src_ptr[2] = s->current_picture[2] + (offset >> 2);

            } else {

                src_ptr[0] = s->last_picture[0] + offset;

                src_ptr[1] = s->last_picture[1] + (offset >> 2);

                src_ptr[2] = s->last_picture[2] + (offset >> 2);

            }

            avctx->draw_horiz_band(avctx, src_ptr, s->linesize,

                                   y, s->width, h);

        }

    }

    

    if (s->h263_msmpeg4 && s->msmpeg4_version<4 && s->pict_type==I_TYPE)

        if(msmpeg4_decode_ext_header(s, buf_size) < 0) return -1;

    

    /* divx 5.01+ bistream reorder stuff */

    if(s->h263_pred && s->bitstream_buffer_size==0){

        int current_pos= get_bits_count(&s->gb)/8;

        if(   buf_size - current_pos > 5 

           && buf_size - current_pos < BITSTREAM_BUFFER_SIZE){

            memcpy(s->bitstream_buffer, buf + current_pos, buf_size - current_pos);

            s->bitstream_buffer_size= buf_size - current_pos;

        }

    }else

        s->bitstream_buffer_size=0;

  

    MPV_frame_end(s);

    

    if(s->pict_type==B_TYPE || (!s->has_b_frames)){

        pict->data[0] = s->current_picture[0];

        pict->data[1] = s->current_picture[1];

        pict->data[2] = s->current_picture[2];

    } else {

        pict->data[0] = s->last_picture[0];

        pict->data[1] = s->last_picture[1];

        pict->data[2] = s->last_picture[2];

    }

    pict->linesize[0] = s->linesize;

    pict->linesize[1] = s->linesize / 2;

    pict->linesize[2] = s->linesize / 2;



    avctx->quality = s->qscale;



    /* Return the Picture timestamp as the frame number */

    /* we substract 1 because it is added on utils.c    */

    avctx->frame_number = s->picture_number - 1;



    /* dont output the last pic after seeking 

       note we allready added +1 for the current pix in MPV_frame_end(s) */

    if(s->num_available_buffers>=2 || (!s->has_b_frames))

        *data_size = sizeof(AVPicture);



    return buf_size;

}
