void ff_h264_pred_direct_motion(H264Context * const h, int *mb_type){

    MpegEncContext * const s = &h->s;

    int b8_stride = h->b8_stride;

    int b4_stride = h->b_stride;

    int mb_xy = h->mb_xy;

    int mb_type_col[2];

    const int16_t (*l1mv0)[2], (*l1mv1)[2];

    const int8_t *l1ref0, *l1ref1;

    const int is_b8x8 = IS_8X8(*mb_type);

    unsigned int sub_mb_type;

    int i8, i4;



    assert(h->ref_list[1][0].reference&3);



#define MB_TYPE_16x16_OR_INTRA (MB_TYPE_16x16|MB_TYPE_INTRA4x4|MB_TYPE_INTRA16x16|MB_TYPE_INTRA_PCM)



    if(IS_INTERLACED(h->ref_list[1][0].mb_type[mb_xy])){ // AFL/AFR/FR/FL -> AFL/FL

        if(!IS_INTERLACED(*mb_type)){                    //     AFR/FR    -> AFL/FL

            mb_xy= s->mb_x + ((s->mb_y&~1) + h->col_parity)*s->mb_stride;

            b8_stride = 0;

        }else{

            mb_xy += h->col_fieldoff; // non zero for FL -> FL & differ parity

        }

        goto single_col;

    }else{                                               // AFL/AFR/FR/FL -> AFR/FR

        if(IS_INTERLACED(*mb_type)){                     // AFL       /FL -> AFR/FR

            mb_xy= s->mb_x + (s->mb_y&~1)*s->mb_stride;

            mb_type_col[0] = h->ref_list[1][0].mb_type[mb_xy];

            mb_type_col[1] = h->ref_list[1][0].mb_type[mb_xy + s->mb_stride];

            b8_stride *= 3;

            b4_stride *= 6;



            sub_mb_type = MB_TYPE_16x16|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_SUB_8x8 */

            if(    (mb_type_col[0] & MB_TYPE_16x16_OR_INTRA)

                && (mb_type_col[1] & MB_TYPE_16x16_OR_INTRA)

                && !is_b8x8){

                *mb_type   |= MB_TYPE_16x8 |MB_TYPE_L0L1|MB_TYPE_DIRECT2; /* B_16x8 */

            }else{

                *mb_type   |= MB_TYPE_8x8|MB_TYPE_L0L1;

            }

        }else{                                           //     AFR/FR    -> AFR/FR

single_col:

            mb_type_col[0] =

            mb_type_col[1] = h->ref_list[1][0].mb_type[mb_xy];

            if(IS_8X8(mb_type_col[0]) && !h->sps.direct_8x8_inference_flag){

                /* FIXME save sub mb types from previous frames (or derive from MVs)

                * so we know exactly what block size to use */

                sub_mb_type = MB_TYPE_8x8|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_SUB_4x4 */

                *mb_type   |= MB_TYPE_8x8|MB_TYPE_L0L1;

            }else if(!is_b8x8 && (mb_type_col[0] & MB_TYPE_16x16_OR_INTRA)){

                sub_mb_type = MB_TYPE_16x16|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_SUB_8x8 */

                *mb_type   |= MB_TYPE_16x16|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_16x16 */

            }else if(!is_b8x8 && (mb_type_col[0] & (MB_TYPE_16x8|MB_TYPE_8x16))){

                sub_mb_type = MB_TYPE_16x16|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_SUB_8x8 */

                *mb_type   |= MB_TYPE_L0L1|MB_TYPE_DIRECT2 | (mb_type_col[0] & (MB_TYPE_16x8|MB_TYPE_8x16));

            }else{

                sub_mb_type = MB_TYPE_16x16|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_SUB_8x8 */

                *mb_type   |= MB_TYPE_8x8|MB_TYPE_L0L1;

            }

        }

    }



    l1mv0  = &h->ref_list[1][0].motion_val[0][h->mb2b_xy [mb_xy]];

    l1mv1  = &h->ref_list[1][0].motion_val[1][h->mb2b_xy [mb_xy]];

    l1ref0 = &h->ref_list[1][0].ref_index [0][h->mb2b8_xy[mb_xy]];

    l1ref1 = &h->ref_list[1][0].ref_index [1][h->mb2b8_xy[mb_xy]];

    if(!b8_stride){

        if(s->mb_y&1){

            l1ref0 += h->b8_stride;

            l1ref1 += h->b8_stride;

            l1mv0  +=  2*b4_stride;

            l1mv1  +=  2*b4_stride;

        }

    }



    if(h->direct_spatial_mv_pred){

        int ref[2];

        int mv[2][2];

        int list;



        /* FIXME interlacing + spatial direct uses wrong colocated block positions */



        /* ref = min(neighbors) */

        for(list=0; list<2; list++){

            int refa = h->ref_cache[list][scan8[0] - 1];

            int refb = h->ref_cache[list][scan8[0] - 8];

            int refc = h->ref_cache[list][scan8[0] - 8 + 4];

            if(refc == PART_NOT_AVAILABLE)

                refc = h->ref_cache[list][scan8[0] - 8 - 1];

            ref[list] = FFMIN3((unsigned)refa, (unsigned)refb, (unsigned)refc);

            if(ref[list] >= 0){

                pred_motion(h, 0, 4, list, ref[list], &mv[list][0], &mv[list][1]);

            }else{

                int mask= ~(MB_TYPE_L0 << (2*list));

                mv[list][0] = mv[list][1] = 0;

                ref[list] = -1;

                if(!is_b8x8)

                    *mb_type &= mask;

                sub_mb_type &= mask;

            }

        }

        if(ref[0] < 0 && ref[1] < 0){

            ref[0] = ref[1] = 0;

            if(!is_b8x8)

                *mb_type |= MB_TYPE_L0L1;

            sub_mb_type |= MB_TYPE_L0L1;

        }



        if(IS_INTERLACED(*mb_type) != IS_INTERLACED(mb_type_col[0])){

            for(i8=0; i8<4; i8++){

                int x8 = i8&1;

                int y8 = i8>>1;

                int xy8 = x8+y8*b8_stride;

                int xy4 = 3*x8+y8*b4_stride;

                int a,b;



                if(is_b8x8 && !IS_DIRECT(h->sub_mb_type[i8]))

                    continue;

                h->sub_mb_type[i8] = sub_mb_type;



                fill_rectangle(&h->ref_cache[0][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[0], 1);

                fill_rectangle(&h->ref_cache[1][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[1], 1);

                if(!IS_INTRA(mb_type_col[y8]) && !h->ref_list[1][0].long_ref

                   && (   (l1ref0[xy8] == 0 && FFABS(l1mv0[xy4][0]) <= 1 && FFABS(l1mv0[xy4][1]) <= 1)

                       || (l1ref0[xy8]  < 0 && l1ref1[xy8] == 0 && FFABS(l1mv1[xy4][0]) <= 1 && FFABS(l1mv1[xy4][1]) <= 1))){

                    a=b=0;

                    if(ref[0] > 0)

                        a= pack16to32(mv[0][0],mv[0][1]);

                    if(ref[1] > 0)

                        b= pack16to32(mv[1][0],mv[1][1]);

                }else{

                    a= pack16to32(mv[0][0],mv[0][1]);

                    b= pack16to32(mv[1][0],mv[1][1]);

                }

                fill_rectangle(&h->mv_cache[0][scan8[i8*4]], 2, 2, 8, a, 4);

                fill_rectangle(&h->mv_cache[1][scan8[i8*4]], 2, 2, 8, b, 4);

            }

        }else if(IS_16X16(*mb_type)){

            int a,b;



            fill_rectangle(&h->ref_cache[0][scan8[0]], 4, 4, 8, (uint8_t)ref[0], 1);

            fill_rectangle(&h->ref_cache[1][scan8[0]], 4, 4, 8, (uint8_t)ref[1], 1);

            if(!IS_INTRA(mb_type_col[0]) && !h->ref_list[1][0].long_ref

               && (   (l1ref0[0] == 0 && FFABS(l1mv0[0][0]) <= 1 && FFABS(l1mv0[0][1]) <= 1)

                   || (l1ref0[0]  < 0 && l1ref1[0] == 0 && FFABS(l1mv1[0][0]) <= 1 && FFABS(l1mv1[0][1]) <= 1

                       && (h->x264_build>33 || !h->x264_build)))){

                a=b=0;

                if(ref[0] > 0)

                    a= pack16to32(mv[0][0],mv[0][1]);

                if(ref[1] > 0)

                    b= pack16to32(mv[1][0],mv[1][1]);

            }else{

                a= pack16to32(mv[0][0],mv[0][1]);

                b= pack16to32(mv[1][0],mv[1][1]);

            }

            fill_rectangle(&h->mv_cache[0][scan8[0]], 4, 4, 8, a, 4);

            fill_rectangle(&h->mv_cache[1][scan8[0]], 4, 4, 8, b, 4);

        }else{

            for(i8=0; i8<4; i8++){

                const int x8 = i8&1;

                const int y8 = i8>>1;



                if(is_b8x8 && !IS_DIRECT(h->sub_mb_type[i8]))

                    continue;

                h->sub_mb_type[i8] = sub_mb_type;



                fill_rectangle(&h->mv_cache[0][scan8[i8*4]], 2, 2, 8, pack16to32(mv[0][0],mv[0][1]), 4);

                fill_rectangle(&h->mv_cache[1][scan8[i8*4]], 2, 2, 8, pack16to32(mv[1][0],mv[1][1]), 4);

                fill_rectangle(&h->ref_cache[0][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[0], 1);

                fill_rectangle(&h->ref_cache[1][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[1], 1);



                /* col_zero_flag */

                if(!IS_INTRA(mb_type_col[0]) && !h->ref_list[1][0].long_ref && (   l1ref0[x8 + y8*b8_stride] == 0

                                              || (l1ref0[x8 + y8*b8_stride] < 0 && l1ref1[x8 + y8*b8_stride] == 0

                                                  && (h->x264_build>33 || !h->x264_build)))){

                    const int16_t (*l1mv)[2]= l1ref0[x8 + y8*b8_stride] == 0 ? l1mv0 : l1mv1;

                    if(IS_SUB_8X8(sub_mb_type)){

                        const int16_t *mv_col = l1mv[x8*3 + y8*3*b4_stride];

                        if(FFABS(mv_col[0]) <= 1 && FFABS(mv_col[1]) <= 1){

                            if(ref[0] == 0)

                                fill_rectangle(&h->mv_cache[0][scan8[i8*4]], 2, 2, 8, 0, 4);

                            if(ref[1] == 0)

                                fill_rectangle(&h->mv_cache[1][scan8[i8*4]], 2, 2, 8, 0, 4);

                        }

                    }else

                    for(i4=0; i4<4; i4++){

                        const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*b4_stride];

                        if(FFABS(mv_col[0]) <= 1 && FFABS(mv_col[1]) <= 1){

                            if(ref[0] == 0)

                                *(uint32_t*)h->mv_cache[0][scan8[i8*4+i4]] = 0;

                            if(ref[1] == 0)

                                *(uint32_t*)h->mv_cache[1][scan8[i8*4+i4]] = 0;

                        }

                    }

                }

            }

        }

    }else{ /* direct temporal mv pred */

        const int *map_col_to_list0[2] = {h->map_col_to_list0[0], h->map_col_to_list0[1]};

        const int *dist_scale_factor = h->dist_scale_factor;

        int ref_offset= 0;



        if(FRAME_MBAFF && IS_INTERLACED(*mb_type)){

            map_col_to_list0[0] = h->map_col_to_list0_field[s->mb_y&1][0];

            map_col_to_list0[1] = h->map_col_to_list0_field[s->mb_y&1][1];

            dist_scale_factor   =h->dist_scale_factor_field[s->mb_y&1];

        }

        if(h->ref_list[1][0].mbaff && IS_INTERLACED(mb_type_col[0]))

            ref_offset += 16;



        if(IS_INTERLACED(*mb_type) != IS_INTERLACED(mb_type_col[0])){

            int y_shift  = 2*!IS_INTERLACED(*mb_type);

            assert(h->sps.direct_8x8_inference_flag);



            for(i8=0; i8<4; i8++){

                const int x8 = i8&1;

                const int y8 = i8>>1;

                int ref0, scale;

                const int16_t (*l1mv)[2]= l1mv0;



                if(is_b8x8 && !IS_DIRECT(h->sub_mb_type[i8]))

                    continue;

                h->sub_mb_type[i8] = sub_mb_type;



                fill_rectangle(&h->ref_cache[1][scan8[i8*4]], 2, 2, 8, 0, 1);

                if(IS_INTRA(mb_type_col[y8])){

                    fill_rectangle(&h->ref_cache[0][scan8[i8*4]], 2, 2, 8, 0, 1);

                    fill_rectangle(&h-> mv_cache[0][scan8[i8*4]], 2, 2, 8, 0, 4);

                    fill_rectangle(&h-> mv_cache[1][scan8[i8*4]], 2, 2, 8, 0, 4);

                    continue;

                }



                ref0 = l1ref0[x8 + y8*b8_stride];

                if(ref0 >= 0)

                    ref0 = map_col_to_list0[0][ref0 + ref_offset];

                else{

                    ref0 = map_col_to_list0[1][l1ref1[x8 + y8*b8_stride] + ref_offset];

                    l1mv= l1mv1;

                }

                scale = dist_scale_factor[ref0];

                fill_rectangle(&h->ref_cache[0][scan8[i8*4]], 2, 2, 8, ref0, 1);



                {

                    const int16_t *mv_col = l1mv[x8*3 + y8*b4_stride];

                    int my_col = (mv_col[1]<<y_shift)/2;

                    int mx = (scale * mv_col[0] + 128) >> 8;

                    int my = (scale * my_col + 128) >> 8;

                    fill_rectangle(&h->mv_cache[0][scan8[i8*4]], 2, 2, 8, pack16to32(mx,my), 4);

                    fill_rectangle(&h->mv_cache[1][scan8[i8*4]], 2, 2, 8, pack16to32(mx-mv_col[0],my-my_col), 4);

                }

            }

            return;

        }



        /* one-to-one mv scaling */



        if(IS_16X16(*mb_type)){

            int ref, mv0, mv1;



            fill_rectangle(&h->ref_cache[1][scan8[0]], 4, 4, 8, 0, 1);

            if(IS_INTRA(mb_type_col[0])){

                ref=mv0=mv1=0;

            }else{

                const int ref0 = l1ref0[0] >= 0 ? map_col_to_list0[0][l1ref0[0] + ref_offset]

                                                : map_col_to_list0[1][l1ref1[0] + ref_offset];

                const int scale = dist_scale_factor[ref0];

                const int16_t *mv_col = l1ref0[0] >= 0 ? l1mv0[0] : l1mv1[0];

                int mv_l0[2];

                mv_l0[0] = (scale * mv_col[0] + 128) >> 8;

                mv_l0[1] = (scale * mv_col[1] + 128) >> 8;

                ref= ref0;

                mv0= pack16to32(mv_l0[0],mv_l0[1]);

                mv1= pack16to32(mv_l0[0]-mv_col[0],mv_l0[1]-mv_col[1]);

            }

            fill_rectangle(&h->ref_cache[0][scan8[0]], 4, 4, 8, ref, 1);

            fill_rectangle(&h-> mv_cache[0][scan8[0]], 4, 4, 8, mv0, 4);

            fill_rectangle(&h-> mv_cache[1][scan8[0]], 4, 4, 8, mv1, 4);

        }else{

            for(i8=0; i8<4; i8++){

                const int x8 = i8&1;

                const int y8 = i8>>1;

                int ref0, scale;

                const int16_t (*l1mv)[2]= l1mv0;



                if(is_b8x8 && !IS_DIRECT(h->sub_mb_type[i8]))

                    continue;

                h->sub_mb_type[i8] = sub_mb_type;

                fill_rectangle(&h->ref_cache[1][scan8[i8*4]], 2, 2, 8, 0, 1);

                if(IS_INTRA(mb_type_col[0])){

                    fill_rectangle(&h->ref_cache[0][scan8[i8*4]], 2, 2, 8, 0, 1);

                    fill_rectangle(&h-> mv_cache[0][scan8[i8*4]], 2, 2, 8, 0, 4);

                    fill_rectangle(&h-> mv_cache[1][scan8[i8*4]], 2, 2, 8, 0, 4);

                    continue;

                }



                ref0 = l1ref0[x8 + y8*b8_stride];

                if(ref0 >= 0)

                    ref0 = map_col_to_list0[0][ref0 + ref_offset];

                else{

                    ref0 = map_col_to_list0[1][l1ref1[x8 + y8*b8_stride] + ref_offset];

                    l1mv= l1mv1;

                }

                scale = dist_scale_factor[ref0];



                fill_rectangle(&h->ref_cache[0][scan8[i8*4]], 2, 2, 8, ref0, 1);

                if(IS_SUB_8X8(sub_mb_type)){

                    const int16_t *mv_col = l1mv[x8*3 + y8*3*b4_stride];

                    int mx = (scale * mv_col[0] + 128) >> 8;

                    int my = (scale * mv_col[1] + 128) >> 8;

                    fill_rectangle(&h->mv_cache[0][scan8[i8*4]], 2, 2, 8, pack16to32(mx,my), 4);

                    fill_rectangle(&h->mv_cache[1][scan8[i8*4]], 2, 2, 8, pack16to32(mx-mv_col[0],my-mv_col[1]), 4);

                }else

                for(i4=0; i4<4; i4++){

                    const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*b4_stride];

                    int16_t *mv_l0 = h->mv_cache[0][scan8[i8*4+i4]];

                    mv_l0[0] = (scale * mv_col[0] + 128) >> 8;

                    mv_l0[1] = (scale * mv_col[1] + 128) >> 8;

                    *(uint32_t*)h->mv_cache[1][scan8[i8*4+i4]] =

                        pack16to32(mv_l0[0]-mv_col[0],mv_l0[1]-mv_col[1]);

                }

            }

        }

    }

}
