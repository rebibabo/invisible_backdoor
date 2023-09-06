static int vc1_decode_intra_block(VC1Context *v, DCTELEM block[64], int n, int coded, int mquant, int codingset)

{

    GetBitContext *gb = &v->s.gb;

    MpegEncContext *s = &v->s;

    int dc_pred_dir = 0; /* Direction of the DC prediction used */

    int run_diff, i;

    int16_t *dc_val;

    int16_t *ac_val, *ac_val2;

    int dcdiff;

    int mb_pos = s->mb_x + s->mb_y * s->mb_stride;

    int a_avail = v->a_avail, c_avail = v->c_avail;

    int use_pred = s->ac_pred;

    int scale;

    int q1, q2 = 0;



    /* XXX: Guard against dumb values of mquant */

    mquant = (mquant < 1) ? 0 : ( (mquant>31) ? 31 : mquant );



    /* Set DC scale - y and c use the same */

    s->y_dc_scale = s->y_dc_scale_table[mquant];

    s->c_dc_scale = s->c_dc_scale_table[mquant];



    /* Get DC differential */

    if (n < 4) {

        dcdiff = get_vlc2(&s->gb, ff_msmp4_dc_luma_vlc[s->dc_table_index].table, DC_VLC_BITS, 3);

    } else {

        dcdiff = get_vlc2(&s->gb, ff_msmp4_dc_chroma_vlc[s->dc_table_index].table, DC_VLC_BITS, 3);

    }

    if (dcdiff < 0){

        av_log(s->avctx, AV_LOG_ERROR, "Illegal DC VLC\n");

        return -1;

    }

    if (dcdiff)

    {

        if (dcdiff == 119 /* ESC index value */)

        {

            /* TODO: Optimize */

            if (mquant == 1) dcdiff = get_bits(gb, 10);

            else if (mquant == 2) dcdiff = get_bits(gb, 9);

            else dcdiff = get_bits(gb, 8);

        }

        else

        {

            if (mquant == 1)

                dcdiff = (dcdiff<<2) + get_bits(gb, 2) - 3;

            else if (mquant == 2)

                dcdiff = (dcdiff<<1) + get_bits(gb, 1) - 1;

        }

        if (get_bits(gb, 1))

            dcdiff = -dcdiff;

    }



    /* Prediction */

    dcdiff += vc1_pred_dc(&v->s, v->overlap, mquant, n, a_avail, c_avail, &dc_val, &dc_pred_dir);

    *dc_val = dcdiff;



    /* Store the quantized DC coeff, used for prediction */



    if (n < 4) {

        block[0] = dcdiff * s->y_dc_scale;

    } else {

        block[0] = dcdiff * s->c_dc_scale;

    }

    /* Skip ? */

    run_diff = 0;

    i = 0;



    //AC Decoding

    i = 1;



    /* check if AC is needed at all and adjust direction if needed */

    if(!a_avail) dc_pred_dir = 1;

    if(!c_avail) dc_pred_dir = 0;

    if(!a_avail && !c_avail) use_pred = 0;

    ac_val = s->ac_val[0][0] + s->block_index[n] * 16;

    ac_val2 = ac_val;



    scale = mquant * 2 + v->halfpq;



    if(dc_pred_dir) //left

        ac_val -= 16;

    else //top

        ac_val -= 16 * s->block_wrap[n];



    q1 = s->current_picture.qscale_table[mb_pos];

    if(dc_pred_dir && c_avail) q2 = s->current_picture.qscale_table[mb_pos - 1];

    if(!dc_pred_dir && a_avail) q2 = s->current_picture.qscale_table[mb_pos - s->mb_stride];

    if(n && n<4) q2 = q1;



    if(coded) {

        int last = 0, skip, value;

        const int8_t *zz_table;

        int k;



        zz_table = vc1_simple_progressive_8x8_zz;



        while (!last) {

            vc1_decode_ac_coeff(v, &last, &skip, &value, codingset);

            i += skip;

            if(i > 63)

                break;

            block[zz_table[i++]] = value;

        }



        /* apply AC prediction if needed */

        if(use_pred) {

            /* scale predictors if needed*/

            if(q2 && q1!=q2) {

                q1 = q1 * 2 + ((q1 == v->pq) ? v->halfpq : 0) - 1;

                q2 = q2 * 2 + ((q2 == v->pq) ? v->halfpq : 0) - 1;



                if(dc_pred_dir) { //left

                    for(k = 1; k < 8; k++)

                        block[k << 3] += (ac_val[k] * q2 * vc1_dqscale[q1 - 1] + 0x20000) >> 18;

                } else { //top

                    for(k = 1; k < 8; k++)

                        block[k] += (ac_val[k + 8] * q2 * vc1_dqscale[q1 - 1] + 0x20000) >> 18;

                }

            } else {

                if(dc_pred_dir) { //left

                    for(k = 1; k < 8; k++)

                        block[k << 3] += ac_val[k];

                } else { //top

                    for(k = 1; k < 8; k++)

                        block[k] += ac_val[k + 8];

                }

            }

        }

        /* save AC coeffs for further prediction */

        for(k = 1; k < 8; k++) {

            ac_val2[k] = block[k << 3];

            ac_val2[k + 8] = block[k];

        }



        /* scale AC coeffs */

        for(k = 1; k < 64; k++)

            if(block[k]) {

                block[k] *= scale;

                if(!v->pquantizer)

                    block[k] += (block[k] < 0) ? -mquant : mquant;

            }



        if(use_pred) i = 63;

    } else { // no AC coeffs

        int k;



        memset(ac_val2, 0, 16 * 2);

        if(dc_pred_dir) {//left

            if(use_pred) {

                memcpy(ac_val2, ac_val, 8 * 2);

                if(q2 && q1!=q2) {

                    q1 = q1 * 2 + ((q1 == v->pq) ? v->halfpq : 0) - 1;

                    q2 = q2 * 2 + ((q2 == v->pq) ? v->halfpq : 0) - 1;

                    for(k = 1; k < 8; k++)

                        ac_val2[k] = (ac_val2[k] * q2 * vc1_dqscale[q1 - 1] + 0x20000) >> 18;

                }

            }

        } else {//top

            if(use_pred) {

                memcpy(ac_val2 + 8, ac_val + 8, 8 * 2);

                if(q2 && q1!=q2) {

                    q1 = q1 * 2 + ((q1 == v->pq) ? v->halfpq : 0) - 1;

                    q2 = q2 * 2 + ((q2 == v->pq) ? v->halfpq : 0) - 1;

                    for(k = 1; k < 8; k++)

                        ac_val2[k + 8] = (ac_val2[k + 8] * q2 * vc1_dqscale[q1 - 1] + 0x20000) >> 18;

                }

            }

        }



        /* apply AC prediction if needed */

        if(use_pred) {

            if(dc_pred_dir) { //left

                for(k = 1; k < 8; k++) {

                    block[k << 3] = ac_val2[k] * scale;

                    if(!v->pquantizer && block[k << 3])

                        block[k << 3] += (block[k << 3] < 0) ? -mquant : mquant;

                }

            } else { //top

                for(k = 1; k < 8; k++) {

                    block[k] = ac_val2[k + 8] * scale;

                    if(!v->pquantizer && block[k])

                        block[k] += (block[k] < 0) ? -mquant : mquant;

                }

            }

            i = 63;

        }

    }

    s->block_last_index[n] = i;



    return 0;

}
