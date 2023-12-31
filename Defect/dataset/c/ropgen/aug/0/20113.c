static int unpack_vlcs(Vp3DecodeContext *s, GetBitContext *gb,

                        VLC *table, int coeff_index,

                        int first_fragment, int last_fragment,

                        int eob_run)

{

    int i;

    int token;

    int zero_run = 0;

    DCTELEM coeff = 0;

    Vp3Fragment *fragment;

    uint8_t *perm= s->scantable.permutated;

    int bits_to_get;



    if ((first_fragment >= s->fragment_count) ||

        (last_fragment >= s->fragment_count)) {



        av_log(s->avctx, AV_LOG_ERROR, "  vp3:unpack_vlcs(): bad fragment number (%d -> %d ?)\n",

            first_fragment, last_fragment);

        return 0;

    }



    for (i = first_fragment; i <= last_fragment; i++) {

        int fragment_num = s->coded_fragment_list[i];



        if (s->coeff_counts[fragment_num] > coeff_index)

            continue;

        fragment = &s->all_fragments[fragment_num];



        if (!eob_run) {

            /* decode a VLC into a token */

            token = get_vlc2(gb, table->table, 5, 3);

            /* use the token to get a zero run, a coefficient, and an eob run */

            if (token <= 6) {

                eob_run = eob_run_base[token];

                if (eob_run_get_bits[token])

                    eob_run += get_bits(gb, eob_run_get_bits[token]);

                coeff = zero_run = 0;

            } else {

                bits_to_get = coeff_get_bits[token];

                if (!bits_to_get)

                    coeff = coeff_tables[token][0];

                else

                    coeff = coeff_tables[token][get_bits(gb, bits_to_get)];



                zero_run = zero_run_base[token];

                if (zero_run_get_bits[token])

                    zero_run += get_bits(gb, zero_run_get_bits[token]);

            }

        }



        if (!eob_run) {

            s->coeff_counts[fragment_num] += zero_run;

            if (s->coeff_counts[fragment_num] < 64){

                fragment->next_coeff->coeff= coeff;

                fragment->next_coeff->index= perm[s->coeff_counts[fragment_num]++]; //FIXME perm here already?

                fragment->next_coeff->next= s->next_coeff;

                s->next_coeff->next=NULL;

                fragment->next_coeff= s->next_coeff++;

            }

        } else {

            s->coeff_counts[fragment_num] |= 128;

            eob_run--;

        }

    }



    return eob_run;

}
