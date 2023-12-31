static uint_fast8_t vorbis_floor0_decode(vorbis_context *vc,

                                         vorbis_floor_data *vfu, float *vec)

{

    vorbis_floor0 *vf = &vfu->t0;

    float *lsp = vf->lsp;

    uint_fast32_t amplitude;

    uint_fast32_t book_idx;

    uint_fast8_t blockflag = vc->modes[vc->mode_number].blockflag;



    amplitude = get_bits(&vc->gb, vf->amplitude_bits);

    if (amplitude > 0) {

        float last = 0;

        uint_fast16_t lsp_len = 0;

        uint_fast16_t idx;

        vorbis_codebook codebook;



        book_idx = get_bits(&vc->gb, ilog(vf->num_books));

        if (book_idx >= vf->num_books) {

            av_log(vc->avccontext, AV_LOG_ERROR,

                    "floor0 dec: booknumber too high!\n");

            book_idx =  0;

            //FIXME: look above

        }

        AV_DEBUG("floor0 dec: booknumber: %u\n", book_idx);

        codebook = vc->codebooks[vf->book_list[book_idx]];



        while (lsp_len<vf->order) {

            int vec_off;



            AV_DEBUG("floor0 dec: book dimension: %d\n", codebook.dimensions);

            AV_DEBUG("floor0 dec: maximum depth: %d\n", codebook.maxdepth);

            /* read temp vector */

            vec_off = get_vlc2(&vc->gb, codebook.vlc.table,

                               codebook.nb_bits, codebook.maxdepth)

                      * codebook.dimensions;

            AV_DEBUG("floor0 dec: vector offset: %d\n", vec_off);

            /* copy each vector component and add last to it */

            for (idx = 0; idx < codebook.dimensions; ++idx)

                lsp[lsp_len+idx] = codebook.codevectors[vec_off+idx] + last;

            last = lsp[lsp_len+idx-1]; /* set last to last vector component */



            lsp_len += codebook.dimensions;

        }

#ifdef V_DEBUG

        /* DEBUG: output lsp coeffs */

        {

            int idx;

            for (idx = 0; idx < lsp_len; ++idx)

                AV_DEBUG("floor0 dec: coeff at %d is %f\n", idx, lsp[idx]);

        }

#endif



        /* synthesize floor output vector */

        {

            int i;

            int order = vf->order;

            float wstep = M_PI / vf->bark_map_size;



            for (i = 0; i < order; i++)

                lsp[i] = 2.0f * cos(lsp[i]);



            AV_DEBUG("floor0 synth: map_size = %d; m = %d; wstep = %f\n",

                     vf->map_size, order, wstep);



            i = 0;

            while (i < vf->map_size[blockflag]) {

                int j, iter_cond = vf->map[blockflag][i];

                float p = 0.5f;

                float q = 0.5f;

                float two_cos_w = 2.0f * cos(wstep * iter_cond); // needed all times



                /* similar part for the q and p products */

                for (j = 0; j + 1 < order; j += 2) {

                    q *= lsp[j]     - two_cos_w;

                    p *= lsp[j + 1] - two_cos_w;

                }

                if (j == order) { // even order

                    p *= p * (2.0f - two_cos_w);

                    q *= q * (2.0f + two_cos_w);

                } else { // odd order

                    q *= two_cos_w-lsp[j]; // one more time for q



                    /* final step and square */

                    p *= p * (4.f - two_cos_w * two_cos_w);

                    q *= q;

                }



                /* calculate linear floor value */

                {

                    q = exp((((amplitude*vf->amplitude_offset) /

                              (((1 << vf->amplitude_bits) - 1) * sqrt(p + q)))

                             - vf->amplitude_offset) * .11512925f);

                }



                /* fill vector */

                do {

                    vec[i] = q; ++i;

                } while (vf->map[blockflag][i] == iter_cond);

            }

        }

    } else {

        /* this channel is unused */

        return 1;

    }



    AV_DEBUG(" Floor0 decoded\n");



    return 0;

}
