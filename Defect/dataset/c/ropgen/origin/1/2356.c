static int decode_frame(WmallDecodeCtx *s)

{

    GetBitContext* gb = &s->gb;

    int more_frames = 0;

    int len = 0;

    int i;



    /** check for potential output buffer overflow */

    if (s->num_channels * s->samples_per_frame > s->samples_end - s->samples) {

        /** return an error if no frame could be decoded at all */

        av_log(s->avctx, AV_LOG_ERROR,

               "not enough space for the output samples\n");

        s->packet_loss = 1;

        return 0;

    }



    /** get frame length */

    if (s->len_prefix)

        len = get_bits(gb, s->log2_frame_size);



    /** decode tile information */

    if (decode_tilehdr(s)) {

        s->packet_loss = 1;

        return 0;

    }



    /** read drc info */

    if (s->dynamic_range_compression) {

        s->drc_gain = get_bits(gb, 8);

    }



    /** no idea what these are for, might be the number of samples

        that need to be skipped at the beginning or end of a stream */

    if (get_bits1(gb)) {

        int skip;



        /** usually true for the first frame */

        if (get_bits1(gb)) {

            skip = get_bits(gb, av_log2(s->samples_per_frame * 2));

            dprintf(s->avctx, "start skip: %i\n", skip);

        }



        /** sometimes true for the last frame */

        if (get_bits1(gb)) {

            skip = get_bits(gb, av_log2(s->samples_per_frame * 2));

            dprintf(s->avctx, "end skip: %i\n", skip);

        }



    }



    /** reset subframe states */

    s->parsed_all_subframes = 0;

    for (i = 0; i < s->num_channels; i++) {

        s->channel[i].decoded_samples = 0;

        s->channel[i].cur_subframe    = 0;

        s->channel[i].reuse_sf        = 0;

    }



    /** decode all subframes */

    while (!s->parsed_all_subframes) {

        if (decode_subframe(s) < 0) {

            s->packet_loss = 1;

            return 0;

        }

    }



    dprintf(s->avctx, "Frame done\n");



    if (s->skip_frame) {

        s->skip_frame = 0;

    } else

        s->samples += s->num_channels * s->samples_per_frame;



    if (s->len_prefix) {

        if (len != (get_bits_count(gb) - s->frame_offset) + 2) {

            /** FIXME: not sure if this is always an error */

            av_log(s->avctx, AV_LOG_ERROR,

                   "frame[%i] would have to skip %i bits\n", s->frame_num,

                   len - (get_bits_count(gb) - s->frame_offset) - 1);

            s->packet_loss = 1;

            return 0;

        }



        /** skip the rest of the frame data */

        skip_bits_long(gb, len - (get_bits_count(gb) - s->frame_offset) - 1);

    } else {

/*

        while (get_bits_count(gb) < s->num_saved_bits && get_bits1(gb) == 0) {

	    dprintf(s->avctx, "skip1\n");

        }

*/

    }



    /** decode trailer bit */

    more_frames = get_bits1(gb);

    ++s->frame_num;

    return more_frames;

}
