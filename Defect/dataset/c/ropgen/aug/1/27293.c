static void mpegts_write_pes(AVFormatContext *s, AVStream *st,

                             const uint8_t *payload, int payload_size,

                             int64_t pts, int64_t dts, int key)

{

    MpegTSWriteStream *ts_st = st->priv_data;

    MpegTSWrite *ts = s->priv_data;

    uint8_t buf[TS_PACKET_SIZE];

    uint8_t *q;

    int val, is_start, len, header_len, write_pcr, is_dvb_subtitle, is_dvb_teletext, flags;

    int afc_len, stuffing_len;

    int64_t pcr = -1; /* avoid warning */

    int64_t delay = av_rescale(s->max_delay, 90000, AV_TIME_BASE);

    int force_pat = st->codec->codec_type == AVMEDIA_TYPE_VIDEO && key && !ts_st->prev_payload_key;



    is_start = 1;

    while (payload_size > 0) {

        retransmit_si_info(s, force_pat);

        force_pat = 0;



        write_pcr = 0;

        if (ts_st->pid == ts_st->service->pcr_pid) {

            if (ts->mux_rate > 1 || is_start) // VBR pcr period is based on frames

                ts_st->service->pcr_packet_count++;

            if (ts_st->service->pcr_packet_count >=

                ts_st->service->pcr_packet_period) {

                ts_st->service->pcr_packet_count = 0;

                write_pcr = 1;





        if (ts->mux_rate > 1 && dts != AV_NOPTS_VALUE &&

            (dts - get_pcr(ts, s->pb)/300) > delay) {

            /* pcr insert gets priority over null packet insert */

            if (write_pcr)

                mpegts_insert_pcr_only(s, st);

            else

                mpegts_insert_null_packet(s);

            continue; /* recalculate write_pcr and possibly retransmit si_info */




        /* prepare packet header */

        q = buf;

        *q++ = 0x47;

        val = (ts_st->pid >> 8);

        if (is_start)

            val |= 0x40;

        *q++ = val;

        *q++ = ts_st->pid;

        ts_st->cc = (ts_st->cc + 1) & 0xf;

        *q++ = 0x10 | ts_st->cc; // payload indicator + CC

        if (key && is_start && pts != AV_NOPTS_VALUE) {

            // set Random Access for key frames

            if (ts_st->pid == ts_st->service->pcr_pid)

                write_pcr = 1;

            set_af_flag(buf, 0x40);

            q = get_ts_payload_start(buf);


        if (write_pcr) {

            set_af_flag(buf, 0x10);

            q = get_ts_payload_start(buf);

            // add 11, pcr references the last byte of program clock reference base

            if (ts->mux_rate > 1)

                pcr = get_pcr(ts, s->pb);

            else

                pcr = (dts - delay)*300;

            if (dts != AV_NOPTS_VALUE && dts < pcr / 300)

                av_log(s, AV_LOG_WARNING, "dts < pcr, TS is invalid\n");

            extend_af(buf, write_pcr_bits(q, pcr));

            q = get_ts_payload_start(buf);


        if (is_start) {

            int pes_extension = 0;

            int pes_header_stuffing_bytes = 0;

            /* write PES header */

            *q++ = 0x00;

            *q++ = 0x00;

            *q++ = 0x01;

            is_dvb_subtitle = 0;

            is_dvb_teletext = 0;


                if (st->codec->codec_id == AV_CODEC_ID_DIRAC) {

                    *q++ = 0xfd;

                } else

                    *q++ = 0xe0;

            } else if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO &&

                       (st->codec->codec_id == AV_CODEC_ID_MP2 ||

                        st->codec->codec_id == AV_CODEC_ID_MP3 ||

                        st->codec->codec_id == AV_CODEC_ID_AAC)) {

                *q++ = 0xc0;

            } else if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO &&

                        st->codec->codec_id == AV_CODEC_ID_AC3 &&

                        ts->m2ts_mode) {

                *q++ = 0xfd;

            } else {

                *q++ = 0xbd;

                if(st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE) {

                    if (st->codec->codec_id == AV_CODEC_ID_DVB_SUBTITLE) {

                        is_dvb_subtitle = 1;

                    } else if (st->codec->codec_id == AV_CODEC_ID_DVB_TELETEXT) {

                        is_dvb_teletext = 1;




            header_len = 0;

            flags = 0;

            if (pts != AV_NOPTS_VALUE) {

                header_len += 5;

                flags |= 0x80;


            if (dts != AV_NOPTS_VALUE && pts != AV_NOPTS_VALUE && dts != pts) {

                header_len += 5;

                flags |= 0x40;


            if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO &&

                st->codec->codec_id == AV_CODEC_ID_DIRAC) {

                /* set PES_extension_flag */

                pes_extension = 1;

                flags |= 0x01;



                /*

                * One byte for PES2 extension flag +

                * one byte for extension length +

                * one byte for extension id

                */

                header_len += 3;


            /* for Blu-ray AC3 Audio the PES Extension flag should be as follow

             * otherwise it will not play sound on blu-ray

             */

            if (ts->m2ts_mode &&

                st->codec->codec_type == AVMEDIA_TYPE_AUDIO &&

                st->codec->codec_id == AV_CODEC_ID_AC3) {

                        /* set PES_extension_flag */

                        pes_extension = 1;

                        flags |= 0x01;

                        header_len += 3;


            if (is_dvb_teletext) {

                pes_header_stuffing_bytes = 0x24 - header_len;

                header_len = 0x24;


            len = payload_size + header_len + 3;

            /* 3 extra bytes should be added to DVB subtitle payload: 0x20 0x00 at the beginning and trailing 0xff */

            if (is_dvb_subtitle) {

                len += 3;

                payload_size++;


            if (len > 0xffff)





            *q++ = len >> 8;

            *q++ = len;

            val = 0x80;

            /* data alignment indicator is required for subtitle and data streams */

            if (st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE || st->codec->codec_type == AVMEDIA_TYPE_DATA)

                val |= 0x04;

            *q++ = val;

            *q++ = flags;

            *q++ = header_len;

            if (pts != AV_NOPTS_VALUE) {

                write_pts(q, flags >> 6, pts);

                q += 5;


            if (dts != AV_NOPTS_VALUE && pts != AV_NOPTS_VALUE && dts != pts) {

                write_pts(q, 1, dts);

                q += 5;


            if (pes_extension && st->codec->codec_id == AV_CODEC_ID_DIRAC) {

                flags = 0x01;  /* set PES_extension_flag_2 */

                *q++ = flags;

                *q++ = 0x80 | 0x01;  /* marker bit + extension length */

                /*

                * Set the stream id extension flag bit to 0 and

                * write the extended stream id

                */

                *q++ = 0x00 | 0x60;


            /* For Blu-ray AC3 Audio Setting extended flags */

          if (ts->m2ts_mode &&

              pes_extension &&

              st->codec->codec_id == AV_CODEC_ID_AC3) {

                      flags = 0x01; /* set PES_extension_flag_2 */

                      *q++ = flags;

                      *q++ = 0x80 | 0x01; /* marker bit + extension length */

                      *q++ = 0x00 | 0x71; /* for AC3 Audio (specifically on blue-rays) */






            if (is_dvb_subtitle) {

                /* First two fields of DVB subtitles PES data:

                 * data_identifier: for DVB subtitle streams shall be coded with the value 0x20

                 * subtitle_stream_id: for DVB subtitle stream shall be identified by the value 0x00 */

                *q++ = 0x20;

                *q++ = 0x00;


            if (is_dvb_teletext) {

                memset(q, 0xff, pes_header_stuffing_bytes);

                q += pes_header_stuffing_bytes;


            is_start = 0;


        /* header size */

        header_len = q - buf;

        /* data len */

        len = TS_PACKET_SIZE - header_len;

        if (len > payload_size)

            len = payload_size;

        stuffing_len = TS_PACKET_SIZE - header_len - len;

        if (stuffing_len > 0) {

            /* add stuffing with AFC */

            if (buf[3] & 0x20) {

                /* stuffing already present: increase its size */

                afc_len = buf[4] + 1;

                memmove(buf + 4 + afc_len + stuffing_len,

                        buf + 4 + afc_len,

                        header_len - (4 + afc_len));

                buf[4] += stuffing_len;

                memset(buf + 4 + afc_len, 0xff, stuffing_len);

            } else {

                /* add stuffing */

                memmove(buf + 4 + stuffing_len, buf + 4, header_len - 4);

                buf[3] |= 0x20;

                buf[4] = stuffing_len - 1;

                if (stuffing_len >= 2) {

                    buf[5] = 0x00;

                    memset(buf + 6, 0xff, stuffing_len - 2);






        if (is_dvb_subtitle && payload_size == len) {

            memcpy(buf + TS_PACKET_SIZE - len, payload, len - 1);

            buf[TS_PACKET_SIZE - 1] = 0xff; /* end_of_PES_data_field_marker: an 8-bit field with fixed contents 0xff for DVB subtitle */

        } else {

            memcpy(buf + TS_PACKET_SIZE - len, payload, len);




        payload += len;

        payload_size -= len;

        mpegts_prefix_m2ts_header(s);

        avio_write(s->pb, buf, TS_PACKET_SIZE);


    avio_flush(s->pb);

    ts_st->prev_payload_key = key;
