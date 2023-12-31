static int pva_read_packet(AVFormatContext *s, AVPacket *pkt) {

    ByteIOContext *pb = s->pb;

    PVAContext *pvactx = s->priv_data;

    int ret, syncword, streamid, reserved, flags, length, pts_flag;

    int64_t pva_pts = AV_NOPTS_VALUE;



recover:

    syncword = get_be16(pb);

    streamid = get_byte(pb);

    get_byte(pb);               /* counter not used */

    reserved = get_byte(pb);

    flags    = get_byte(pb);

    length   = get_be16(pb);



    pts_flag = flags & 0x10;



    if (syncword != PVA_MAGIC) {

        av_log(s, AV_LOG_ERROR, "invalid syncword\n");

        return AVERROR(EIO);

    }

    if (streamid != PVA_VIDEO_PAYLOAD && streamid != PVA_AUDIO_PAYLOAD) {

        av_log(s, AV_LOG_ERROR, "invalid streamid\n");

        return AVERROR(EIO);

    }

    if (reserved != 0x55) {

        av_log(s, AV_LOG_WARNING, "expected reserved byte to be 0x55\n");

    }

    if (length > PVA_MAX_PAYLOAD_LENGTH) {

        av_log(s, AV_LOG_ERROR, "invalid payload length %u\n", length);

        return AVERROR(EIO);

    }



    if (streamid == PVA_VIDEO_PAYLOAD && pts_flag) {

        pva_pts = get_be32(pb);

        length -= 4;

    } else if (streamid == PVA_AUDIO_PAYLOAD) {

        /* PVA Audio Packets either start with a signaled PES packet or

         * are a continuation of the previous PES packet. New PES packets

         * always start at the beginning of a PVA Packet, never somewhere in

         * the middle. */

        if (!pvactx->continue_pes) {

            int pes_signal, pes_header_data_length, pes_packet_length,

                pes_flags;

            unsigned char pes_header_data[256];



            pes_signal             = get_be24(pb);

            get_byte(pb);

            pes_packet_length      = get_be16(pb);

            pes_flags              = get_be16(pb);

            pes_header_data_length = get_byte(pb);



            if (pes_signal != 1) {

                av_log(s, AV_LOG_WARNING, "expected signaled PES packet, "

                                          "trying to recover\n");

                url_fskip(pb, length - 9);

                goto recover;

            }



            get_buffer(pb, pes_header_data, pes_header_data_length);

            length -= 9 + pes_header_data_length;



            pes_packet_length -= 3 + pes_header_data_length;



            pvactx->continue_pes = pes_packet_length;



            if (pes_flags & 0x80 && (pes_header_data[0] & 0xf0) == 0x20)

                pva_pts = ff_parse_pes_pts(pes_header_data);

        }



        pvactx->continue_pes -= length;



        if (pvactx->continue_pes < 0) {

            av_log(s, AV_LOG_WARNING, "audio data corruption\n");

            pvactx->continue_pes = 0;

        }

    }



    if ((ret = av_get_packet(pb, pkt, length)) <= 0)

        return AVERROR(EIO);



    pkt->stream_index = streamid - 1;

    if (pva_pts != AV_NOPTS_VALUE)

        pkt->pts = pva_pts;



    return ret;

}
