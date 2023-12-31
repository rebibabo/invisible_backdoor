static int asfrtp_parse_packet(AVFormatContext *s, PayloadContext *asf,

                               AVStream *st, AVPacket *pkt,

                               uint32_t *timestamp,

                               const uint8_t *buf, int len, int flags)

{

    AVIOContext *pb = &asf->pb;

    int res, mflags, len_off;

    RTSPState *rt = s->priv_data;



    if (!rt->asf_ctx)




    if (len > 0) {

        int off, out_len = 0;



        if (len < 4)




        av_freep(&asf->buf);



        ffio_init_context(pb, buf, len, 0, NULL, NULL, NULL, NULL);



        while (avio_tell(pb) + 4 < len) {

            int start_off = avio_tell(pb);



            mflags = avio_r8(pb);

            if (mflags & 0x80)

                flags |= RTP_FLAG_KEY;

            len_off = avio_rb24(pb);

            if (mflags & 0x20)   /**< relative timestamp */

                avio_skip(pb, 4);

            if (mflags & 0x10)   /**< has duration */

                avio_skip(pb, 4);

            if (mflags & 0x8)    /**< has location ID */

                avio_skip(pb, 4);

            off = avio_tell(pb);



            if (!(mflags & 0x40)) {

                /**

                 * If 0x40 is not set, the len_off field specifies an offset

                 * of this packet's payload data in the complete (reassembled)

                 * ASF packet. This is used to spread one ASF packet over

                 * multiple RTP packets.

                 */

                if (asf->pktbuf && len_off != avio_tell(asf->pktbuf)) {

                    uint8_t *p;

                    avio_close_dyn_buf(asf->pktbuf, &p);

                    asf->pktbuf = NULL;

                    av_free(p);

                }

                if (!len_off && !asf->pktbuf &&

                    (res = avio_open_dyn_buf(&asf->pktbuf)) < 0)

                    return res;

                if (!asf->pktbuf)

                    return AVERROR(EIO);



                avio_write(asf->pktbuf, buf + off, len - off);

                avio_skip(pb, len - off);

                if (!(flags & RTP_FLAG_MARKER))


                out_len     = avio_close_dyn_buf(asf->pktbuf, &asf->buf);

                asf->pktbuf = NULL;

            } else {

                /**

                 * If 0x40 is set, the len_off field specifies the length of

                 * the next ASF packet that can be read from this payload

                 * data alone. This is commonly the same as the payload size,

                 * but could be less in case of packet splitting (i.e.

                 * multiple ASF packets in one RTP packet).

                 */



                int cur_len = start_off + len_off - off;

                int prev_len = out_len;

                out_len += cur_len;

                asf->buf = av_realloc(asf->buf, out_len);



                memcpy(asf->buf + prev_len, buf + off,

                       FFMIN(cur_len, len - off));

                avio_skip(pb, cur_len);

            }

        }



        init_packetizer(pb, asf->buf, out_len);

        pb->pos += rt->asf_pb_pos;

        pb->eof_reached = 0;

        rt->asf_ctx->pb = pb;

    }



    for (;;) {

        int i;



        res = av_read_packet(rt->asf_ctx, pkt);

        rt->asf_pb_pos = avio_tell(pb);

        if (res != 0)

            break;

        for (i = 0; i < s->nb_streams; i++) {

            if (s->streams[i]->id == rt->asf_ctx->streams[pkt->stream_index]->id) {

                pkt->stream_index = i;

                return 1; // FIXME: return 0 if last packet

            }

        }

        av_free_packet(pkt);

    }



    return res == 1 ? -1 : res;

}