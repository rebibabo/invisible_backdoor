static int sap_read_header(AVFormatContext *s)

{

    struct SAPState *sap = s->priv_data;

    char host[1024], path[1024], url[1024];

    uint8_t recvbuf[RTP_MAX_PACKET_LENGTH];

    int port;

    int ret, i;

    AVInputFormat* infmt;



    if (!ff_network_init())

        return AVERROR(EIO);



    av_url_split(NULL, 0, NULL, 0, host, sizeof(host), &port,

                 path, sizeof(path), s->filename);

    if (port < 0)

        port = 9875;



    if (!host[0]) {

        /* Listen for announcements on sap.mcast.net if no host was specified */

        av_strlcpy(host, "224.2.127.254", sizeof(host));

    }



    ff_url_join(url, sizeof(url), "udp", NULL, host, port, "?localport=%d",

                port);

    ret = ffurl_open(&sap->ann_fd, url, AVIO_FLAG_READ,

                     &s->interrupt_callback, NULL);

    if (ret)

        goto fail;



    while (1) {

        int addr_type, auth_len;

        int pos;



        ret = ffurl_read(sap->ann_fd, recvbuf, sizeof(recvbuf) - 1);

        if (ret == AVERROR(EAGAIN))

            continue;

        if (ret < 0)

            goto fail;

        recvbuf[ret] = '\0'; /* Null terminate for easier parsing */

        if (ret < 8) {

            av_log(s, AV_LOG_WARNING, "Received too short packet\n");

            continue;

        }



        if ((recvbuf[0] & 0xe0) != 0x20) {

            av_log(s, AV_LOG_WARNING, "Unsupported SAP version packet "

                                      "received\n");

            continue;

        }



        if (recvbuf[0] & 0x04) {

            av_log(s, AV_LOG_WARNING, "Received stream deletion "

                                      "announcement\n");

            continue;

        }

        addr_type = recvbuf[0] & 0x10;

        auth_len  = recvbuf[1];

        sap->hash = AV_RB16(&recvbuf[2]);

        pos = 4;

        if (addr_type)

            pos += 16; /* IPv6 */

        else

            pos += 4; /* IPv4 */

        pos += auth_len * 4;

        if (pos + 4 >= ret) {

            av_log(s, AV_LOG_WARNING, "Received too short packet\n");

            continue;

        }

#define MIME "application/sdp"

        if (strcmp(&recvbuf[pos], MIME) == 0) {

            pos += strlen(MIME) + 1;

        } else if (strncmp(&recvbuf[pos], "v=0\r\n", 5) == 0) {

            // Direct SDP without a mime type

        } else {

            av_log(s, AV_LOG_WARNING, "Unsupported mime type %s\n",

                                      &recvbuf[pos]);

            continue;

        }



        sap->sdp = av_strdup(&recvbuf[pos]);

        break;

    }



    av_log(s, AV_LOG_VERBOSE, "SDP:\n%s\n", sap->sdp);

    ffio_init_context(&sap->sdp_pb, sap->sdp, strlen(sap->sdp), 0, NULL, NULL,

                  NULL, NULL);



    infmt = av_find_input_format("sdp");

    if (!infmt)

        goto fail;

    sap->sdp_ctx = avformat_alloc_context();

    if (!sap->sdp_ctx) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }

    sap->sdp_ctx->max_delay = s->max_delay;

    sap->sdp_ctx->pb        = &sap->sdp_pb;

    sap->sdp_ctx->interrupt_callback = s->interrupt_callback;



    av_assert0(!sap->sdp_ctx->codec_whitelist && !sap->sdp_ctx->format_whitelist);

    sap->sdp_ctx-> codec_whitelist = av_strdup(s->codec_whitelist);

    sap->sdp_ctx->format_whitelist = av_strdup(s->format_whitelist);



    ret = avformat_open_input(&sap->sdp_ctx, "temp.sdp", infmt, NULL);

    if (ret < 0)

        goto fail;

    if (sap->sdp_ctx->ctx_flags & AVFMTCTX_NOHEADER)

        s->ctx_flags |= AVFMTCTX_NOHEADER;

    for (i = 0; i < sap->sdp_ctx->nb_streams; i++) {

        AVStream *st = avformat_new_stream(s, NULL);

        if (!st) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        st->id = i;

        avcodec_copy_context(st->codec, sap->sdp_ctx->streams[i]->codec);

        st->time_base = sap->sdp_ctx->streams[i]->time_base;

    }



    return 0;



fail:

    sap_read_close(s);

    return ret;

}
