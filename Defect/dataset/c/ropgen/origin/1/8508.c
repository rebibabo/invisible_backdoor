static int rtp_read_header(AVFormatContext *s)

{

    uint8_t recvbuf[RTP_MAX_PACKET_LENGTH];

    char host[500], sdp[500];

    int ret, port;

    URLContext* in = NULL;

    int payload_type;

    AVCodecContext codec = { 0 };

    struct sockaddr_storage addr;

    AVIOContext pb;

    socklen_t addrlen = sizeof(addr);

    RTSPState *rt = s->priv_data;



    if (!ff_network_init())

        return AVERROR(EIO);



    if (!rt->protocols) {

        rt->protocols = ffurl_get_protocols(NULL, NULL);

        if (!rt->protocols)

            return AVERROR(ENOMEM);

    }



    ret = ffurl_open(&in, s->filename, AVIO_FLAG_READ,

                     &s->interrupt_callback, NULL, rt->protocols);

    if (ret)

        goto fail;



    while (1) {

        ret = ffurl_read(in, recvbuf, sizeof(recvbuf));

        if (ret == AVERROR(EAGAIN))

            continue;

        if (ret < 0)

            goto fail;

        if (ret < 12) {

            av_log(s, AV_LOG_WARNING, "Received too short packet\n");

            continue;

        }



        if ((recvbuf[0] & 0xc0) != 0x80) {

            av_log(s, AV_LOG_WARNING, "Unsupported RTP version packet "

                                      "received\n");

            continue;

        }



        if (RTP_PT_IS_RTCP(recvbuf[1]))

            continue;



        payload_type = recvbuf[1] & 0x7f;

        break;

    }

    getsockname(ffurl_get_file_handle(in), (struct sockaddr*) &addr, &addrlen);

    ffurl_close(in);

    in = NULL;



    if (ff_rtp_get_codec_info(&codec, payload_type)) {

        av_log(s, AV_LOG_ERROR, "Unable to receive RTP payload type %d "

                                "without an SDP file describing it\n",

                                 payload_type);

        goto fail;

    }

    if (codec.codec_type != AVMEDIA_TYPE_DATA) {

        av_log(s, AV_LOG_WARNING, "Guessing on RTP content - if not received "

                                  "properly you need an SDP file "

                                  "describing it\n");

    }



    av_url_split(NULL, 0, NULL, 0, host, sizeof(host), &port,

                 NULL, 0, s->filename);



    snprintf(sdp, sizeof(sdp),

             "v=0\r\nc=IN IP%d %s\r\nm=%s %d RTP/AVP %d\r\n",

             addr.ss_family == AF_INET ? 4 : 6, host,

             codec.codec_type == AVMEDIA_TYPE_DATA  ? "application" :

             codec.codec_type == AVMEDIA_TYPE_VIDEO ? "video" : "audio",

             port, payload_type);

    av_log(s, AV_LOG_VERBOSE, "SDP:\n%s\n", sdp);



    ffio_init_context(&pb, sdp, strlen(sdp), 0, NULL, NULL, NULL, NULL);

    s->pb = &pb;



    /* sdp_read_header initializes this again */

    ff_network_close();



    rt->media_type_mask = (1 << (AVMEDIA_TYPE_DATA+1)) - 1;



    ret = sdp_read_header(s);

    s->pb = NULL;

    return ret;



fail:

    if (in)

        ffurl_close(in);

    ff_network_close();

    return ret;

}
