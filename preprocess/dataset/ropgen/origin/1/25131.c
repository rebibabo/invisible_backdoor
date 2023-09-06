static int sdp_read_header(AVFormatContext *s)

{

    RTSPState *rt = s->priv_data;

    RTSPStream *rtsp_st;

    int size, i, err;

    char *content;

    char url[1024];



    if (!ff_network_init())

        return AVERROR(EIO);



    if (s->max_delay < 0) /* Not set by the caller */

        s->max_delay = DEFAULT_REORDERING_DELAY;

    if (rt->rtsp_flags & RTSP_FLAG_CUSTOM_IO)

        rt->lower_transport = RTSP_LOWER_TRANSPORT_CUSTOM;



    /* read the whole sdp file */

    /* XXX: better loading */

    content = av_malloc(SDP_MAX_SIZE);



    size = avio_read(s->pb, content, SDP_MAX_SIZE - 1);

    if (size <= 0) {

        av_free(content);

        return AVERROR_INVALIDDATA;

    }

    content[size] ='\0';



    err = ff_sdp_parse(s, content);

    av_freep(&content);

    if (err) goto fail;



    /* open each RTP stream */

    for (i = 0; i < rt->nb_rtsp_streams; i++) {

        char namebuf[50];

        rtsp_st = rt->rtsp_streams[i];



        if (!(rt->rtsp_flags & RTSP_FLAG_CUSTOM_IO)) {

            AVDictionary *opts = map_to_opts(rt);



            getnameinfo((struct sockaddr*) &rtsp_st->sdp_ip, sizeof(rtsp_st->sdp_ip),

                        namebuf, sizeof(namebuf), NULL, 0, NI_NUMERICHOST);

            ff_url_join(url, sizeof(url), "rtp", NULL,

                        namebuf, rtsp_st->sdp_port,

                        "?localport=%d&ttl=%d&connect=%d&write_to_source=%d",

                        rtsp_st->sdp_port, rtsp_st->sdp_ttl,

                        rt->rtsp_flags & RTSP_FLAG_FILTER_SRC ? 1 : 0,

                        rt->rtsp_flags & RTSP_FLAG_RTCP_TO_SOURCE ? 1 : 0);



            append_source_addrs(url, sizeof(url), "sources",

                                rtsp_st->nb_include_source_addrs,

                                rtsp_st->include_source_addrs);

            append_source_addrs(url, sizeof(url), "block",

                                rtsp_st->nb_exclude_source_addrs,

                                rtsp_st->exclude_source_addrs);

            err = ffurl_open(&rtsp_st->rtp_handle, url, AVIO_FLAG_READ_WRITE,

                           &s->interrupt_callback, &opts);



            av_dict_free(&opts);



            if (err < 0) {

                err = AVERROR_INVALIDDATA;

                goto fail;

            }

        }

        if ((err = ff_rtsp_open_transport_ctx(s, rtsp_st)))

            goto fail;

    }

    return 0;

fail:

    ff_rtsp_close_streams(s);

    ff_network_close();

    return err;

}