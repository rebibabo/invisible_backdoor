qio_channel_websock_source_prepare(GSource *source,

                                   gint *timeout)

{

    QIOChannelWebsockSource *wsource = (QIOChannelWebsockSource *)source;

    GIOCondition cond = 0;

    *timeout = -1;



    if (wsource->wioc->rawinput.offset) {

        cond |= G_IO_IN;

    }

    if (wsource->wioc->rawoutput.offset < QIO_CHANNEL_WEBSOCK_MAX_BUFFER) {

        cond |= G_IO_OUT;

    }



    return cond & wsource->condition;

}
