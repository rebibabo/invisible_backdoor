static gboolean nbd_accept(QIOChannel *ioc, GIOCondition condition,

                           gpointer opaque)

{

    QIOChannelSocket *cioc;



    if (!nbd_server) {

        return FALSE;

    }



    cioc = qio_channel_socket_accept(QIO_CHANNEL_SOCKET(ioc),

                                     NULL);

    if (!cioc) {

        return TRUE;

    }



    qio_channel_set_name(QIO_CHANNEL(cioc), "nbd-server");

    nbd_client_new(NULL, cioc,

                   nbd_server->tlscreds, NULL,

                   nbd_client_put);

    object_unref(OBJECT(cioc));

    return TRUE;

}
