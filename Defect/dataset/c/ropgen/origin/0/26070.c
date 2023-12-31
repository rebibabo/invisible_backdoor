static void nbd_teardown_connection(BlockDriverState *bs)

{

    NBDClientSession *client = nbd_get_client_session(bs);



    if (!client->ioc) { /* Already closed */

        return;

    }



    /* finish any pending coroutines */

    qio_channel_shutdown(client->ioc,

                         QIO_CHANNEL_SHUTDOWN_BOTH,

                         NULL);

    nbd_recv_coroutines_enter_all(bs);



    nbd_client_detach_aio_context(bs);

    object_unref(OBJECT(client->sioc));

    client->sioc = NULL;

    object_unref(OBJECT(client->ioc));

    client->ioc = NULL;

}
