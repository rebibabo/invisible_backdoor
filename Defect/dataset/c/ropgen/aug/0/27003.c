static ssize_t nbd_co_send_reply(NBDRequestData *req, NBDReply *reply,

                                 int len)

{

    NBDClient *client = req->client;

    ssize_t rc, ret;



    g_assert(qemu_in_coroutine());

    qemu_co_mutex_lock(&client->send_lock);

    client->send_coroutine = qemu_coroutine_self();



    if (!len) {

        rc = nbd_send_reply(client->ioc, reply);

    } else {

        qio_channel_set_cork(client->ioc, true);

        rc = nbd_send_reply(client->ioc, reply);

        if (rc >= 0) {

            ret = write_sync(client->ioc, req->data, len, NULL);

            if (ret < 0) {

                rc = -EIO;

            }

        }

        qio_channel_set_cork(client->ioc, false);

    }



    client->send_coroutine = NULL;

    qemu_co_mutex_unlock(&client->send_lock);

    return rc;

}
