static coroutine_fn void nbd_co_client_start(void *opaque)

{

    NBDClientNewData *data = opaque;

    NBDClient *client = data->client;

    NBDExport *exp = client->exp;



    if (exp) {

        nbd_export_get(exp);

        QTAILQ_INSERT_TAIL(&exp->clients, client, next);

    }

    qemu_co_mutex_init(&client->send_lock);



    if (nbd_negotiate(data)) {

        client_close(client);

        goto out;

    }



    nbd_client_receive_next_request(client);



out:

    g_free(data);

}
