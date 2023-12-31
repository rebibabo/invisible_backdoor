void nbd_client_new(NBDExport *exp,

                    QIOChannelSocket *sioc,

                    QCryptoTLSCreds *tlscreds,

                    const char *tlsaclname,

                    void (*close_fn)(NBDClient *))

{

    NBDClient *client;

    NBDClientNewData *data = g_new(NBDClientNewData, 1);



    client = g_malloc0(sizeof(NBDClient));

    client->refcount = 1;

    client->exp = exp;

    client->tlscreds = tlscreds;

    if (tlscreds) {

        object_ref(OBJECT(client->tlscreds));

    }

    client->tlsaclname = g_strdup(tlsaclname);

    client->sioc = sioc;

    object_ref(OBJECT(client->sioc));

    client->ioc = QIO_CHANNEL(sioc);

    object_ref(OBJECT(client->ioc));

    client->close = close_fn;



    data->client = client;

    data->co = qemu_coroutine_create(nbd_co_client_start, data);

    qemu_coroutine_enter(data->co);

}
