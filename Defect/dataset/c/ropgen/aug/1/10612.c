int nbd_client_co_pdiscard(BlockDriverState *bs, int64_t offset, int count)

{

    NBDClientSession *client = nbd_get_client_session(bs);

    NBDRequest request = {

        .type = NBD_CMD_TRIM,

        .from = offset,

        .len = count,

    };

    NBDReply reply;

    ssize_t ret;



    if (!(client->nbdflags & NBD_FLAG_SEND_TRIM)) {

        return 0;

    }



    nbd_coroutine_start(client, &request);

    ret = nbd_co_send_request(bs, &request, NULL);

    if (ret < 0) {

        reply.error = -ret;

    } else {

        nbd_co_receive_reply(client, &request, &reply, NULL);

    }

    nbd_coroutine_end(bs, &request);

    return -reply.error;



}
