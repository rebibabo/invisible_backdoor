int nbd_client_co_flush(BlockDriverState *bs)

{

    NBDClientSession *client = nbd_get_client_session(bs);

    NBDRequest request = { .type = NBD_CMD_FLUSH };

    NBDReply reply;

    ssize_t ret;



    if (!(client->nbdflags & NBD_FLAG_SEND_FLUSH)) {

        return 0;

    }



    request.from = 0;

    request.len = 0;



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
