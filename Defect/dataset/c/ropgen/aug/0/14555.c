int nbd_client_co_pwritev(BlockDriverState *bs, uint64_t offset,

                          uint64_t bytes, QEMUIOVector *qiov, int flags)

{

    NbdClientSession *client = nbd_get_client_session(bs);

    struct nbd_request request = {

        .type = NBD_CMD_WRITE,

        .from = offset,

        .len = bytes,

    };

    struct nbd_reply reply;

    ssize_t ret;



    if (flags & BDRV_REQ_FUA) {

        assert(client->nbdflags & NBD_FLAG_SEND_FUA);

        request.type |= NBD_CMD_FLAG_FUA;

    }



    assert(bytes <= NBD_MAX_BUFFER_SIZE);



    nbd_coroutine_start(client, &request);

    ret = nbd_co_send_request(bs, &request, qiov);

    if (ret < 0) {

        reply.error = -ret;

    } else {

        nbd_co_receive_reply(client, &request, &reply, NULL);

    }

    nbd_coroutine_end(client, &request);

    return -reply.error;

}
