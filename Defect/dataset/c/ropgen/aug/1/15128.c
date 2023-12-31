int nbd_client_co_pwrite_zeroes(BlockDriverState *bs, int64_t offset,

                                int count, BdrvRequestFlags flags)

{

    ssize_t ret;

    NBDClientSession *client = nbd_get_client_session(bs);

    NBDRequest request = {

        .type = NBD_CMD_WRITE_ZEROES,

        .from = offset,

        .len = count,

    };

    NBDReply reply;



    if (!(client->nbdflags & NBD_FLAG_SEND_WRITE_ZEROES)) {

        return -ENOTSUP;

    }



    if (flags & BDRV_REQ_FUA) {

        assert(client->nbdflags & NBD_FLAG_SEND_FUA);

        request.flags |= NBD_CMD_FLAG_FUA;

    }

    if (!(flags & BDRV_REQ_MAY_UNMAP)) {

        request.flags |= NBD_CMD_FLAG_NO_HOLE;

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
