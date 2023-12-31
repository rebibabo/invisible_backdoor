static void nbd_recv_coroutines_enter_all(BlockDriverState *bs)

{

    NBDClientSession *s = nbd_get_client_session(bs);

    int i;



    for (i = 0; i < MAX_NBD_REQUESTS; i++) {

        if (s->recv_coroutine[i]) {

            qemu_coroutine_enter(s->recv_coroutine[i]);

        }

    }

    BDRV_POLL_WHILE(bs, s->read_reply_co);

}
