static void nbd_coroutine_end(BlockDriverState *bs,

                              NBDRequest *request)

{

    NBDClientSession *s = nbd_get_client_session(bs);

    int i = HANDLE_TO_INDEX(s, request->handle);



    s->recv_coroutine[i] = NULL;

    s->in_flight--;

    qemu_co_queue_next(&s->free_sema);



    /* Kick the read_reply_co to get the next reply.  */

    if (s->read_reply_co) {

        aio_co_wake(s->read_reply_co);

    }

}
