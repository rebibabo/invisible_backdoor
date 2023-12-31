static int handle_buffered_iopage(XenIOState *state)

{

    buffered_iopage_t *buf_page = state->buffered_io_page;

    buf_ioreq_t *buf_req = NULL;

    ioreq_t req;

    int qw;



    if (!buf_page) {

        return 0;

    }



    memset(&req, 0x00, sizeof(req));



    for (;;) {

        uint32_t rdptr = buf_page->read_pointer, wrptr;



        xen_rmb();

        wrptr = buf_page->write_pointer;

        xen_rmb();

        if (rdptr != buf_page->read_pointer) {

            continue;

        }

        if (rdptr == wrptr) {

            break;

        }

        buf_req = &buf_page->buf_ioreq[rdptr % IOREQ_BUFFER_SLOT_NUM];

        req.size = 1UL << buf_req->size;

        req.count = 1;

        req.addr = buf_req->addr;

        req.data = buf_req->data;

        req.state = STATE_IOREQ_READY;

        req.dir = buf_req->dir;

        req.df = 1;

        req.type = buf_req->type;

        req.data_is_ptr = 0;

        xen_rmb();

        qw = (req.size == 8);

        if (qw) {

            if (rdptr + 1 == wrptr) {

                hw_error("Incomplete quad word buffered ioreq");

            }

            buf_req = &buf_page->buf_ioreq[(rdptr + 1) %

                                           IOREQ_BUFFER_SLOT_NUM];

            req.data |= ((uint64_t)buf_req->data) << 32;

            xen_rmb();

        }



        handle_ioreq(state, &req);



        atomic_add(&buf_page->read_pointer, qw + 1);

    }



    return req.count;

}
