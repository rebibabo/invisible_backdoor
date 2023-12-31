static void cpu_handle_ioreq(void *opaque)

{

    XenIOState *state = opaque;

    ioreq_t *req = cpu_get_ioreq(state);



    handle_buffered_iopage(state);

    if (req) {

        handle_ioreq(state, req);



        if (req->state != STATE_IOREQ_INPROCESS) {

            fprintf(stderr, "Badness in I/O request ... not in service?!: "

                    "%x, ptr: %x, port: %"PRIx64", "

                    "data: %"PRIx64", count: %" FMT_ioreq_size

                    ", size: %" FMT_ioreq_size

                    ", type: %"FMT_ioreq_size"\n",

                    req->state, req->data_is_ptr, req->addr,

                    req->data, req->count, req->size, req->type);

            destroy_hvm_domain(false);

            return;

        }



        xen_wmb(); /* Update ioreq contents /then/ update state. */



        /*

         * We do this before we send the response so that the tools

         * have the opportunity to pick up on the reset before the

         * guest resumes and does a hlt with interrupts disabled which

         * causes Xen to powerdown the domain.

         */

        if (runstate_is_running()) {

            if (qemu_shutdown_requested_get()) {

                destroy_hvm_domain(false);

            }

            if (qemu_reset_requested_get()) {

                qemu_system_reset(VMRESET_REPORT);

                destroy_hvm_domain(true);

            }

        }



        req->state = STATE_IORESP_READY;

        xc_evtchn_notify(state->xce_handle, state->ioreq_local_port[state->send_vcpu]);

    }

}
