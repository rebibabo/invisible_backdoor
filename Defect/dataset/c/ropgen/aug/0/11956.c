pvscsi_convert_sglist(PVSCSIRequest *r)

{

    int chunk_size;

    uint64_t data_length = r->req.dataLen;

    PVSCSISGState sg = r->sg;

    while (data_length) {

        while (!sg.resid) {

            pvscsi_get_next_sg_elem(&sg);

            trace_pvscsi_convert_sglist(r->req.context, r->sg.dataAddr,

                                        r->sg.resid);

        }

        assert(data_length > 0);

        chunk_size = MIN((unsigned) data_length, sg.resid);

        if (chunk_size) {

            qemu_sglist_add(&r->sgl, sg.dataAddr, chunk_size);

        }



        sg.dataAddr += chunk_size;

        data_length -= chunk_size;

        sg.resid -= chunk_size;

    }

}
