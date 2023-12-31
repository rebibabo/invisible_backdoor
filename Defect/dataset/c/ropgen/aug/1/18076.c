long vnc_client_write_sasl(VncState *vs)

{

    long ret;



    VNC_DEBUG("Write SASL: Pending output %p size %zd offset %zd "

              "Encoded: %p size %d offset %d\n",

              vs->output.buffer, vs->output.capacity, vs->output.offset,

              vs->sasl.encoded, vs->sasl.encodedLength, vs->sasl.encodedOffset);



    if (!vs->sasl.encoded) {

        int err;

        err = sasl_encode(vs->sasl.conn,

                          (char *)vs->output.buffer,

                          vs->output.offset,

                          (const char **)&vs->sasl.encoded,

                          &vs->sasl.encodedLength);

        if (err != SASL_OK)

            return vnc_client_io_error(vs, -1, NULL);



        vs->sasl.encodedRawLength = vs->output.offset;

        vs->sasl.encodedOffset = 0;




    ret = vnc_client_write_buf(vs,

                               vs->sasl.encoded + vs->sasl.encodedOffset,

                               vs->sasl.encodedLength - vs->sasl.encodedOffset);

    if (!ret)

        return 0;



    vs->sasl.encodedOffset += ret;

    if (vs->sasl.encodedOffset == vs->sasl.encodedLength) {






        vs->output.offset -= vs->sasl.encodedRawLength;

        vs->sasl.encoded = NULL;

        vs->sasl.encodedOffset = vs->sasl.encodedLength = 0;




    /* Can't merge this block with one above, because

     * someone might have written more unencrypted

     * data in vs->output while we were processing

     * SASL encoded output

     */

    if (vs->output.offset == 0) {

        if (vs->ioc_tag) {

            g_source_remove(vs->ioc_tag);


        vs->ioc_tag = qio_channel_add_watch(

            vs->ioc, G_IO_IN, vnc_client_io, vs, NULL);




    return ret;
