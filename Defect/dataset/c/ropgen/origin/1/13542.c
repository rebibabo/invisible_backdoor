static void matroska_merge_packets(AVPacket *out, AVPacket *in)

{

    out->data = av_realloc(out->data, out->size+in->size);

    memcpy(out->data+out->size, in->data, in->size);

    out->size += in->size;

    av_destruct_packet(in);

    av_free(in);

}
