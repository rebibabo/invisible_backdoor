int avio_close(AVIOContext *s)

{

    AVIOInternal *internal;

    URLContext *h;



    if (!s)

        return 0;



    avio_flush(s);

    internal = s->opaque;

    h        = internal->h;



    av_opt_free(internal);



    av_freep(&internal->protocols);

    av_freep(&s->opaque);

    av_freep(&s->buffer);

    av_free(s);

    return ffurl_close(h);

}
