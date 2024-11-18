static int ogg_read_close(AVFormatContext *s)

{

    struct ogg *ogg = s->priv_data;

    int i;



    for (i = 0; i < ogg->nstreams; i++) {

        av_free(ogg->streams[i].buf);

        av_free(ogg->streams[i].private);

    }

    av_free(ogg->streams);

    return 0;

}
