static int mov_read_stts(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = st->priv_data;

    unsigned int i, entries;

    int64_t duration=0;

    int64_t total_sample_count=0;



    get_byte(pb); /* version */

    get_be24(pb); /* flags */

    entries = get_be32(pb);



    dprintf(c->fc, "track[%i].stts.entries = %i\n", c->fc->nb_streams-1, entries);



    if(entries >= UINT_MAX / sizeof(*sc->stts_data))

        return -1;

    sc->stts_data = av_malloc(entries * sizeof(*sc->stts_data));

    if (!sc->stts_data)

        return AVERROR(ENOMEM);

    sc->stts_count = entries;



    for(i=0; i<entries; i++) {

        int sample_duration;

        int sample_count;



        sample_count=get_be32(pb);

        sample_duration = get_be32(pb);

        sc->stts_data[i].count= sample_count;

        sc->stts_data[i].duration= sample_duration;



        dprintf(c->fc, "sample_count=%d, sample_duration=%d\n",sample_count,sample_duration);



        duration+=(int64_t)sample_duration*sample_count;

        total_sample_count+=sample_count;

    }



    st->nb_frames= total_sample_count;

    if(duration)

        st->duration= duration;

    return 0;

}
