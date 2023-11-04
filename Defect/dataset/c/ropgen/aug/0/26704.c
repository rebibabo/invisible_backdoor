static int add_doubles_metadata(int count,

                                const char *name, const char *sep,

                                TiffContext *s)

{

    char *ap;

    int i;

    double *dp;



    if (bytestream2_get_bytes_left(&s->gb) < count * sizeof(int64_t))

        return -1;



    dp = av_malloc(count * sizeof(double));

    if (!dp)

        return AVERROR(ENOMEM);



    for (i = 0; i < count; i++)

        dp[i] = tget_double(&s->gb, s->le);

    ap = doubles2str(dp, count, sep);

    av_freep(&dp);

    if (!ap)

        return AVERROR(ENOMEM);

    av_dict_set(&s->picture.metadata, name, ap, AV_DICT_DONT_STRDUP_VAL);

    return 0;

}