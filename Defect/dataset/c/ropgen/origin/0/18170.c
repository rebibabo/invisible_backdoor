static char *shorts2str(int *sp, int count, const char *sep)

{

    int i;

    char *ap, *ap0;

    if (!sep) sep = ", ";

    ap = av_malloc((5 + strlen(sep)) * count);

    if (!ap)

        return NULL;

    ap0   = ap;

    ap[0] = '\0';

    for (i = 0; i < count; i++) {

        int l = snprintf(ap, 5 + strlen(sep), "%d%s", sp[i], sep);

        ap += l;

    }

    ap0[strlen(ap0) - strlen(sep)] = '\0';

    return ap0;

}