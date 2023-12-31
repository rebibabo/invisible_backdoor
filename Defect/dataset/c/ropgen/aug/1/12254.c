static uint8_t get_sot(J2kDecoderContext *s)

{

    if (s->buf_end - s->buf < 4)

        return AVERROR(EINVAL);



    s->curtileno = bytestream_get_be16(&s->buf); ///< Isot

    if((unsigned)s->curtileno >= s->numXtiles * s->numYtiles){

        s->curtileno=0;

        return AVERROR(EINVAL);

    }



    s->buf += 4; ///< Psot (ignored)



    if (!bytestream_get_byte(&s->buf)){ ///< TPsot

        J2kTile *tile = s->tile + s->curtileno;



        /* copy defaults */

        memcpy(tile->codsty, s->codsty, s->ncomponents * sizeof(J2kCodingStyle));

        memcpy(tile->qntsty, s->qntsty, s->ncomponents * sizeof(J2kQuantStyle));

    }

    bytestream_get_byte(&s->buf); ///< TNsot



    return 0;

}
