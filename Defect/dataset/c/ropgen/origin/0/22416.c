int ff_aac_ac3_parse(AVCodecParserContext *s1,

                     AVCodecContext *avctx,

                     const uint8_t **poutbuf, int *poutbuf_size,

                     const uint8_t *buf, int buf_size)

{

    AACAC3ParseContext *s = s1->priv_data;

    const uint8_t *buf_ptr;

    int len, sample_rate, bit_rate, channels, samples;



    *poutbuf = NULL;

    *poutbuf_size = 0;



    buf_ptr = buf;

    while (buf_size > 0) {

        int size_needed= s->frame_size ? s->frame_size : s->header_size;

        len = s->inbuf_ptr - s->inbuf;



        if(len<size_needed){

            len = FFMIN(size_needed - len, buf_size);

            memcpy(s->inbuf_ptr, buf_ptr, len);

            buf_ptr      += len;

            s->inbuf_ptr += len;

            buf_size     -= len;

        }



        if (s->frame_size == 0) {

            if ((s->inbuf_ptr - s->inbuf) == s->header_size) {

                len = s->sync(s->inbuf, &channels, &sample_rate, &bit_rate,

                              &samples);

                if (len == 0) {

                    /* no sync found : move by one byte (inefficient, but simple!) */

                    memmove(s->inbuf, s->inbuf + 1, s->header_size - 1);

                    s->inbuf_ptr--;

                } else {

                    s->frame_size = len;

                    /* update codec info */

                    avctx->sample_rate = sample_rate;

                    avctx->channels = channels;

                    /* allow downmixing to mono or stereo for AC3 */

                    if(avctx->request_channels > 0 &&

                            avctx->request_channels < channels &&

                            avctx->request_channels <= 2 &&

                            avctx->codec_id == CODEC_ID_AC3) {

                        avctx->channels = avctx->request_channels;

                    }

                    avctx->bit_rate = bit_rate;

                    avctx->frame_size = samples;

                }

            }

        } else {

            if(s->inbuf_ptr - s->inbuf == s->frame_size){

                *poutbuf = s->inbuf;

                *poutbuf_size = s->frame_size;

                s->inbuf_ptr = s->inbuf;

                s->frame_size = 0;

                break;

            }

        }

    }

    return buf_ptr - buf;

}
