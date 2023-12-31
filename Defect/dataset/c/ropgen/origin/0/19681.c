static int ws_snd_decode_frame(AVCodecContext *avctx,

                void *data, int *data_size,

                AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

//    WSSNDContext *c = avctx->priv_data;



    int in_size, out_size;

    int sample = 128;

    int i;

    uint8_t *samples = data;

    uint8_t *samples_end;



    if (!buf_size)

        return 0;



    if (buf_size < 4) {

        av_log(avctx, AV_LOG_ERROR, "packet is too small\n");

        return AVERROR(EINVAL);

    }



    out_size = AV_RL16(&buf[0]);

    in_size = AV_RL16(&buf[2]);

    buf += 4;



    if (out_size > *data_size) {

        av_log(avctx, AV_LOG_ERROR, "Frame is too large to fit in buffer\n");

        return -1;

    }

    if (in_size > buf_size) {

        av_log(avctx, AV_LOG_ERROR, "Frame data is larger than input buffer\n");

        return -1;

    }

    samples_end = samples + out_size;



    if (in_size == out_size) {

        for (i = 0; i < out_size; i++)

            *samples++ = *buf++;

        *data_size = out_size;

        return buf_size;

    }



    while (samples < samples_end && buf - avpkt->data < buf_size) {

        int code, smp, size;

        uint8_t count;

        code = (*buf) >> 6;

        count = (*buf) & 0x3F;

        buf++;



        /* make sure we don't write past the output buffer */

        switch (code) {

        case 0:  smp = 4;                              break;

        case 1:  smp = 2;                              break;

        case 2:  smp = (count & 0x20) ? 1 : count + 1; break;

        default: smp = count + 1;                      break;

        }

        if (samples_end - samples < smp)

            break;



        /* make sure we don't read past the input buffer */

        size = ((code == 2 && (count & 0x20)) || code == 3) ? 0 : count + 1;

        if ((buf - avpkt->data) + size > buf_size)

            break;



        switch(code) {

        case 0: /* ADPCM 2-bit */

            for (count++; count > 0; count--) {

                code = *buf++;

                sample += ws_adpcm_2bit[code & 0x3];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                sample += ws_adpcm_2bit[(code >> 2) & 0x3];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                sample += ws_adpcm_2bit[(code >> 4) & 0x3];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                sample += ws_adpcm_2bit[(code >> 6) & 0x3];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

            }

            break;

        case 1: /* ADPCM 4-bit */

            for (count++; count > 0; count--) {

                code = *buf++;

                sample += ws_adpcm_4bit[code & 0xF];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                sample += ws_adpcm_4bit[code >> 4];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

            }

            break;

        case 2: /* no compression */

            if (count & 0x20) { /* big delta */

                int8_t t;

                t = count;

                t <<= 3;

                sample += t >> 3;

                sample = av_clip_uint8(sample);

                *samples++ = sample;

            } else { /* copy */

                for (count++; count > 0; count--) {

                    *samples++ = *buf++;

                }

                sample = buf[-1];

            }

            break;

        default: /* run */

            for(count++; count > 0; count--) {

                *samples++ = sample;

            }

        }

    }



    *data_size = samples - (uint8_t *)data;



    return buf_size;

}
