static int flic_decode_frame_15_16BPP(AVCodecContext *avctx,

                                      void *data, int *data_size,

                                      uint8_t *buf, int buf_size)

{

    /* Note, the only difference between the 15Bpp and 16Bpp */

    /* Format is the pixel format, the packets are processed the same. */

    FlicDecodeContext *s = (FlicDecodeContext *)avctx->priv_data;



    int stream_ptr = 0;

    int pixel_ptr;

    unsigned char palette_idx1;



    unsigned int frame_size;

    int num_chunks;



    unsigned int chunk_size;

    int chunk_type;



    int i, j;



    int lines;

    int compressed_lines;

    signed short line_packets;

    int y_ptr;

    int byte_run;

    int pixel_skip;

    int pixel_countdown;

    unsigned char *pixels;

    int pixel;

    int pixel_limit;



    s->frame.reference = 1;

    s->frame.buffer_hints = FF_BUFFER_HINTS_VALID | FF_BUFFER_HINTS_PRESERVE | FF_BUFFER_HINTS_REUSABLE;

    if (avctx->reget_buffer(avctx, &s->frame) < 0) {

        av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

        return -1;

    }



    pixels = s->frame.data[0];

    pixel_limit = s->avctx->height * s->frame.linesize[0];



    frame_size = LE_32(&buf[stream_ptr]);

    stream_ptr += 6;  /* skip the magic number */

    num_chunks = LE_16(&buf[stream_ptr]);

    stream_ptr += 10;  /* skip padding */



    frame_size -= 16;



    /* iterate through the chunks */

    while ((frame_size > 0) && (num_chunks > 0)) {

        chunk_size = LE_32(&buf[stream_ptr]);

        stream_ptr += 4;

        chunk_type = LE_16(&buf[stream_ptr]);

        stream_ptr += 2;



        switch (chunk_type) {

        case FLI_256_COLOR:

        case FLI_COLOR:

            /* For some reason, it seems that non-paletised flics do include one of these */

            /* chunks in their first frame.  Why i do not know, it seems rather extraneous */

/*            av_log(avctx, AV_LOG_ERROR, "Unexpected Palette chunk %d in non-paletised FLC\n",chunk_type);*/

            stream_ptr = stream_ptr + chunk_size - 6;

            break;



        case FLI_DELTA:

        case FLI_DTA_LC:

            y_ptr = 0;

            compressed_lines = LE_16(&buf[stream_ptr]);

            stream_ptr += 2;

            while (compressed_lines > 0) {

                line_packets = LE_16(&buf[stream_ptr]);

                stream_ptr += 2;

                if (line_packets < 0) {

                    line_packets = -line_packets;

                    y_ptr += line_packets * s->frame.linesize[0];

                } else {

                    compressed_lines--;

                    pixel_ptr = y_ptr;

                    pixel_countdown = s->avctx->width;

                    for (i = 0; i < line_packets; i++) {

                        /* account for the skip bytes */

                        pixel_skip = buf[stream_ptr++];

                        pixel_ptr += (pixel_skip*2); /* Pixel is 2 bytes wide */

                        pixel_countdown -= pixel_skip;

                        byte_run = (signed char)(buf[stream_ptr++]);

                        if (byte_run < 0) {

                            byte_run = -byte_run;

                            pixel    = LE_16(&buf[stream_ptr]);

                            stream_ptr += 2;

                            CHECK_PIXEL_PTR(byte_run);

                            for (j = 0; j < byte_run; j++, pixel_countdown -= 2) {

                                *((signed short*)(&pixels[pixel_ptr])) = pixel;

                                pixel_ptr += 2;

                            }

                        } else {

                            CHECK_PIXEL_PTR(byte_run);

                            for (j = 0; j < byte_run; j++, pixel_countdown--) {

                                *((signed short*)(&pixels[pixel_ptr])) = LE_16(&buf[stream_ptr]);

                                stream_ptr += 2;

                                pixel_ptr += 2;

                            }

                        }

                    }



                    y_ptr += s->frame.linesize[0];

                }

            }

            break;



        case FLI_LC:

            av_log(avctx, AV_LOG_ERROR, "Unexpected FLI_LC chunk in non-paletised FLC\n");

            stream_ptr = stream_ptr + chunk_size - 6;

            break;



        case FLI_BLACK:

            /* set the whole frame to 0x0000 which is balck in both 15Bpp and 16Bpp modes. */

            memset(pixels, 0x0000,

                   s->frame.linesize[0] * s->avctx->height * 2);

            break;



        case FLI_BRUN:

            y_ptr = 0;

            for (lines = 0; lines < s->avctx->height; lines++) {

                pixel_ptr = y_ptr;

                /* disregard the line packets; instead, iterate through all

                 * pixels on a row */

                stream_ptr++;

                pixel_countdown = (s->avctx->width * 2);



                while (pixel_countdown > 0) {

                    byte_run = (signed char)(buf[stream_ptr++]);

                    if (byte_run > 0) {

                        palette_idx1 = buf[stream_ptr++];

                        CHECK_PIXEL_PTR(byte_run);

                        for (j = 0; j < byte_run; j++) {

                            pixels[pixel_ptr++] = palette_idx1;

                            pixel_countdown--;

                            if (pixel_countdown < 0)

                                av_log(avctx, AV_LOG_ERROR, "pixel_countdown < 0 (%d)\n",

                                       pixel_countdown);

                        }

                    } else {  /* copy bytes if byte_run < 0 */

                        byte_run = -byte_run;

                        CHECK_PIXEL_PTR(byte_run);

                        for (j = 0; j < byte_run; j++) {

                            palette_idx1 = buf[stream_ptr++];

                            pixels[pixel_ptr++] = palette_idx1;

                            pixel_countdown--;

                            if (pixel_countdown < 0)

                                av_log(avctx, AV_LOG_ERROR, "pixel_countdown < 0 (%d)\n",

                                       pixel_countdown);

                        }

                    }

                }



                /* Now FLX is strange, in that it is "byte" as opposed to "pixel" run length compressed.

                 * This doesnt give us any good oportunity to perform word endian conversion

                 * during decompression. So if its requried (ie, this isnt a LE target, we do

                 * a second pass over the line here, swapping the bytes.

                 */

                pixel = 0xFF00;

                if (0xFF00 != LE_16(&pixel)) /* Check if its not an LE Target */

                {

                  pixel_ptr = y_ptr;

                  pixel_countdown = s->avctx->width;

                  while (pixel_countdown > 0) {

                    *((signed short*)(&pixels[pixel_ptr])) = LE_16(&buf[pixel_ptr]);

                    pixel_ptr += 2;

                  }

                }

                y_ptr += s->frame.linesize[0];

            }

            break;



        case FLI_DTA_BRUN:

            y_ptr = 0;

            for (lines = 0; lines < s->avctx->height; lines++) {

                pixel_ptr = y_ptr;

                /* disregard the line packets; instead, iterate through all

                 * pixels on a row */

                stream_ptr++;

                pixel_countdown = s->avctx->width; /* Width is in pixels, not bytes */



                while (pixel_countdown > 0) {

                    byte_run = (signed char)(buf[stream_ptr++]);

                    if (byte_run > 0) {

                        pixel    = LE_16(&buf[stream_ptr]);

                        stream_ptr += 2;

                        CHECK_PIXEL_PTR(byte_run);

                        for (j = 0; j < byte_run; j++) {

                            *((signed short*)(&pixels[pixel_ptr])) = pixel;

                            pixel_ptr += 2;

                            pixel_countdown--;

                            if (pixel_countdown < 0)

                                av_log(avctx, AV_LOG_ERROR, "pixel_countdown < 0 (%d)\n",

                                       pixel_countdown);

                        }

                    } else {  /* copy pixels if byte_run < 0 */

                        byte_run = -byte_run;

                        CHECK_PIXEL_PTR(byte_run);

                        for (j = 0; j < byte_run; j++) {

                            *((signed short*)(&pixels[pixel_ptr])) = LE_16(&buf[stream_ptr]);

                            stream_ptr += 2;

                            pixel_ptr  += 2;

                            pixel_countdown--;

                            if (pixel_countdown < 0)

                                av_log(avctx, AV_LOG_ERROR, "pixel_countdown < 0 (%d)\n",

                                       pixel_countdown);

                        }

                    }

                }



                y_ptr += s->frame.linesize[0];

            }

            break;



        case FLI_COPY:

        case FLI_DTA_COPY:

            /* copy the chunk (uncompressed frame) */

            if (chunk_size - 6 > (unsigned int)(s->avctx->width * s->avctx->height)*2) {

                av_log(avctx, AV_LOG_ERROR, "In chunk FLI_COPY : source data (%d bytes) " \

                       "bigger than image, skipping chunk\n", chunk_size - 6);

                stream_ptr += chunk_size - 6;

            } else {



                for (y_ptr = 0; y_ptr < s->frame.linesize[0] * s->avctx->height;

                     y_ptr += s->frame.linesize[0]) {



                    pixel_countdown = s->avctx->width;

                    pixel_ptr = 0;

                    while (pixel_countdown > 0) {

                      *((signed short*)(&pixels[y_ptr + pixel_ptr])) = LE_16(&buf[stream_ptr+pixel_ptr]);

                      pixel_ptr += 2;

                      pixel_countdown--;

                    }

                    stream_ptr += s->avctx->width*2;

                }

            }

            break;



        case FLI_MINI:

            /* some sort of a thumbnail? disregard this chunk... */

            stream_ptr += chunk_size - 6;

            break;



        default:

            av_log(avctx, AV_LOG_ERROR, "Unrecognized chunk type: %d\n", chunk_type);

            break;

        }



        frame_size -= chunk_size;

        num_chunks--;

    }



    /* by the end of the chunk, the stream ptr should equal the frame

     * size (minus 1, possibly); if it doesn't, issue a warning */

    if ((stream_ptr != buf_size) && (stream_ptr != buf_size - 1))

        av_log(avctx, AV_LOG_ERROR, "Processed FLI chunk where chunk size = %d " \

               "and final chunk ptr = %d\n", buf_size, stream_ptr);





    *data_size=sizeof(AVFrame);

    *(AVFrame*)data = s->frame;



    return buf_size;

}
