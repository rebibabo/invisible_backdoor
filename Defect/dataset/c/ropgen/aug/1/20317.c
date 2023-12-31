static int decode_dds1(uint8_t *frame, int width, int height,

                       const uint8_t *src, const uint8_t *src_end)

{

    const uint8_t *frame_start = frame;

    const uint8_t *frame_end   = frame + width * height;

    int mask = 0x10000, bitbuf = 0;

    int i, v, offset, count, segments;



    segments = bytestream_get_le16(&src);

    while (segments--) {

        if (mask == 0x10000) {

            if (src >= src_end)

                return -1;

            bitbuf = bytestream_get_le16(&src);

            mask = 1;

        }

        if (src_end - src < 2 || frame_end - frame < 2)

            return -1;

        if (bitbuf & mask) {

            v = bytestream_get_le16(&src);

            offset = (v & 0x1FFF) << 2;

            count = ((v >> 13) + 2) << 1;

            if (frame - frame_start < offset || frame_end - frame < count*2 + width)

                return -1;

            for (i = 0; i < count; i++) {

                frame[0] = frame[1] =

                frame[width] = frame[width + 1] = frame[-offset];



                frame += 2;

            }

        } else if (bitbuf & (mask << 1)) {

            frame += bytestream_get_le16(&src) * 2;

        } else {

            frame[0] = frame[1] =

            frame[width] = frame[width + 1] =  *src++;

            frame += 2;

            frame[0] = frame[1] =

            frame[width] = frame[width + 1] =  *src++;

            frame += 2;

        }

        mask <<= 2;

    }



    return 0;

}
