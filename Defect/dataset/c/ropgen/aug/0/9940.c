static int decode_wdlt(GetByteContext *gb, uint8_t *frame, int width, int height)

{

    const uint8_t *frame_end   = frame + width * height;

    uint8_t *line_ptr;

    int count, i, v, lines, segments;



    lines = bytestream2_get_le16(gb);

    if (lines > height)

        return -1;



    while (lines--) {

        if (bytestream2_get_bytes_left(gb) < 2)

            return -1;

        segments = bytestream2_get_le16u(gb);

        while ((segments & 0xC000) == 0xC000) {

            unsigned delta = -((int16_t)segments * width);

            if (frame_end - frame <= delta)

                return -1;

            frame    += delta;

            segments = bytestream2_get_le16(gb);

        }

        if (segments & 0x8000) {

            frame[width - 1] = segments & 0xFF;

            segments = bytestream2_get_le16(gb);

        }

        line_ptr = frame;

        frame += width;

        while (segments--) {

            if (frame - line_ptr <= bytestream2_peek_byte(gb))

                return -1;

            line_ptr += bytestream2_get_byte(gb);

            count = (int8_t)bytestream2_get_byte(gb);

            if (count >= 0) {

                if (frame - line_ptr < count * 2)

                    return -1;

                if (bytestream2_get_buffer(gb, line_ptr, count * 2) != count * 2)

                    return -1;

                line_ptr += count * 2;

            } else {

                count = -count;

                if (frame - line_ptr < count * 2)

                    return -1;

                v = bytestream2_get_le16(gb);

                for (i = 0; i < count; i++)

                    bytestream_put_le16(&line_ptr, v);

            }

        }

    }



    return 0;

}
