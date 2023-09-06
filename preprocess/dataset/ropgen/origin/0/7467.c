static int decode_type2(GetByteContext *gb, PutByteContext *pb)

{

    unsigned repeat = 0, first = 1, opcode;

    int i, len, pos;



    while (bytestream2_get_bytes_left(gb) > 0) {

        GetByteContext gbc;



        while (bytestream2_get_bytes_left(gb) > 0) {

            if (first) {

                first = 0;

                if (bytestream2_peek_byte(gb) > 17) {

                    len = bytestream2_get_byte(gb) - 17;

                    if (len < 4) {

                        do {

                            bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                            --len;

                        } while (len);

                        opcode = bytestream2_peek_byte(gb);

                        continue;

                    } else {

                        do {

                            bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                            --len;

                        } while (len);

                        opcode = bytestream2_peek_byte(gb);

                        if (opcode < 0x10) {

                            bytestream2_skip(gb, 1);

                            pos = - (opcode >> 2) - 4 * bytestream2_get_byte(gb) - 2049;



                            bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);

                            bytestream2_seek(&gbc, bytestream2_tell_p(pb) + pos, SEEK_SET);



                            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                            len = opcode & 3;

                            if (!len) {

                                repeat = 1;

                            } else {

                                do {

                                    bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                                    --len;

                                } while (len);

                                opcode = bytestream2_peek_byte(gb);

                            }

                            continue;

                        }

                    }

                    repeat = 0;

                }

                repeat = 1;

            }

            if (repeat) {

                repeat = 0;

                opcode = bytestream2_peek_byte(gb);

                if (opcode < 0x10) {

                    bytestream2_skip(gb, 1);

                    if (!opcode) {

                        if (!bytestream2_peek_byte(gb)) {

                            do {

                                bytestream2_skip(gb, 1);

                                opcode += 255;

                            } while (!bytestream2_peek_byte(gb) && bytestream2_get_bytes_left(gb) > 0);

                        }

                        opcode += bytestream2_get_byte(gb) + 15;

                    }

                    bytestream2_put_le32(pb, bytestream2_get_le32(gb));

                    for (i = opcode - 1; i > 0; --i)

                        bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                    opcode = bytestream2_peek_byte(gb);

                    if (opcode < 0x10) {

                        bytestream2_skip(gb, 1);

                        pos = - (opcode >> 2) - 4 * bytestream2_get_byte(gb) - 2049;



                        bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);

                        bytestream2_seek(&gbc, bytestream2_tell_p(pb) + pos, SEEK_SET);



                        bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                        bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                        bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                        len = opcode & 3;

                        if (!len) {

                            repeat = 1;

                        } else {

                            do {

                                bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                                --len;

                            } while (len);

                            opcode = bytestream2_peek_byte(gb);

                        }

                        continue;

                    }

                }

            }



            if (opcode >= 0x40) {

                bytestream2_skip(gb, 1);

                pos = - ((opcode >> 2) & 7) - 1 - 8 * bytestream2_get_byte(gb);

                len = (opcode >> 5) - 1;



                bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);

                bytestream2_seek(&gbc, bytestream2_tell_p(pb) + pos, SEEK_SET);



                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                do {

                    bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                    --len;

                } while (len);



                len = opcode & 3;



                if (!len) {

                    repeat = 1;

                } else {

                    do {

                        bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                        --len;

                    } while (len);

                    opcode = bytestream2_peek_byte(gb);

                }

                continue;

            } else if (opcode < 0x20) {

                break;

            }

            len = opcode & 0x1F;

            bytestream2_skip(gb, 1);

            if (!len) {

                if (!bytestream2_peek_byte(gb)) {

                    do {

                        bytestream2_skip(gb, 1);

                        len += 255;

                    } while (!bytestream2_peek_byte(gb) && bytestream2_get_bytes_left(gb) > 0);

                }

                len += bytestream2_get_byte(gb) + 31;

            }

            i = bytestream2_get_le16(gb);

            pos = - (i >> 2) - 1;



            bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);

            bytestream2_seek(&gbc, bytestream2_tell_p(pb) + pos, SEEK_SET);



            if (len < 6 || bytestream2_tell_p(pb) - bytestream2_tell(&gbc) < 4) {

                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                do {

                    bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                    --len;

                } while (len);

            } else {

                bytestream2_put_le32(pb, bytestream2_get_le32(&gbc));

                for (len = len - 2; len; --len)

                    bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            }

            len = i & 3;

            if (!len) {

                repeat = 1;

            } else {

                do {

                    bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                    --len;

                } while (len);

                opcode = bytestream2_peek_byte(gb);

            }

        }

        bytestream2_skip(gb, 1);

        if (opcode < 0x10) {

            pos = -(opcode >> 2) - 1 - 4 * bytestream2_get_byte(gb);



            bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);

            bytestream2_seek(&gbc, bytestream2_tell_p(pb) + pos, SEEK_SET);



            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            len = opcode & 3;

            if (!len) {

                repeat = 1;

            } else {

                do {

                    bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                    --len;

                } while (len);

                opcode = bytestream2_peek_byte(gb);

            }

            continue;

        }

        len = opcode & 7;

        if (!len) {

            if (!bytestream2_peek_byte(gb)) {

                do {

                    bytestream2_skip(gb, 1);

                    len += 255;

                } while (!bytestream2_peek_byte(gb) && bytestream2_get_bytes_left(gb) > 0);

            }

            len += bytestream2_get_byte(gb) + 7;

        }

        i = bytestream2_get_le16(gb);

        pos = bytestream2_tell_p(pb) - 2048 * (opcode & 8);

        pos = pos - (i >> 2);

        if (pos == bytestream2_tell_p(pb))

            break;



        pos = pos - 0x4000;

        bytestream2_init(&gbc, pb->buffer_start, pb->buffer_end - pb->buffer_start);

        bytestream2_seek(&gbc, pos, SEEK_SET);



        if (len < 6 || bytestream2_tell_p(pb) - bytestream2_tell(&gbc) < 4) {

            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

            do {

                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

                --len;

            } while (len);

        } else {

            bytestream2_put_le32(pb, bytestream2_get_le32(&gbc));

            for (len = len - 2; len; --len)

                bytestream2_put_byte(pb, bytestream2_get_byte(&gbc));

        }



        len = i & 3;

        if (!len) {

            repeat = 1;

        } else {

            do {

                bytestream2_put_byte(pb, bytestream2_get_byte(gb));

                --len;

            } while (len);

            opcode = bytestream2_peek_byte(gb);

        }

    }



    return 0;

}
