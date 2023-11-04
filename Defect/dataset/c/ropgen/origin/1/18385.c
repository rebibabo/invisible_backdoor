static int decode_cell_data(Cell *cell, uint8_t *block, uint8_t *ref_block,
                            int pitch, int h_zoom, int v_zoom, int mode,
                            const vqEntry *delta[2], int swap_quads[2],
                            const uint8_t **data_ptr, const uint8_t *last_ptr)
{
    int           x, y, line, num_lines;
    int           rle_blocks = 0;
    uint8_t       code, *dst, *ref;
    const vqEntry *delta_tab;
    unsigned int  dyad1, dyad2;
    uint64_t      pix64;
    int           skip_flag = 0, is_top_of_cell, is_first_row = 1;
    int           row_offset, blk_row_offset, line_offset;
    row_offset     =  pitch;
    blk_row_offset = (row_offset << (2 + v_zoom)) - (cell->width << 2);
    line_offset    = v_zoom ? row_offset : 0;
    for (y = 0; y < cell->height; is_first_row = 0, y += 1 + v_zoom) {
        for (x = 0; x < cell->width; x += 1 + h_zoom) {
            ref = ref_block;
            dst = block;
            if (rle_blocks > 0) {
                if (mode <= 4) {
                    RLE_BLOCK_COPY;
                } else if (mode == 10 && !cell->mv_ptr) {
                    RLE_BLOCK_COPY_8;
                }
                rle_blocks--;
            } else {
                for (line = 0; line < 4;) {
                    num_lines = 1;
                    is_top_of_cell = is_first_row && !line;
                    /* select primary VQ table for odd, secondary for even lines */
                    if (mode <= 4)
                        delta_tab = delta[line & 1];
                    else
                        delta_tab = delta[1];
                    BUFFER_PRECHECK;
                    code = bytestream_get_byte(data_ptr);
                    if (code < 248) {
                        if (code < delta_tab->num_dyads) {
                            BUFFER_PRECHECK;
                            dyad1 = bytestream_get_byte(data_ptr);
                            dyad2 = code;
                            if (dyad1 >= delta_tab->num_dyads || dyad1 >= 248)
                        } else {
                            /* process QUADS */
                            code -= delta_tab->num_dyads;
                            dyad1 = code / delta_tab->quad_exp;
                            dyad2 = code % delta_tab->quad_exp;
                            if (swap_quads[line & 1])
                                FFSWAP(unsigned int, dyad1, dyad2);
                        }
                        if (mode <= 4) {
                            APPLY_DELTA_4;
                        } else if (mode == 10 && !cell->mv_ptr) {
                            APPLY_DELTA_8;
                        } else {
                            APPLY_DELTA_1011_INTER;
                        }
                    } else {
                        /* process RLE codes */
                        switch (code) {
                        case RLE_ESC_FC:
                            skip_flag  = 0;
                            rle_blocks = 1;
                            code       = 253;
                            /* FALLTHROUGH */
                        case RLE_ESC_FF:
                        case RLE_ESC_FE:
                        case RLE_ESC_FD:
                            num_lines = 257 - code - line;
                            if (num_lines <= 0)
                                return IV3_BAD_RLE;
                            if (mode <= 4) {
                                RLE_LINES_COPY;
                            } else if (mode == 10 && !cell->mv_ptr) {
                                RLE_LINES_COPY_M10;
                            }
                            break;
                        case RLE_ESC_FB:
                            BUFFER_PRECHECK;
                            code = bytestream_get_byte(data_ptr);
                            rle_blocks = (code & 0x1F) - 1; /* set block counter */
                            if (code >= 64 || rle_blocks < 0)
                                return IV3_BAD_COUNTER;
                            skip_flag = code & 0x20;
                            num_lines = 4 - line; /* enforce next block processing */
                            if (mode >= 10 || (cell->mv_ptr || !skip_flag)) {
                                if (mode <= 4) {
                                    RLE_LINES_COPY;
                                } else if (mode == 10 && !cell->mv_ptr) {
                                    RLE_LINES_COPY_M10;
                                }
                            }
                            break;
                        case RLE_ESC_F9:
                            skip_flag  = 1;
                            rle_blocks = 1;
                            /* FALLTHROUGH */
                        case RLE_ESC_FA:
                            if (line)
                                return IV3_BAD_RLE;
                            num_lines = 4; /* enforce next block processing */
                            if (cell->mv_ptr) {
                                if (mode <= 4) {
                                    RLE_LINES_COPY;
                                } else if (mode == 10 && !cell->mv_ptr) {
                                    RLE_LINES_COPY_M10;
                                }
                            }
                            break;
                        default:
                            return IV3_UNSUPPORTED;
                        }
                    }
                    line += num_lines;
                    ref  += row_offset * (num_lines << v_zoom);
                    dst  += row_offset * (num_lines << v_zoom);
                }
            }
            /* move to next horizontal block */
            block     += 4 << h_zoom;
            ref_block += 4 << h_zoom;
        }
        /* move to next line of blocks */
        ref_block += blk_row_offset;
        block     += blk_row_offset;
    }
    return IV3_NOERR;
}