static inline int svq3_mc_dir(SVQ3Context *s, int size, int mode,

                              int dir, int avg)

{

    int i, j, k, mx, my, dx, dy, x, y;

    const int part_width    = ((size & 5) == 4) ? 4 : 16 >> (size & 1);

    const int part_height   = 16 >> ((unsigned)(size + 1) / 3);

    const int extra_width   = (mode == PREDICT_MODE) ? -16 * 6 : 0;

    const int h_edge_pos    = 6 * (s->h_edge_pos - part_width)  - extra_width;

    const int v_edge_pos    = 6 * (s->v_edge_pos - part_height) - extra_width;



    for (i = 0; i < 16; i += part_height)

        for (j = 0; j < 16; j += part_width) {

            const int b_xy = (4 * s->mb_x + (j >> 2)) +

                             (4 * s->mb_y + (i >> 2)) * s->b_stride;

            int dxy;

            x = 16 * s->mb_x + j;

            y = 16 * s->mb_y + i;

            k = (j >> 2 & 1) + (i >> 1 & 2) +

                (j >> 1 & 4) + (i      & 8);



            if (mode != PREDICT_MODE) {

                svq3_pred_motion(s, k, part_width >> 2, dir, 1, &mx, &my);

            } else {

                mx = s->next_pic->motion_val[0][b_xy][0] << 1;

                my = s->next_pic->motion_val[0][b_xy][1] << 1;



                if (dir == 0) {

                    mx = mx * s->frame_num_offset /

                         s->prev_frame_num_offset + 1 >> 1;

                    my = my * s->frame_num_offset /

                         s->prev_frame_num_offset + 1 >> 1;

                } else {

                    mx = mx * (s->frame_num_offset - s->prev_frame_num_offset) /

                         s->prev_frame_num_offset + 1 >> 1;

                    my = my * (s->frame_num_offset - s->prev_frame_num_offset) /

                         s->prev_frame_num_offset + 1 >> 1;

                }

            }



            /* clip motion vector prediction to frame border */

            mx = av_clip(mx, extra_width - 6 * x, h_edge_pos - 6 * x);

            my = av_clip(my, extra_width - 6 * y, v_edge_pos - 6 * y);



            /* get (optional) motion vector differential */

            if (mode == PREDICT_MODE) {

                dx = dy = 0;

            } else {

                dy = get_interleaved_se_golomb(&s->gb_slice);

                dx = get_interleaved_se_golomb(&s->gb_slice);



                if (dx != (int16_t)dx || dy != (int16_t)dy) {

                    av_log(s->avctx, AV_LOG_ERROR, "invalid MV vlc\n");

                    return -1;

                }

            }



            /* compute motion vector */

            if (mode == THIRDPEL_MODE) {

                int fx, fy;

                mx  = (mx + 1 >> 1) + dx;

                my  = (my + 1 >> 1) + dy;

                fx  = (unsigned)(mx + 0x30000) / 3 - 0x10000;

                fy  = (unsigned)(my + 0x30000) / 3 - 0x10000;

                dxy = (mx - 3 * fx) + 4 * (my - 3 * fy);



                svq3_mc_dir_part(s, x, y, part_width, part_height,

                                 fx, fy, dxy, 1, dir, avg);

                mx += mx;

                my += my;

            } else if (mode == HALFPEL_MODE || mode == PREDICT_MODE) {

                mx  = (unsigned)(mx + 1 + 0x30000) / 3 + dx - 0x10000;

                my  = (unsigned)(my + 1 + 0x30000) / 3 + dy - 0x10000;

                dxy = (mx & 1) + 2 * (my & 1);



                svq3_mc_dir_part(s, x, y, part_width, part_height,

                                 mx >> 1, my >> 1, dxy, 0, dir, avg);

                mx *= 3;

                my *= 3;

            } else {

                mx = (unsigned)(mx + 3 + 0x60000) / 6 + dx - 0x10000;

                my = (unsigned)(my + 3 + 0x60000) / 6 + dy - 0x10000;



                svq3_mc_dir_part(s, x, y, part_width, part_height,

                                 mx, my, 0, 0, dir, avg);

                mx *= 6;

                my *= 6;

            }



            /* update mv_cache */

            if (mode != PREDICT_MODE) {

                int32_t mv = pack16to32(mx, my);



                if (part_height == 8 && i < 8) {

                    AV_WN32A(s->mv_cache[dir][scan8[k] + 1 * 8], mv);



                    if (part_width == 8 && j < 8)

                        AV_WN32A(s->mv_cache[dir][scan8[k] + 1 + 1 * 8], mv);

                }

                if (part_width == 8 && j < 8)

                    AV_WN32A(s->mv_cache[dir][scan8[k] + 1], mv);

                if (part_width == 4 || part_height == 4)

                    AV_WN32A(s->mv_cache[dir][scan8[k]], mv);

            }



            /* write back motion vectors */

            fill_rectangle(s->cur_pic->motion_val[dir][b_xy],

                           part_width >> 2, part_height >> 2, s->b_stride,

                           pack16to32(mx, my), 4);

        }



    return 0;

}
