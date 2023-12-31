void ff_ivi_recompose53(const IVIPlaneDesc *plane, uint8_t *dst,
                        const int dst_pitch, const int num_bands)
{
    int             x, y, indx;
    int32_t         p0, p1, p2, p3, tmp0, tmp1, tmp2;
    int32_t         b0_1, b0_2, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b2_4, b2_5, b2_6;
    int32_t         b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, b3_9;
    int32_t         pitch, back_pitch;
    const IDWTELEM *b0_ptr, *b1_ptr, *b2_ptr, *b3_ptr;
    /* all bands should have the same pitch */
    pitch = plane->bands[0].pitch;
    /* pixels at the position "y-1" will be set to pixels at the "y" for the 1st iteration */
    back_pitch = 0;
    /* get pointers to the wavelet bands */
    b0_ptr = plane->bands[0].buf;
    b1_ptr = plane->bands[1].buf;
    b2_ptr = plane->bands[2].buf;
    b3_ptr = plane->bands[3].buf;
    for (y = 0; y < plane->height; y += 2) {
        /* load storage variables with values */
        if (num_bands > 0) {
            b0_1 = b0_ptr[0];
            b0_2 = b0_ptr[pitch];
        }
        if (num_bands > 1) {
            b1_1 = b1_ptr[back_pitch];
            b1_2 = b1_ptr[0];
            b1_3 = b1_1 - b1_2*6 + b1_ptr[pitch];
        }
        if (num_bands > 2) {
            b2_2 = b2_ptr[0];     // b2[x,  y  ]
            b2_3 = b2_2;          // b2[x+1,y  ] = b2[x,y]
            b2_5 = b2_ptr[pitch]; // b2[x  ,y+1]
            b2_6 = b2_5;          // b2[x+1,y+1] = b2[x,y+1]
        }
        if (num_bands > 3) {
            b3_2 = b3_ptr[back_pitch]; // b3[x  ,y-1]
            b3_3 = b3_2;               // b3[x+1,y-1] = b3[x  ,y-1]
            b3_5 = b3_ptr[0];          // b3[x  ,y  ]
            b3_6 = b3_5;               // b3[x+1,y  ] = b3[x  ,y  ]
            b3_8 = b3_2 - b3_5*6 + b3_ptr[pitch];
            b3_9 = b3_8;
        }
        for (x = 0, indx = 0; x < plane->width; x+=2, indx++) {
            /* some values calculated in the previous iterations can */
            /* be reused in the next ones, so do appropriate copying */
            b2_1 = b2_2; // b2[x-1,y  ] = b2[x,  y  ]
            b2_2 = b2_3; // b2[x  ,y  ] = b2[x+1,y  ]
            b2_4 = b2_5; // b2[x-1,y+1] = b2[x  ,y+1]
            b2_5 = b2_6; // b2[x  ,y+1] = b2[x+1,y+1]
            b3_1 = b3_2; // b3[x-1,y-1] = b3[x  ,y-1]
            b3_2 = b3_3; // b3[x  ,y-1] = b3[x+1,y-1]
            b3_4 = b3_5; // b3[x-1,y  ] = b3[x  ,y  ]
            b3_5 = b3_6; // b3[x  ,y  ] = b3[x+1,y  ]
            b3_7 = b3_8; // vert_HPF(x-1)
            b3_8 = b3_9; // vert_HPF(x  )
            p0 = p1 = p2 = p3 = 0;
            /* process the LL-band by applying LPF both vertically and horizontally */
            if (num_bands > 0) {
                tmp0 = b0_1;
                tmp2 = b0_2;
                b0_1 = b0_ptr[indx+1];
                b0_2 = b0_ptr[pitch+indx+1];
                tmp1 = tmp0 + b0_1;
                p0 =  tmp0 << 4;
                p1 =  tmp1 << 3;
                p2 = (tmp0 + tmp2) << 3;
                p3 = (tmp1 + tmp2 + b0_2) << 2;
            }
            /* process the HL-band by applying HPF vertically and LPF horizontally */
            if (num_bands > 1) {
                tmp0 = b1_2;
                tmp1 = b1_1;
                b1_2 = b1_ptr[indx+1];
                b1_1 = b1_ptr[back_pitch+indx+1];
                tmp2 = tmp1 - tmp0*6 + b1_3;
                b1_3 = b1_1 - b1_2*6 + b1_ptr[pitch+indx+1];
                p0 += (tmp0 + tmp1) << 3;
                p1 += (tmp0 + tmp1 + b1_1 + b1_2) << 2;
                p2 +=  tmp2 << 2;
                p3 += (tmp2 + b1_3) << 1;
            }
            /* process the LH-band by applying LPF vertically and HPF horizontally */
            if (num_bands > 2) {
                b2_3 = b2_ptr[indx+1];
                b2_6 = b2_ptr[pitch+indx+1];
                tmp0 = b2_1 + b2_2;
                tmp1 = b2_1 - b2_2*6 + b2_3;
                p0 += tmp0 << 3;
                p1 += tmp1 << 2;
                p2 += (tmp0 + b2_4 + b2_5) << 2;
                p3 += (tmp1 + b2_4 - b2_5*6 + b2_6) << 1;
            }
            /* process the HH-band by applying HPF both vertically and horizontally */
            if (num_bands > 3) {
                b3_6 = b3_ptr[indx+1];            // b3[x+1,y  ]
                b3_3 = b3_ptr[back_pitch+indx+1]; // b3[x+1,y-1]
                tmp0 = b3_1 + b3_4;
                tmp1 = b3_2 + b3_5;
                tmp2 = b3_3 + b3_6;
                b3_9 = b3_3 - b3_6*6 + b3_ptr[pitch+indx+1];
                p0 += (tmp0 + tmp1) << 2;
                p1 += (tmp0 - tmp1*6 + tmp2) << 1;
                p2 += (b3_7 + b3_8) << 1;
                p3 +=  b3_7 - b3_8*6 + b3_9;
            }
            /* output four pixels */
            dst[x]             = av_clip_uint8((p0 >> 6) + 128);
            dst[x+1]           = av_clip_uint8((p1 >> 6) + 128);
            dst[dst_pitch+x]   = av_clip_uint8((p2 >> 6) + 128);
            dst[dst_pitch+x+1] = av_clip_uint8((p3 >> 6) + 128);
        }// for x
        dst += dst_pitch << 1;
        back_pitch = -pitch;
        b0_ptr += pitch;
        b1_ptr += pitch;
        b2_ptr += pitch;
        b3_ptr += pitch;
    }
}