static void store_slice_c(uint8_t *dst, const uint16_t *src,

                          int dst_linesize, int src_linesize,

                          int width, int height, int log2_scale,

                          const uint8_t dither[8][8])

{

    int y, x;



#define STORE(pos) do {                                                     \

    temp = ((src[x + y*src_linesize + pos] << log2_scale) + d[pos]) >> 6;   \

    if (temp & 0x100)                                                       \

        temp = ~(temp >> 31);                                               \

    dst[x + y*dst_linesize + pos] = temp;                                   \

} while (0)



    for (y = 0; y < height; y++) {

        const uint8_t *d = dither[y];

        for (x = 0; x < width; x += 8) {

            int temp;

            STORE(0);

            STORE(1);

            STORE(2);

            STORE(3);

            STORE(4);

            STORE(5);

            STORE(6);

            STORE(7);

        }

    }

}
