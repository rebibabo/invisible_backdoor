static uint8_t *advance_line(uint8_t *start, uint8_t *line,

                             int stride, int *y, int h, int interleave)

{

    *y += interleave;



    if (*y < h) {

        return line + interleave * stride;

    } else {

        *y = (*y + 1) & (interleave - 1);

        if (*y) {

            return start + *y * stride;

        } else {

            return NULL;

        }

    }

}