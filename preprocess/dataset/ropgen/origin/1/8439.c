int scsi_build_sense(uint8_t *in_buf, int in_len,

                     uint8_t *buf, int len, bool fixed)

{

    bool fixed_in;

    SCSISense sense;

    if (!fixed && len < 8) {

        return 0;

    }



    if (in_len == 0) {

        sense.key = NO_SENSE;

        sense.asc = 0;

        sense.ascq = 0;

    } else {

        fixed_in = (in_buf[0] & 2) == 0;



        if (fixed == fixed_in) {

            memcpy(buf, in_buf, MIN(len, in_len));

            return MIN(len, in_len);

        }



        if (fixed_in) {

            sense.key = in_buf[2];

            sense.asc = in_buf[12];

            sense.ascq = in_buf[13];

        } else {

            sense.key = in_buf[1];

            sense.asc = in_buf[2];

            sense.ascq = in_buf[3];

        }

    }



    memset(buf, 0, len);

    if (fixed) {

        /* Return fixed format sense buffer */

        buf[0] = 0x70;

        buf[2] = sense.key;

        buf[7] = 10;

        buf[12] = sense.asc;

        buf[13] = sense.ascq;

        return MIN(len, 18);

    } else {

        /* Return descriptor format sense buffer */

        buf[0] = 0x72;

        buf[1] = sense.key;

        buf[2] = sense.asc;

        buf[3] = sense.ascq;

        return 8;

    }

}