static int bdrv_check_request(BlockDriverState *bs, int64_t sector_num,

                              int nb_sectors)

{

    if (nb_sectors < 0 || nb_sectors > BDRV_REQUEST_MAX_SECTORS) {

        return -EIO;

    }



    return bdrv_check_byte_request(bs, sector_num * BDRV_SECTOR_SIZE,

                                   nb_sectors * BDRV_SECTOR_SIZE);

}