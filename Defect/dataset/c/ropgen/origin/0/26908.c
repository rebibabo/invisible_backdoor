int bdrv_read(BlockDriverState *bs, int64_t sector_num,

              uint8_t *buf, int nb_sectors)

{

    BlockDriver *drv = bs->drv;



    if (!drv)

        return -ENOMEDIUM;



    if (sector_num == 0 && bs->boot_sector_enabled && nb_sectors > 0) {

            memcpy(buf, bs->boot_sector_data, 512);

        sector_num++;

        nb_sectors--;

        buf += 512;

        if (nb_sectors == 0)

            return 0;

    }

    if (drv->bdrv_pread) {

        int ret, len;

        len = nb_sectors * 512;

        ret = drv->bdrv_pread(bs, sector_num * 512, buf, len);

        if (ret < 0)

            return ret;

        else if (ret != len)

            return -EINVAL;

        else {

	    bs->rd_bytes += (unsigned) len;

	    bs->rd_ops ++;

            return 0;

	}

    } else {

        return drv->bdrv_read(bs, sector_num, buf, nb_sectors);

    }

}
