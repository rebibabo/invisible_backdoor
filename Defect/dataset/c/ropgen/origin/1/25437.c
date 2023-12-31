static int vmdk_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVVmdkState *s = bs->opaque;

    uint32_t magic;

    int l1_size, i, ret;



    if (parent_open)

        // Parent must be opened as RO.

        flags = BDRV_O_RDONLY;

    fprintf(stderr, "(VMDK) image open: flags=0x%x filename=%s\n", flags, bs->filename);



    ret = bdrv_file_open(&s->hd, filename, flags | BDRV_O_AUTOGROW);

    if (ret < 0)

        return ret;

    if (bdrv_pread(s->hd, 0, &magic, sizeof(magic)) != sizeof(magic))

        goto fail;



    magic = be32_to_cpu(magic);

    if (magic == VMDK3_MAGIC) {

        VMDK3Header header;



        if (bdrv_pread(s->hd, sizeof(magic), &header, sizeof(header)) != sizeof(header))

            goto fail;

        s->cluster_sectors = le32_to_cpu(header.granularity);

        s->l2_size = 1 << 9;

        s->l1_size = 1 << 6;

        bs->total_sectors = le32_to_cpu(header.disk_sectors);

        s->l1_table_offset = le32_to_cpu(header.l1dir_offset) << 9;

        s->l1_backup_table_offset = 0;

        s->l1_entry_sectors = s->l2_size * s->cluster_sectors;

    } else if (magic == VMDK4_MAGIC) {

        VMDK4Header header;



        if (bdrv_pread(s->hd, sizeof(magic), &header, sizeof(header)) != sizeof(header))

            goto fail;

        bs->total_sectors = le64_to_cpu(header.capacity);

        s->cluster_sectors = le64_to_cpu(header.granularity);

        s->l2_size = le32_to_cpu(header.num_gtes_per_gte);

        s->l1_entry_sectors = s->l2_size * s->cluster_sectors;

        if (s->l1_entry_sectors <= 0)

            goto fail;

        s->l1_size = (bs->total_sectors + s->l1_entry_sectors - 1)

            / s->l1_entry_sectors;

        s->l1_table_offset = le64_to_cpu(header.rgd_offset) << 9;

        s->l1_backup_table_offset = le64_to_cpu(header.gd_offset) << 9;



        if (parent_open)

            s->is_parent = 1;

        else

            s->is_parent = 0;



        // try to open parent images, if exist

        if (vmdk_parent_open(bs, filename) != 0)

            goto fail;

        // write the CID once after the image creation

        s->parent_cid = vmdk_read_cid(bs,1);

    } else {

        goto fail;

    }



    /* read the L1 table */

    l1_size = s->l1_size * sizeof(uint32_t);

    s->l1_table = qemu_malloc(l1_size);

    if (!s->l1_table)

        goto fail;

    if (bdrv_pread(s->hd, s->l1_table_offset, s->l1_table, l1_size) != l1_size)

        goto fail;

    for(i = 0; i < s->l1_size; i++) {

        le32_to_cpus(&s->l1_table[i]);

    }



    if (s->l1_backup_table_offset) {

        s->l1_backup_table = qemu_malloc(l1_size);

        if (!s->l1_backup_table)

            goto fail;

        if (bdrv_pread(s->hd, s->l1_backup_table_offset, s->l1_backup_table, l1_size) != l1_size)

            goto fail;

        for(i = 0; i < s->l1_size; i++) {

            le32_to_cpus(&s->l1_backup_table[i]);

        }

    }



    s->l2_cache = qemu_malloc(s->l2_size * L2_CACHE_SIZE * sizeof(uint32_t));

    if (!s->l2_cache)

        goto fail;

    return 0;

 fail:

    qemu_free(s->l1_backup_table);

    qemu_free(s->l1_table);

    qemu_free(s->l2_cache);

    bdrv_delete(s->hd);

    return -1;

}
