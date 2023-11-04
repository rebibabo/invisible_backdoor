static int qcow_open(BlockDriverState *bs, QDict *options, int flags,
                     Error **errp)
{
    BDRVQcowState *s = bs->opaque;
    int len, i, shift, ret;
    QCowHeader header;
    ret = bdrv_pread(bs->file, 0, &header, sizeof(header));
    if (ret < 0) {
    be32_to_cpus(&header.magic);
    be32_to_cpus(&header.version);
    be64_to_cpus(&header.backing_file_offset);
    be32_to_cpus(&header.backing_file_size);
    be32_to_cpus(&header.mtime);
    be64_to_cpus(&header.size);
    be32_to_cpus(&header.crypt_method);
    be64_to_cpus(&header.l1_table_offset);
    if (header.magic != QCOW_MAGIC) {
        error_setg(errp, "Image not in qcow format");
    if (header.version != QCOW_VERSION) {
        char version[64];
        snprintf(version, sizeof(version), "QCOW version %" PRIu32,
                 header.version);
        error_set(errp, QERR_UNKNOWN_BLOCK_FORMAT_FEATURE,
                  bs->device_name, "qcow", version);
        ret = -ENOTSUP;
    if (header.size <= 1) {
        error_setg(errp, "Image size is too small (must be at least 2 bytes)");
    if (header.cluster_bits < 9 || header.cluster_bits > 16) {
        error_setg(errp, "Cluster size must be between 512 and 64k");
    if (header.crypt_method > QCOW_CRYPT_AES) {
        error_setg(errp, "invalid encryption method in qcow header");
    s->crypt_method_header = header.crypt_method;
    if (s->crypt_method_header) {
        bs->encrypted = 1;
    s->cluster_bits = header.cluster_bits;
    s->cluster_size = 1 << s->cluster_bits;
    s->cluster_sectors = 1 << (s->cluster_bits - 9);
    s->l2_bits = header.l2_bits;
    s->l2_size = 1 << s->l2_bits;
    bs->total_sectors = header.size / 512;
    s->cluster_offset_mask = (1LL << (63 - s->cluster_bits)) - 1;
    /* read the level 1 table */
    shift = s->cluster_bits + s->l2_bits;
    s->l1_size = (header.size + (1LL << shift) - 1) >> shift;
    s->l1_table_offset = header.l1_table_offset;
    s->l1_table = g_malloc(s->l1_size * sizeof(uint64_t));
    ret = bdrv_pread(bs->file, s->l1_table_offset, s->l1_table,
               s->l1_size * sizeof(uint64_t));
    if (ret < 0) {
    for(i = 0;i < s->l1_size; i++) {
        be64_to_cpus(&s->l1_table[i]);
    /* alloc L2 cache */
    s->l2_cache = g_malloc(s->l2_size * L2_CACHE_SIZE * sizeof(uint64_t));
    s->cluster_cache = g_malloc(s->cluster_size);
    s->cluster_data = g_malloc(s->cluster_size);
    s->cluster_cache_offset = -1;
    /* read the backing file name */
    if (header.backing_file_offset != 0) {
        len = header.backing_file_size;
        if (len > 1023) {
            len = 1023;
        ret = bdrv_pread(bs->file, header.backing_file_offset,
                   bs->backing_file, len);
        if (ret < 0) {
        bs->backing_file[len] = '\0';
    /* Disable migration when qcow images are used */
    error_set(&s->migration_blocker,
              QERR_BLOCK_FORMAT_FEATURE_NOT_SUPPORTED,
              "qcow", bs->device_name, "live migration");
    migrate_add_blocker(s->migration_blocker);
    qemu_co_mutex_init(&s->lock);
    return 0;
 fail:
    g_free(s->l1_table);
    g_free(s->l2_cache);
    g_free(s->cluster_cache);
    g_free(s->cluster_data);
    return ret;