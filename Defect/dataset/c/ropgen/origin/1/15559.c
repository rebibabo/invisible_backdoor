static int local_unlinkat(FsContext *ctx, V9fsPath *dir,

                          const char *name, int flags)

{

    int ret;

    V9fsString fullname;

    char *buffer;



    v9fs_string_init(&fullname);



    v9fs_string_sprintf(&fullname, "%s/%s", dir->data, name);

    if (ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        if (flags == AT_REMOVEDIR) {

            /*

             * If directory remove .virtfs_metadata contained in the

             * directory

             */

            buffer = g_strdup_printf("%s/%s/%s", ctx->fs_root,

                                     fullname.data, VIRTFS_META_DIR);

            ret = remove(buffer);

            g_free(buffer);

            if (ret < 0 && errno != ENOENT) {

                /*

                 * We didn't had the .virtfs_metadata file. May be file created

                 * in non-mapped mode ?. Ignore ENOENT.

                 */

                goto err_out;

            }

        }

        /*

         * Now remove the name from parent directory

         * .virtfs_metadata directory.

         */

        buffer = local_mapped_attr_path(ctx, fullname.data);

        ret = remove(buffer);

        g_free(buffer);

        if (ret < 0 && errno != ENOENT) {

            /*

             * We didn't had the .virtfs_metadata file. May be file created

             * in non-mapped mode ?. Ignore ENOENT.

             */

            goto err_out;

        }

    }

    /* Remove the name finally */

    buffer = rpath(ctx, fullname.data);

    ret = remove(buffer);

    g_free(buffer);



err_out:

    v9fs_string_free(&fullname);

    return ret;

}
