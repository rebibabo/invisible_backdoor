static int local_chmod(FsContext *fs_ctx, V9fsPath *fs_path, FsCred *credp)

{

    char *buffer;

    int ret = -1;

    char *path = fs_path->data;



    if (fs_ctx->export_flags & V9FS_SM_MAPPED) {

        buffer = rpath(fs_ctx, path);

        ret = local_set_xattr(buffer, credp);

        g_free(buffer);

    } else if (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        return local_set_mapped_file_attr(fs_ctx, path, credp);

    } else if ((fs_ctx->export_flags & V9FS_SM_PASSTHROUGH) ||

               (fs_ctx->export_flags & V9FS_SM_NONE)) {

        buffer = rpath(fs_ctx, path);

        ret = chmod(buffer, credp->fc_mode);

        g_free(buffer);

    }

    return ret;

}
