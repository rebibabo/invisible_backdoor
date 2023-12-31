static int local_name_to_path(FsContext *ctx, V9fsPath *dir_path,

                              const char *name, V9fsPath *target)

{

    if (ctx->export_flags & V9FS_SM_MAPPED_FILE &&

        local_is_mapped_file_metadata(ctx, name)) {

        errno = EINVAL;

        return -1;

    }



    if (dir_path) {

        v9fs_path_sprintf(target, "%s/%s", dir_path->data, name);

    } else if (strcmp(name, "/")) {

        v9fs_path_sprintf(target, "%s", name);

    } else {

        /* We want the path of the export root to be relative, otherwise

         * "*at()" syscalls would treat it as "/" in the host.

         */

        v9fs_path_sprintf(target, "%s", ".");

    }

    return 0;

}
