static int mp_user_removexattr(FsContext *ctx,

                               const char *path, const char *name)

{

    char *buffer;

    int ret;



    if (strncmp(name, "user.virtfs.", 12) == 0) {

        /*

         * Don't allow fetch of user.virtfs namesapce

         * in case of mapped security

         */

        errno = EACCES;

        return -1;

    }

    buffer = rpath(ctx, path);

    ret = lremovexattr(buffer, name);

    g_free(buffer);

    return ret;

}
