static int mp_user_setxattr(FsContext *ctx, const char *path, const char *name,

                            void *value, size_t size, int flags)

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

    ret = lsetxattr(buffer, name, value, size, flags);

    g_free(buffer);

    return ret;

}
