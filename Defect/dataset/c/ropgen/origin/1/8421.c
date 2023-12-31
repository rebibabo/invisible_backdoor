static ssize_t mp_user_getxattr(FsContext *ctx, const char *path,

                                const char *name, void *value, size_t size)

{

    char *buffer;

    ssize_t ret;



    if (strncmp(name, "user.virtfs.", 12) == 0) {

        /*

         * Don't allow fetch of user.virtfs namesapce

         * in case of mapped security

         */

        errno = ENOATTR;

        return -1;

    }

    buffer = rpath(ctx, path);

    ret = lgetxattr(buffer, name, value, size);

    g_free(buffer);

    return ret;

}
