static int local_utimensat(FsContext *s, V9fsPath *fs_path,

                           const struct timespec *buf)

{

    char *buffer;

    int ret;

    char *path = fs_path->data;



    buffer = rpath(s, path);

    ret = qemu_utimens(buffer, buf);

    g_free(buffer);

    return ret;

}
