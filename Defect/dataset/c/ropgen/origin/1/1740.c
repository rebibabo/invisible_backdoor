static int local_open(FsContext *ctx, V9fsPath *fs_path,

                      int flags, V9fsFidOpenState *fs)

{

    char *buffer;

    char *path = fs_path->data;

    int fd;



    buffer = rpath(ctx, path);

    fd = open(buffer, flags | O_NOFOLLOW);

    g_free(buffer);

    if (fd == -1) {

        return -1;

    }

    fs->fd = fd;

    return fs->fd;

}
