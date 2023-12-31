static void usb_mtp_object_readdir(MTPState *s, MTPObject *o)

{

    struct dirent *entry;

    DIR *dir;



    if (o->have_children) {

        return;

    }

    o->have_children = true;



    dir = opendir(o->path);

    if (!dir) {

        return;

    }

#ifdef __linux__

    int watchfd = usb_mtp_add_watch(s->inotifyfd, o->path);

    if (watchfd == -1) {

        fprintf(stderr, "usb-mtp: failed to add watch for %s\n", o->path);

    } else {

        trace_usb_mtp_inotify_event(s->dev.addr, o->path,

                                    0, "Watch Added");

        o->watchfd = watchfd;

    }

#endif

    while ((entry = readdir(dir)) != NULL) {

        usb_mtp_add_child(s, o, entry->d_name);

    }

    closedir(dir);

}
