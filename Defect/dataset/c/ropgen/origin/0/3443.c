static void do_change(const char *device, const char *target, const char *fmt)

{

    if (strcmp(device, "vnc") == 0) {

	do_change_vnc(target);

    } else {

	do_change_block(device, target, fmt);

    }

}
