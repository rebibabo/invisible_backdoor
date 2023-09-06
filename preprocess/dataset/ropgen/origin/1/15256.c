static CharDriverState *net_vhost_parse_chardev(const NetdevVhostUserOptions *opts)

{

    CharDriverState *chr = qemu_chr_find(opts->chardev);

    VhostUserChardevProps props;



    if (chr == NULL) {

        error_report("chardev \"%s\" not found", opts->chardev);

        return NULL;

    }



    /* inspect chardev opts */

    memset(&props, 0, sizeof(props));

    if (qemu_opt_foreach(chr->opts, net_vhost_chardev_opts, &props, NULL)) {

        return NULL;

    }



    if (!props.is_socket || !props.is_unix) {

        error_report("chardev \"%s\" is not a unix socket",

                     opts->chardev);

        return NULL;

    }



    qemu_chr_fe_claim_no_fail(chr);



    return chr;

}
