static int slirp_smb(SlirpState* s, const char *exported_dir,
                     struct in_addr vserver_addr)
{
    static int instance;
    char smb_conf[128];
    char smb_cmdline[128];
    struct passwd *passwd;
    FILE *f;
    passwd = getpwuid(geteuid());
    if (!passwd) {
        error_report("failed to retrieve user name");
    snprintf(s->smb_dir, sizeof(s->smb_dir), "/tmp/qemu-smb.%ld-%d",
             (long)getpid(), instance++);
    if (mkdir(s->smb_dir, 0700) < 0) {
        error_report("could not create samba server dir '%s'", s->smb_dir);
    snprintf(smb_conf, sizeof(smb_conf), "%s/%s", s->smb_dir, "smb.conf");
    f = fopen(smb_conf, "w");
    if (!f) {
        slirp_smb_cleanup(s);
        error_report("could not create samba server configuration file '%s'",
                     smb_conf);
    fprintf(f,
            "[global]\n"
            "private dir=%s\n"
            "socket address=127.0.0.1\n"
            "pid directory=%s\n"
            "lock directory=%s\n"
            "state directory=%s\n"
            "log file=%s/log.smbd\n"
            "smb passwd file=%s/smbpasswd\n"
            "security = share\n"
            "[qemu]\n"
            "path=%s\n"
            "read only=no\n"
            "guest ok=yes\n"
            "force user=%s\n",
            s->smb_dir,
            s->smb_dir,
            s->smb_dir,
            s->smb_dir,
            s->smb_dir,
            s->smb_dir,
            exported_dir,
            passwd->pw_name
            );
    fclose(f);
    snprintf(smb_cmdline, sizeof(smb_cmdline), "%s -s %s",
             CONFIG_SMBD_COMMAND, smb_conf);
    if (slirp_add_exec(s->slirp, 0, smb_cmdline, &vserver_addr, 139) < 0) {
        slirp_smb_cleanup(s);
        error_report("conflicting/invalid smbserver address");
    return 0;