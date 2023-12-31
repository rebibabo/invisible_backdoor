int ga_install_service(const char *path, const char *logfile,

                       const char *state_dir)

{

    int ret = EXIT_FAILURE;

    SC_HANDLE manager;

    SC_HANDLE service;

    TCHAR module_fname[MAX_PATH];

    GString *cmdline;

    SERVICE_DESCRIPTION desc = { (char *)QGA_SERVICE_DESCRIPTION };



    if (GetModuleFileName(NULL, module_fname, MAX_PATH) == 0) {

        printf_win_error("No full path to service's executable");

        return EXIT_FAILURE;

    }



    cmdline = g_string_new(module_fname);

    g_string_append(cmdline, " -d");



    if (path) {

        g_string_append_printf(cmdline, " -p %s", path);

    }

    if (logfile) {

        g_string_append_printf(cmdline, " -l %s -v", logfile);

    }

    if (state_dir) {

        g_string_append_printf(cmdline, " -t %s", state_dir);

    }



    g_debug("service's cmdline: %s", cmdline->str);



    manager = OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);

    if (manager == NULL) {

        printf_win_error("No handle to service control manager");

        goto out_strings;

    }



    service = CreateService(manager, QGA_SERVICE_NAME, QGA_SERVICE_DISPLAY_NAME,

        SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS, SERVICE_AUTO_START,

        SERVICE_ERROR_NORMAL, cmdline->str, NULL, NULL, NULL, NULL, NULL);

    if (service == NULL) {

        printf_win_error("Failed to install service");

        goto out_manager;

    }



    ChangeServiceConfig2(service, SERVICE_CONFIG_DESCRIPTION, &desc);

    fprintf(stderr, "Service was installed successfully.\n");

    ret = EXIT_SUCCESS;

    CloseServiceHandle(service);



out_manager:

    CloseServiceHandle(manager);



out_strings:

    g_string_free(cmdline, TRUE);

    return ret;

}
