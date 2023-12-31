uint32_t do_arm_semihosting(CPUState *env)

{

    target_ulong args;

    char * s;

    int nr;

    uint32_t ret;

    uint32_t len;

#ifdef CONFIG_USER_ONLY

    TaskState *ts = env->opaque;

#else

    CPUState *ts = env;

#endif



    nr = env->regs[0];

    args = env->regs[1];

    switch (nr) {

    case SYS_OPEN:

        if (!(s = lock_user_string(ARG(0))))

            /* FIXME - should this error code be -TARGET_EFAULT ? */

            return (uint32_t)-1;

        if (ARG(1) >= 12)

            return (uint32_t)-1;

        if (strcmp(s, ":tt") == 0) {

            if (ARG(1) < 4)

                return STDIN_FILENO;

            else

                return STDOUT_FILENO;

        }

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "open,%s,%x,1a4", ARG(0),

			   (int)ARG(2)+1, gdb_open_modeflags[ARG(1)]);

            return env->regs[0];

        } else {

            ret = set_swi_errno(ts, open(s, open_modeflags[ARG(1)], 0644));

        }

        unlock_user(s, ARG(0), 0);

        return ret;

    case SYS_CLOSE:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "close,%x", ARG(0));

            return env->regs[0];

        } else {

            return set_swi_errno(ts, close(ARG(0)));

        }

    case SYS_WRITEC:

        {

          char c;



          if (get_user_u8(c, args))

              /* FIXME - should this error code be -TARGET_EFAULT ? */

              return (uint32_t)-1;

          /* Write to debug console.  stderr is near enough.  */

          if (use_gdb_syscalls()) {

                gdb_do_syscall(arm_semi_cb, "write,2,%x,1", args);

                return env->regs[0];

          } else {

                return write(STDERR_FILENO, &c, 1);

          }

        }

    case SYS_WRITE0:

        if (!(s = lock_user_string(args)))

            /* FIXME - should this error code be -TARGET_EFAULT ? */

            return (uint32_t)-1;

        len = strlen(s);

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "write,2,%x,%x\n", args, len);

            ret = env->regs[0];

        } else {

            ret = write(STDERR_FILENO, s, len);

        }

        unlock_user(s, args, 0);

        return ret;

    case SYS_WRITE:

        len = ARG(2);

        if (use_gdb_syscalls()) {

            arm_semi_syscall_len = len;

            gdb_do_syscall(arm_semi_cb, "write,%x,%x,%x", ARG(0), ARG(1), len);

            return env->regs[0];

        } else {

            if (!(s = lock_user(VERIFY_READ, ARG(1), len, 1)))

                /* FIXME - should this error code be -TARGET_EFAULT ? */

                return (uint32_t)-1;

            ret = set_swi_errno(ts, write(ARG(0), s, len));

            unlock_user(s, ARG(1), 0);

            if (ret == (uint32_t)-1)

                return -1;

            return len - ret;

        }

    case SYS_READ:

        len = ARG(2);

        if (use_gdb_syscalls()) {

            arm_semi_syscall_len = len;

            gdb_do_syscall(arm_semi_cb, "read,%x,%x,%x", ARG(0), ARG(1), len);

            return env->regs[0];

        } else {

            if (!(s = lock_user(VERIFY_WRITE, ARG(1), len, 0)))

                /* FIXME - should this error code be -TARGET_EFAULT ? */

                return (uint32_t)-1;

            do

              ret = set_swi_errno(ts, read(ARG(0), s, len));

            while (ret == -1 && errno == EINTR);

            unlock_user(s, ARG(1), len);

            if (ret == (uint32_t)-1)

                return -1;

            return len - ret;

        }

    case SYS_READC:

       /* XXX: Read from debug cosole. Not implemented.  */

        return 0;

    case SYS_ISTTY:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "isatty,%x", ARG(0));

            return env->regs[0];

        } else {

            return isatty(ARG(0));

        }

    case SYS_SEEK:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "lseek,%x,%x,0", ARG(0), ARG(1));

            return env->regs[0];

        } else {

            ret = set_swi_errno(ts, lseek(ARG(0), ARG(1), SEEK_SET));

            if (ret == (uint32_t)-1)

              return -1;

            return 0;

        }

    case SYS_FLEN:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_flen_cb, "fstat,%x,%x",

			   ARG(0), env->regs[13]-64);

            return env->regs[0];

        } else {

            struct stat buf;

            ret = set_swi_errno(ts, fstat(ARG(0), &buf));

            if (ret == (uint32_t)-1)

                return -1;

            return buf.st_size;

        }

    case SYS_TMPNAM:

        /* XXX: Not implemented.  */

        return -1;

    case SYS_REMOVE:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "unlink,%s", ARG(0), (int)ARG(1)+1);

            ret = env->regs[0];

        } else {

            if (!(s = lock_user_string(ARG(0))))

                /* FIXME - should this error code be -TARGET_EFAULT ? */

                return (uint32_t)-1;

            ret =  set_swi_errno(ts, remove(s));

            unlock_user(s, ARG(0), 0);

        }

        return ret;

    case SYS_RENAME:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "rename,%s,%s",

                           ARG(0), (int)ARG(1)+1, ARG(2), (int)ARG(3)+1);

            return env->regs[0];

        } else {

            char *s2;

            s = lock_user_string(ARG(0));

            s2 = lock_user_string(ARG(2));

            if (!s || !s2)

                /* FIXME - should this error code be -TARGET_EFAULT ? */

                ret = (uint32_t)-1;

            else

                ret = set_swi_errno(ts, rename(s, s2));

            if (s2)

                unlock_user(s2, ARG(2), 0);

            if (s)

                unlock_user(s, ARG(0), 0);

            return ret;

        }

    case SYS_CLOCK:

        return clock() / (CLOCKS_PER_SEC / 100);

    case SYS_TIME:

        return set_swi_errno(ts, time(NULL));

    case SYS_SYSTEM:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "system,%s", ARG(0), (int)ARG(1)+1);

            return env->regs[0];

        } else {

            if (!(s = lock_user_string(ARG(0))))

                /* FIXME - should this error code be -TARGET_EFAULT ? */

                return (uint32_t)-1;

            ret = set_swi_errno(ts, system(s));

            unlock_user(s, ARG(0), 0);

            return ret;

        }

    case SYS_ERRNO:

#ifdef CONFIG_USER_ONLY

        return ts->swi_errno;

#else

        return syscall_err;

#endif

    case SYS_GET_CMDLINE:

#ifdef CONFIG_USER_ONLY

        /* Build a commandline from the original argv.  */

        {

            char *arm_cmdline_buffer;

            const char *host_cmdline_buffer;



            unsigned int i;

            unsigned int arm_cmdline_len = ARG(1);

            unsigned int host_cmdline_len =

                ts->info->arg_end-ts->info->arg_start;



            if (!arm_cmdline_len || host_cmdline_len > arm_cmdline_len) {

                return -1; /* not enough space to store command line */

            }



            if (!host_cmdline_len) {

                /* We special-case the "empty command line" case (argc==0).

                   Just provide the terminating 0. */

                arm_cmdline_buffer = lock_user(VERIFY_WRITE, ARG(0), 1, 0);

                arm_cmdline_buffer[0] = 0;

                unlock_user(arm_cmdline_buffer, ARG(0), 1);



                /* Adjust the commandline length argument. */

                SET_ARG(1, 0);

                return 0;

            }



            /* lock the buffers on the ARM side */

            arm_cmdline_buffer =

                lock_user(VERIFY_WRITE, ARG(0), host_cmdline_len, 0);

            host_cmdline_buffer =

                lock_user(VERIFY_READ, ts->info->arg_start,

                                       host_cmdline_len, 1);



            if (arm_cmdline_buffer && host_cmdline_buffer)

            {

                /* the last argument is zero-terminated;

                   no need for additional termination */

                memcpy(arm_cmdline_buffer, host_cmdline_buffer,

                       host_cmdline_len);



                /* separate arguments by white spaces */

                for (i = 0; i < host_cmdline_len-1; i++) {

                    if (arm_cmdline_buffer[i] == 0) {

                        arm_cmdline_buffer[i] = ' ';

                    }

                }



                /* Adjust the commandline length argument. */

                SET_ARG(1, host_cmdline_len-1);

            }



            /* Unlock the buffers on the ARM side.  */

            unlock_user(arm_cmdline_buffer, ARG(0), host_cmdline_len);

            unlock_user((void*)host_cmdline_buffer, ts->info->arg_start, 0);



            /* Return success if we could return a commandline.  */

            return (arm_cmdline_buffer && host_cmdline_buffer) ? 0 : -1;

        }

#else

        return -1;

#endif

    case SYS_HEAPINFO:

        {

            uint32_t *ptr;

            uint32_t limit;



#ifdef CONFIG_USER_ONLY

            /* Some C libraries assume the heap immediately follows .bss, so

               allocate it using sbrk.  */

            if (!ts->heap_limit) {

                long ret;



                ts->heap_base = do_brk(0);

                limit = ts->heap_base + ARM_ANGEL_HEAP_SIZE;

                /* Try a big heap, and reduce the size if that fails.  */

                for (;;) {

                    ret = do_brk(limit);

                    if (ret != -1)

                        break;

                    limit = (ts->heap_base >> 1) + (limit >> 1);

                }

                ts->heap_limit = limit;

            }



            if (!(ptr = lock_user(VERIFY_WRITE, ARG(0), 16, 0)))

                /* FIXME - should this error code be -TARGET_EFAULT ? */

                return (uint32_t)-1;

            ptr[0] = tswap32(ts->heap_base);

            ptr[1] = tswap32(ts->heap_limit);

            ptr[2] = tswap32(ts->stack_base);

            ptr[3] = tswap32(0); /* Stack limit.  */

            unlock_user(ptr, ARG(0), 16);

#else

            limit = ram_size;

            if (!(ptr = lock_user(VERIFY_WRITE, ARG(0), 16, 0)))

                /* FIXME - should this error code be -TARGET_EFAULT ? */

                return (uint32_t)-1;

            /* TODO: Make this use the limit of the loaded application.  */

            ptr[0] = tswap32(limit / 2);

            ptr[1] = tswap32(limit);

            ptr[2] = tswap32(limit); /* Stack base */

            ptr[3] = tswap32(0); /* Stack limit.  */

            unlock_user(ptr, ARG(0), 16);

#endif

            return 0;

        }

    case SYS_EXIT:

        gdb_exit(env, 0);

        exit(0);

    default:

        fprintf(stderr, "qemu: Unsupported SemiHosting SWI 0x%02x\n", nr);

        cpu_dump_state(env, stderr, fprintf, 0);

        abort();

    }

}
