uint32_t do_arm_semihosting(CPUARMState *env)

{

    target_ulong args;

    char * s;

    int nr;

    uint32_t ret;

    uint32_t len;

#ifdef CONFIG_USER_ONLY

    TaskState *ts = env->opaque;

#else

    CPUARMState *ts = env;

#endif



    nr = env->regs[0];

    args = env->regs[1];

    switch (nr) {

    case TARGET_SYS_OPEN:

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

    case TARGET_SYS_CLOSE:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "close,%x", ARG(0));

            return env->regs[0];

        } else {

            return set_swi_errno(ts, close(ARG(0)));

        }

    case TARGET_SYS_WRITEC:

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

    case TARGET_SYS_WRITE0:

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

    case TARGET_SYS_WRITE:

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

    case TARGET_SYS_READ:

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

    case TARGET_SYS_READC:

       /* XXX: Read from debug console. Not implemented.  */

        return 0;

    case TARGET_SYS_ISTTY:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "isatty,%x", ARG(0));

            return env->regs[0];

        } else {

            return isatty(ARG(0));

        }

    case TARGET_SYS_SEEK:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(arm_semi_cb, "lseek,%x,%x,0", ARG(0), ARG(1));

            return env->regs[0];

        } else {

            ret = set_swi_errno(ts, lseek(ARG(0), ARG(1), SEEK_SET));

            if (ret == (uint32_t)-1)

              return -1;

            return 0;

        }

    case TARGET_SYS_FLEN:

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

    case TARGET_SYS_TMPNAM:

        /* XXX: Not implemented.  */

        return -1;

    case TARGET_SYS_REMOVE:

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

    case TARGET_SYS_RENAME:

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

    case TARGET_SYS_CLOCK:

        return clock() / (CLOCKS_PER_SEC / 100);

    case TARGET_SYS_TIME:

        return set_swi_errno(ts, time(NULL));

    case TARGET_SYS_SYSTEM:

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

    case TARGET_SYS_ERRNO:

#ifdef CONFIG_USER_ONLY

        return ts->swi_errno;

#else

        return syscall_err;

#endif

    case TARGET_SYS_GET_CMDLINE:

        {

            /* Build a command-line from the original argv.

             *

             * The inputs are:

             *     * ARG(0), pointer to a buffer of at least the size

             *               specified in ARG(1).

             *     * ARG(1), size of the buffer pointed to by ARG(0) in

             *               bytes.

             *

             * The outputs are:

             *     * ARG(0), pointer to null-terminated string of the

             *               command line.

             *     * ARG(1), length of the string pointed to by ARG(0).

             */



            char *output_buffer;

            size_t input_size = ARG(1);

            size_t output_size;

            int status = 0;



            /* Compute the size of the output string.  */

#if !defined(CONFIG_USER_ONLY)

            output_size = strlen(ts->boot_info->kernel_filename)

                        + 1  /* Separating space.  */

                        + strlen(ts->boot_info->kernel_cmdline)

                        + 1; /* Terminating null byte.  */

#else

            unsigned int i;



            output_size = ts->info->arg_end - ts->info->arg_start;

            if (!output_size) {

                /* We special-case the "empty command line" case (argc==0).

                   Just provide the terminating 0. */

                output_size = 1;

            }

#endif



            if (output_size > input_size) {

                 /* Not enough space to store command-line arguments.  */

                return -1;

            }



            /* Adjust the command-line length.  */

            SET_ARG(1, output_size - 1);



            /* Lock the buffer on the ARM side.  */

            output_buffer = lock_user(VERIFY_WRITE, ARG(0), output_size, 0);

            if (!output_buffer) {

                return -1;

            }



            /* Copy the command-line arguments.  */

#if !defined(CONFIG_USER_ONLY)

            pstrcpy(output_buffer, output_size, ts->boot_info->kernel_filename);

            pstrcat(output_buffer, output_size, " ");

            pstrcat(output_buffer, output_size, ts->boot_info->kernel_cmdline);

#else

            if (output_size == 1) {

                /* Empty command-line.  */

                output_buffer[0] = '\0';

                goto out;

            }



            if (copy_from_user(output_buffer, ts->info->arg_start,

                               output_size)) {

                status = -1;

                goto out;

            }



            /* Separate arguments by white spaces.  */

            for (i = 0; i < output_size - 1; i++) {

                if (output_buffer[i] == 0) {

                    output_buffer[i] = ' ';

                }

            }

        out:

#endif

            /* Unlock the buffer on the ARM side.  */

            unlock_user(output_buffer, ARG(0), output_size);



            return status;

        }

    case TARGET_SYS_HEAPINFO:

        {

            uint32_t *ptr;

            uint32_t limit;



#ifdef CONFIG_USER_ONLY

            /* Some C libraries assume the heap immediately follows .bss, so

               allocate it using sbrk.  */

            if (!ts->heap_limit) {

                abi_ulong ret;



                ts->heap_base = do_brk(0);

                limit = ts->heap_base + ARM_ANGEL_HEAP_SIZE;

                /* Try a big heap, and reduce the size if that fails.  */

                for (;;) {

                    ret = do_brk(limit);

                    if (ret >= limit) {

                        break;

                    }

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

    case TARGET_SYS_EXIT:

        gdb_exit(env, 0);

        exit(0);

    default:

        fprintf(stderr, "qemu: Unsupported SemiHosting SWI 0x%02x\n", nr);

        cpu_dump_state(env, stderr, fprintf, 0);

        abort();

    }

}
